# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Batched pure-Kawasaki-MC benchmark for ``nvalchemi.mc.Kawasaki`` with UMA.

Runs W independent replicas of a periodic Au-Pt fcc cell as one ``Batch`` (no
HybridMCMD, no MD) and reports wall time, walker throughput and peak GPU
memory. Without ``--batch-width`` it profiles a width ladder with
``nvalchemi.scheduling.SimulationBatchPlanner`` and runs at the LARGEST width
whose peak reserved memory fits ``--target-memory-fraction`` (default 0.85):
fill the GPU. ``--batch-width N`` forces a width (the efficiency matrix does
this to compare settings at widths 1/2/4).

Initial compositions come from benchmark_npt_md_only_single_point.py's
``_build_state_matching_fcc_random_box()``, one composition seed per replica
(base seed + replica index). Velocities are zero (MC only).

``--inference-settings`` defaults to "turbo": a Kawasaki swap never changes a
replica's composition and MC-only never changes the graph shape, the one case
where both compile and merge_mole are valid. ``--energy-only`` skips the
forces/stress backward pass, which Monte Carlo never reads (see
_uma_inference.py). Kawasaki draws only unlike-species pairs
(``unlike_pairs_only=True``, its default), so reported acceptance is acceptance
among real swaps.

Diagnostics survive a time-limit kill: progress.jsonl, partial_metrics.json
and stacks.txt next to ``--output`` are kept current throughout (per-width
profile results as each finishes, one record per production block, a
heartbeat with GPU/host memory every ``--heartbeat-seconds``) -- see
_run_diagnostics.py. ``--profile-budget-seconds`` and
``--wall-budget-seconds`` end the sweep / production run early at a block
boundary so metrics.json is still written, flagged ``truncated``.

    python benchmark/uma_efficiency/benchmark_batched_pure_kawasaki.py \
        --n-atoms 500 --n-blocks 100 --temperature-k 1200.0 \
        --cutoff-angstrom 3.40 --batch-width 4 \
        --inference-settings turbo --energy-only --device cuda \
        --output kawasaki_mc/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch._dynamo
import torch._inductor.config

# run_campaign.py (workload builders, size tables) is in the sibling benchmark folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hybrid_sgc_npt"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_campaign as campaign  # noqa: E402 -- needs the sys.path insert above
from _run_diagnostics import (  # noqa: E402 -- sibling module, same directory
    MCBlockRunner,
    RunDiagnostics,
    profile_widths_incrementally,
    run_production_blocks,
)
from _uma_inference import describe as describe_inference_settings  # noqa: E402
from _uma_inference import resolve as resolve_inference_settings  # noqa: E402
from _uma_inference import restrict_to_energy_only  # noqa: E402
from benchmark_npt_md_only_single_point import (  # noqa: E402 -- sibling module, same directory
    _build_state_matching_fcc_random_box,
    _composition_fingerprint,
)

from nvalchemi.data import Batch  # noqa: E402
from nvalchemi.mc import Kawasaki  # noqa: E402
from nvalchemi.models.uma import UMAWrapper  # noqa: E402
from nvalchemi.scheduling import SimulationBatchPlanner  # noqa: E402

# A scaling probe, not a max-width search: three widths show how memory and wall
# time grow with batch width. Pick production widths with plan_run.py.
DEFAULT_CANDIDATE_WIDTHS = (1, 2, 4)


def _build_batch(
    template, width: int, base_seed: int, temperature_k: float, device: torch.device
) -> Batch:
    replicas = []
    for replica in range(width):
        data = _build_state_matching_fcc_random_box(
            template,
            composition_seed=base_seed + replica,
            velocity_seed=base_seed + replica,
            temperature_k=temperature_k,
            device=device,
        )
        data.velocities.zero_()  # irrelevant to a Monte-Carlo-only run
        replicas.append(data)
    return Batch.from_data_list(replicas)


def select_batch_width(
    model,
    template,
    *,
    cutoff_angstrom: float,
    mc_steps_per_block: int,
    temperature_k: float,
    base_seed: int,
    candidate_widths: tuple[int, ...],
    warmup_blocks: int,
    measured_blocks: int,
    target_memory_fraction: float,
    device: torch.device,
    diag: RunDiagnostics,
    profile_budget_seconds: float | None,
    unlike_pairs_only: bool = True,
) -> tuple[int, list[dict], str | None]:
    """Profile ascending widths one at a time (see profile_widths_incrementally);
    returns the selected width, the profile, and why the sweep stopped early."""
    planner = SimulationBatchPlanner(memory_fraction=target_memory_fraction)
    total_memory_bytes = torch.cuda.get_device_properties(device).total_memory
    memory_limit_bytes = int(total_memory_bytes * target_memory_fraction)
    call_labels = ("warmup", "measured") if warmup_blocks else ("measured",)

    def workload_factory(width: int, dev: torch.device):
        # Static-shape compile recompiles per width; drop the previous width's graphs
        # so they neither hold memory nor exhaust dynamo's recompile limit (8).
        torch._dynamo.reset()
        batch = _build_batch(template, width, base_seed, temperature_k, dev)
        sampler = Kawasaki(
            model=model,
            temperature=temperature_k,
            cutoff=cutoff_angstrom,
            unlike_pairs_only=unlike_pairs_only,
            random_seed=campaign.SEED,
        )
        runner = MCBlockRunner(
            sampler,
            mc_steps_per_block,
            diag,
            "profile_blocks",
            tracked_z=campaign.SPECIES[1],
            call_labels=call_labels,
            batch_width=width,
        )
        return runner, batch

    measurements, stop_reason = profile_widths_incrementally(
        planner,
        workload_factory,
        candidate_widths,
        device=device,
        warmup_blocks=warmup_blocks,
        measured_blocks=measured_blocks,
        target_memory_fraction=target_memory_fraction,
        diag=diag,
        profile_budget_seconds=profile_budget_seconds,
    )
    eligible = [
        measurement
        for measurement in measurements
        if measurement.status == "ok"
        and measurement.peak_reserved_bytes is not None
        and measurement.peak_reserved_bytes <= memory_limit_bytes
    ]
    if not eligible:
        raise RuntimeError(
            f"no candidate width fit within {target_memory_fraction:.0%} of device memory; "
            f"measurements={measurements}"
        )
    selected_width = max(measurement.batch_width for measurement in eligible)
    profile = [
        {
            "batch_width": measurement.batch_width,
            "status": measurement.status,
            "walker_blocks_per_second": measurement.walker_blocks_per_second,
            "peak_reserved_bytes": measurement.peak_reserved_bytes,
            "peak_reserved_GiB": (measurement.peak_reserved_bytes / 1024**3)
            if measurement.peak_reserved_bytes
            else None,
            "error": measurement.error,
        }
        for measurement in measurements
    ]
    return selected_width, profile, stop_reason


def main() -> None:
    """Command-line entry point: profile batched Kawasaki MC across batch widths."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n-atoms", type=int, default=500, choices=sorted(campaign.SIZE_REPEATS)
    )
    parser.add_argument("--n-blocks", type=int, default=100)
    parser.add_argument(
        "--mc-step-fraction", type=float, default=campaign.MC_STEP_FRACTION
    )
    parser.add_argument("--temperature-k", type=float, default=1200.0)
    parser.add_argument("--cutoff-angstrom", type=float, default=3.40)
    parser.add_argument("--composition-seed", type=int, default=2026090102)
    parser.add_argument(
        "--batch-width",
        default="auto",
        help="Replica count to use, or 'auto' (default) to sweep and pick the largest width that fits the GPU.",
    )
    parser.add_argument(
        "--candidate-widths",
        default=",".join(str(w) for w in DEFAULT_CANDIDATE_WIDTHS),
        help="Comma-separated ascending widths to try in 'auto' mode.",
    )
    parser.add_argument("--target-memory-fraction", type=float, default=0.85)
    parser.add_argument("--profile-warmup-blocks", type=int, default=1)
    parser.add_argument("--profile-measured-blocks", type=int, default=2)
    parser.add_argument(
        "--inference-settings",
        default="turbo",
        help="A fairchem preset (default/turbo/batch) or a comma-separated key=value spec of "
        "InferenceSettings fields, e.g. 'compile=true,tf32=true,activation_checkpointing=false'.",
    )
    parser.add_argument(
        "--species-blind-proposals",
        action="store_true",
        help="Use Kawasaki's older species-blind edge draw (unlike_pairs_only=False), which also "
        "spends model calls on same-species no-ops. For A/B against the default unlike-pair draw.",
    )
    parser.add_argument(
        "--energy-only",
        action="store_true",
        help="Stop UMA computing forces/stress, removing the autograd backward pass every MC step "
        "pays and discards. MC-only runs never need forces; never use this with MD/NPT in the loop.",
    )
    parser.add_argument(
        "--inductor-shape-padding",
        action="store_true",
        help="Keep inductor's pad_mm pass on. Off by default: at large widths it benchmarks real padded "
        "copies of the batched bmm operands during compilation (~20 GiB at once), which OOMs a 40 GB A100 "
        "at widths whose steady-state run fits.",
    )
    parser.add_argument(
        "--profile-budget-seconds",
        type=float,
        default=None,
        help="Stop the width sweep (at a width boundary) once it has run this long; widest fitting width so far wins.",
    )
    parser.add_argument(
        "--wall-budget-seconds",
        type=float,
        default=None,
        help="Stop production (at a block boundary) once the process has run this long, still writing metrics.json. "
        "Set below the Slurm time limit.",
    )
    parser.add_argument("--heartbeat-seconds", type=float, default=60.0)
    parser.add_argument(
        "--stack-dump-seconds",
        type=float,
        default=1800.0,
        help="Interval for periodic all-thread stack dumps to stacks.txt (0 disables).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_batched_kawasaki_mc_only/metrics.json"),
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit(
            "this benchmark requires CUDA (SimulationBatchPlanner.profile() requires a GPU)"
        )

    torch._inductor.config.shape_padding = args.inductor_shape_padding

    process_start = time.perf_counter()
    diag = RunDiagnostics(
        args.output.parent,
        device,
        heartbeat_seconds=args.heartbeat_seconds,
        stack_dump_seconds=args.stack_dump_seconds,
    )
    diag.update(args=vars(args))
    try:
        _run(args, device, process_start, diag)
    except BaseException as error:
        diag.update(error=repr(error))
        diag.close("failed")
        raise
    diag.close("completed")


def _run(
    args, device: torch.device, process_start: float, diag: RunDiagnostics
) -> None:
    repeats = campaign.SIZE_REPEATS[args.n_atoms]
    template = campaign.build_ase_structure(
        campaign.TEMPLATE_SYMBOL,
        campaign.CRYSTAL_STRUCTURE,
        campaign.LATTICE_A_ANG,
        repeats,
        cubic=campaign.CONVENTIONAL_CELL,
    )
    if len(template) != args.n_atoms:
        raise ValueError(
            f"expected {args.n_atoms} atoms, repeats={repeats} built {len(template)}"
        )

    diag.set_phase("model_load")
    model_load_start = time.perf_counter()
    inference_settings = resolve_inference_settings(args.inference_settings)
    model = UMAWrapper.from_checkpoint(
        campaign.CHECKPOINT,
        task_name=campaign.TASK,
        device=str(device),
        inference_settings=inference_settings,
    )
    energy_only = {"applied": False}
    if args.energy_only:
        energy_only = restrict_to_energy_only(model)
        diag.event("energy_only", **energy_only)
    diag.update(
        energy_only=energy_only,
        inference_settings_resolved=describe_inference_settings(inference_settings),
    )
    torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - model_load_start
    diag.update(model_load_seconds=model_load_seconds)

    mc_steps_per_block = max(1, round(args.mc_step_fraction * args.n_atoms))

    if args.batch_width == "auto":
        candidate_widths = tuple(
            int(value) for value in args.candidate_widths.split(",")
        )
        selected_width, width_profile, profile_stop_reason = select_batch_width(
            model,
            template,
            cutoff_angstrom=args.cutoff_angstrom,
            mc_steps_per_block=mc_steps_per_block,
            temperature_k=args.temperature_k,
            base_seed=args.composition_seed,
            candidate_widths=candidate_widths,
            warmup_blocks=args.profile_warmup_blocks,
            measured_blocks=args.profile_measured_blocks,
            target_memory_fraction=args.target_memory_fraction,
            device=device,
            diag=diag,
            profile_budget_seconds=args.profile_budget_seconds,
            unlike_pairs_only=not args.species_blind_proposals,
        )
    else:
        selected_width = int(args.batch_width)
        width_profile = []
        profile_stop_reason = None
    diag.update(selected_width=selected_width, profile_stop_reason=profile_stop_reason)

    # Fresh batch for the timed production run -- never reuse a
    # profiling-sweep batch, which already ran real MC trials.
    diag.set_phase("production_setup", batch_width=selected_width)
    torch._dynamo.reset()
    batch = _build_batch(
        template, selected_width, args.composition_seed, args.temperature_k, device
    )
    composition_fingerprints = [
        _composition_fingerprint(replica.atomic_numbers)
        for replica in batch.to_data_list()
    ]
    sampler = Kawasaki(
        model=model,
        temperature=args.temperature_k,
        cutoff=args.cutoff_angstrom,
        unlike_pairs_only=not args.species_blind_proposals,
        random_seed=campaign.SEED,
    )

    runner = MCBlockRunner(
        sampler,
        mc_steps_per_block,
        diag,
        "production_blocks",
        tracked_z=campaign.SPECIES[1],
        batch_width=selected_width,
    )
    result, blocks_completed, truncated_reason, run_seconds = run_production_blocks(
        runner,
        batch,
        args.n_blocks,
        process_start=process_start,
        wall_budget_seconds=args.wall_budget_seconds,
        diag=diag,
    )
    total_steps = blocks_completed * mc_steps_per_block
    diag.set_phase("write_metrics")
    process_seconds = time.perf_counter() - process_start

    n_atoms = result.num_nodes // selected_width
    final_energies_eV = (
        result.energy.flatten().tolist()
        if getattr(result, "energy", None) is not None
        else []
    )
    stats = sampler.stats

    metrics = {
        "backend": "nvalchemi_toolkit",
        "scope": "Batched MC-only: nvalchemi.mc.Kawasaki run directly over a batch of replicas, no HybridMCMD/NPT in the loop",
        "sampler": "kawasaki",
        "n_atoms_per_walker": n_atoms,
        "batch_width": selected_width,
        "batch_width_auto_selected": args.batch_width == "auto",
        "target_memory_fraction": args.target_memory_fraction,
        "batch_width_profile": width_profile,
        "n_blocks": args.n_blocks,
        "n_blocks_completed": blocks_completed,
        "truncated": truncated_reason is not None,
        "truncated_reason": truncated_reason,
        "profile_stop_reason": profile_stop_reason,
        "mc_step_fraction": args.mc_step_fraction,
        "mc_steps_per_block": mc_steps_per_block,
        "total_mc_steps": total_steps,
        "temperature_K": args.temperature_k,
        "neighbor_cutoff_angstrom": args.cutoff_angstrom,
        "species": list(campaign.SPECIES),
        "checkpoint": campaign.CHECKPOINT,
        "task_name": campaign.TASK,
        "inference_settings": args.inference_settings,
        "inference_settings_resolved": describe_inference_settings(inference_settings),
        "energy_only": energy_only,
        "unlike_pairs_only": not args.species_blind_proposals,
        "inductor_shape_padding": args.inductor_shape_padding,
        "composition_fingerprints": composition_fingerprints,
        "composition_seed_base": args.composition_seed,
        "model_load_seconds": model_load_seconds,
        "mc_run_wall_seconds": run_seconds,
        "aggregate_walker_blocks_per_second": selected_width
        * blocks_completed
        / run_seconds
        if blocks_completed
        else None,
        "process_wall_seconds_from_model_load": process_seconds,
        "final_energies_eV": final_energies_eV,
        "final_energy_eV_per_atom_mean": (
            sum(final_energies_eV) / len(final_energies_eV) / n_atoms
            if final_energies_eV
            else None
        ),
        "production_block_records": diag.state.get("production_blocks", []),
        "mc_attempted": stats.attempted,
        "mc_accepted": stats.accepted,
        "mc_acceptance": stats.acceptance,
        "gpu_total_memory_GB": torch.cuda.get_device_properties(device).total_memory
        / 1024**3,
        "peak_gpu_memory_allocated_GB": torch.cuda.max_memory_allocated(device)
        / 1024**3,
        "peak_gpu_memory_reserved_GB": torch.cuda.max_memory_reserved(device) / 1024**3,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
    print(
        f"[benchmark] width={selected_width} replicas x {n_atoms} atoms, {blocks_completed}/{args.n_blocks} blocks: "
        f"mc_run={run_seconds:.2f}s, model_load={model_load_seconds:.2f}s, "
        f"aggregate_walker_blocks_per_second={metrics['aggregate_walker_blocks_per_second']}, "
        f"peak_gpu_reserved={metrics['peak_gpu_memory_reserved_GB']:.2f} GB"
    )


if __name__ == "__main__":
    main()
