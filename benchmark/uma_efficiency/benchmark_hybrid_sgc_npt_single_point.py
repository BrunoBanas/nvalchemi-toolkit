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
"""Wall-time / memory / energy benchmark for hybrid MC-NPT (SGC-NPT or Kawasaki-NPT).

Runs ``--n-walkers`` walkers (default 1) of an ``--n-atoms`` Au-Pt fcc cell at
one (temperature, delta_mu) state point for ``--n-blocks`` hybrid blocks and
writes metrics.json with wall time, peak GPU memory, final energies and a
per-phase timing breakdown (MC block, MD block, and the MC<->MD hand-off; see
TimedHybridMCMD). ``--sampler kawasaki`` swaps the MC half for Kawasaki with
the same NPT stage; ``--md-only`` skips MC entirely (pure NPT with the same
walkers and MD blocks), so the MD phase compares directly across all three.

Every hybrid-block parameter (CHECKPOINT, TASK, SPECIES, DT_FS,
MD_STEPS_PER_BLOCK, MC_STEP_FRACTION, THERMOSTAT_TIME_FS, BAROSTAT_TIME_FS)
except the inference settings is read from
benchmark/hybrid_sgc_npt/run_campaign.py, so this benchmark cannot drift from
the campaign's block definition.

``--inference-settings`` defaults to run_campaign.py's INFERENCE_SETTINGS
(``compile=false,merge_mole=false,tf32=true,activation_checkpointing=false``),
the recommended SGC-NPT spec; ``--mc-energy-only`` runs MC blocks energy-only.
For Kawasaki-NPT and pure NPT (fixed composition) merge_mole without compile
is faster -- see docs/userguide/dynamics_simulations.md.

    python benchmark/uma_efficiency/benchmark_hybrid_sgc_npt_single_point.py \
        --n-atoms 500 --n-walkers 2 --n-blocks 20 --temperature-k 1200.0 \
        --delta-mu-ev 0.5 --mc-energy-only --device cuda \
        --output sgc_npt/metrics.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

# run_campaign.py (workload builders, size tables) is in the sibling benchmark folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hybrid_sgc_npt"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_campaign as campaign  # noqa: E402 -- needs the sys.path insert above
from _uma_inference import (
    describe as describe_inference_settings,  # noqa: E402 -- sibling module
)
from _uma_inference import resolve as resolve_inference_settings  # noqa: E402

from nvalchemi.hybrid.scheduler import HybridMCMD  # noqa: E402
from nvalchemi.mc import Kawasaki  # noqa: E402
from nvalchemi.models.uma import UMAWrapper  # noqa: E402
from nvalchemi.scheduling import RunSpec  # noqa: E402


class TimedHybridMCMD(HybridMCMD):
    """HybridMCMD.run(), instrumented with per-phase wall-clock timers.

    Times each of the four calls HybridMCMD.run() already makes per block:
      - mc_seconds:             self.run_mc_block()   -- MC trial moves (+ energy-only re-baseline)
      - md_compute_seconds:     self.md.compute()     -- force/energy recompute
                                after MC, before MD resumes (so a rejected MC
                                move's stale derivatives never reach the integrator)
      - md_seconds:             self.md.run()         -- MD integration steps
      - mc_synchronize_seconds: self.mc.synchronize() -- MC-state resync after MD,
                                before the next block's MC trials

    handoff_seconds = md_compute_seconds + mc_synchronize_seconds: the two calls
    that exist only to keep MC and MD consistent with each other, as opposed to
    doing MC or MD work themselves -- this is the "hand-off" cost between the two
    stages. There is no separate host<->device transfer to time here: mc and md
    already share one GPU-resident batch (and here one model), so the hand-off is
    purely these two GPU-side calls.

    Every timer brackets a torch.cuda.synchronize() (skipped off-CUDA) so it
    reflects actual kernel completion, not just async launch latency -- without
    that, a cheap-looking phase could just be where a previous phase's queued GPU
    work happens to finish.
    """

    def __init__(self, *args, device=None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._timing_device = device
        self.block_timings: list[dict[str, float]] = []
        # --md-only: the same NPT stage, walkers and md_steps blocks, with no MC
        # at all -- an NPT-only baseline directly comparable with the hybrid's md phase.
        self.md_only = False

    def _sync(self) -> None:
        if self._timing_device is not None and self._timing_device.type == "cuda":
            torch.cuda.synchronize(self._timing_device)

    def run(self, batch, n_blocks: int):
        """Run *n_blocks* MC-MD blocks, timing the MC, hand-off and MD phases of each."""
        if n_blocks < 1:
            raise ValueError("n_blocks must be positive")
        if getattr(batch, "forces", None) is None:
            raise ValueError("hybrid MC-MD requires preallocated batch.forces")
        with self.md:
            t0 = time.perf_counter()
            self.md.compute(batch)
            self._sync()
            t1 = time.perf_counter()
            self.mc.synchronize(batch)
            self._sync()
            t2 = time.perf_counter()
            self.block_timings.append(
                {
                    "block": -1,  # warm-up: initial compute + synchronize, before block 0's MC
                    "md_compute_seconds": t1 - t0,
                    "mc_synchronize_seconds": t2 - t1,
                    "handoff_seconds": t2 - t0,
                }
            )

            for block in range(n_blocks):
                if self.md_only:
                    t0 = time.perf_counter()
                    self.md.run(batch, n_steps=self.md_steps)
                    self._sync()
                    self.block_timings.append(
                        {
                            "block": block,
                            "mc_seconds": 0.0,
                            "md_compute_seconds": 0.0,
                            "md_seconds": time.perf_counter() - t0,
                            "mc_synchronize_seconds": 0.0,
                            "handoff_seconds": 0.0,
                        }
                    )
                    continue
                t0 = time.perf_counter()
                # run_mc_block, not mc.run: applies mc_energy_only (its baseline
                # re-evaluation is timed as MC, where it belongs).
                self.run_mc_block(
                    batch
                )  # the samplers keep atomic_masses in step with species
                self._sync()
                t1 = time.perf_counter()

                self.md.compute(batch)
                self._sync()
                t2 = time.perf_counter()

                self.md.run(batch, n_steps=self.md_steps)
                self._sync()
                t3 = time.perf_counter()

                self.mc.synchronize(batch)
                self._sync()
                t4 = time.perf_counter()

                self.block_timings.append(
                    {
                        "block": block,
                        "mc_seconds": t1 - t0,
                        "md_compute_seconds": t2 - t1,
                        "md_seconds": t3 - t2,
                        "mc_synchronize_seconds": t4 - t3,
                        "handoff_seconds": (t2 - t1) + (t4 - t3),
                    }
                )
        return batch


def _phase_stats(values: list[float]) -> dict[str, float]:
    return {
        "mean_seconds": statistics.fmean(values),
        # Median = steady state: a one-off compile/recompile block skews the mean.
        "median_seconds": statistics.median(values),
        "std_seconds": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min_seconds": min(values),
        "max_seconds": max(values),
        "total_seconds": sum(values),
    }


def main() -> None:
    """Command-line entry point: time one hybrid SGC-NPT state point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n-atoms", type=int, default=2048, choices=sorted(campaign.SIZE_REPEATS)
    )
    parser.add_argument("--n-blocks", type=int, default=100)
    parser.add_argument("--temperature-k", type=float, default=1200.0)
    parser.add_argument(
        "--delta-mu-ev", type=float, default=0.5, help="mu(Pt) - mu(Au), eV"
    )
    parser.add_argument(
        "--inference-settings",
        default=campaign.INFERENCE_SETTINGS,
        help=(
            "fairchem UMA inference preset, or a key=value InferenceSettings spec (see "
            "_uma_inference.py). Defaults to run_campaign.py's own "
            f"INFERENCE_SETTINGS ({campaign.INFERENCE_SETTINGS!r}), the toolkit's "
            "recommended setting for SGC-NPT."
        ),
    )
    parser.add_argument(
        "--sampler",
        choices=["sgc", "kawasaki"],
        default="sgc",
        help="MC half of the hybrid: SGC transmutations (SGC-NPT, the default) or composition-"
        "conserving Kawasaki swaps (Kawasaki-NPT). NPT, the walker set and block schedule are the same.",
    )
    parser.add_argument("--kawasaki-cutoff-angstrom", type=float, default=3.40)
    parser.add_argument(
        "--species-blind-proposals",
        action="store_true",
        help="Kawasaki only: the older species-blind edge draw (unlike_pairs_only=False).",
    )
    parser.add_argument(
        "--md-only",
        action="store_true",
        help="Run only the NPT half (same NPT stage, walkers and --n-blocks x md_steps schedule), "
        "no MC: the NPT-only baseline for the hybrid's md phase.",
    )
    parser.add_argument(
        "--n-walkers",
        type=int,
        default=1,
        help="Independent walkers batched into one hybrid run (the batch width); walker i "
        "uses seed SEED + i, as in run_campaign.py.",
    )
    parser.add_argument(
        "--mc-energy-only",
        action="store_true",
        help="Energy-only UMA evaluation during MC blocks (no forces/stress autograd); MD keeps full outputs.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_hybrid_sgc_npt_single_point/metrics.json"),
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    process_start = time.perf_counter()

    model_load_start = time.perf_counter()
    inference_settings = resolve_inference_settings(args.inference_settings)
    model = UMAWrapper.from_checkpoint(
        campaign.CHECKPOINT,
        task_name=campaign.TASK,
        device=str(device),
        inference_settings=inference_settings,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - model_load_start

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

    run = RunSpec(
        run_id=f"benchmark.atoms{args.n_atoms}.T{args.temperature_k:g}.mu{args.delta_mu_ev:g}",
        temperature_k=args.temperature_k,
        pressure_ev_per_a3=campaign.PRESSURE_EV_PER_A3,
        chemical_potentials_ev={
            campaign.SPECIES[0]: 0.0,
            campaign.SPECIES[1]: args.delta_mu_ev,
        },
        species=campaign.SPECIES,
        batch_group="benchmark",
    )

    campaign.MC_ENERGY_ONLY = args.mc_energy_only  # make_workload reads it
    if args.n_walkers < 1:
        raise SystemExit("--n-walkers must be >= 1")
    runs = (
        tuple(
            RunSpec(
                run_id=f"{run.run_id}.w{index}",
                temperature_k=run.temperature_k,
                pressure_ev_per_a3=run.pressure_ev_per_a3,
                chemical_potentials_ev=run.chemical_potentials_ev,
                species=run.species,
                batch_group=run.batch_group,
            )
            for index in range(args.n_walkers)
        )
        if args.n_walkers > 1
        else (run,)
    )
    hybrid, batch = campaign.make_workload(
        model, template, runs, (None,) * len(runs), device
    )
    if args.sampler == "kawasaki":
        # Same NPT stage and walkers as the SGC workload; only the MC half changes.
        kawasaki = Kawasaki(
            model=hybrid.md.model,
            temperature=args.temperature_k,
            cutoff=args.kawasaki_cutoff_angstrom,
            unlike_pairs_only=not args.species_blind_proposals,
            random_seed=campaign.SEED,
        )
        hybrid = TimedHybridMCMD(
            mc=kawasaki,
            md=hybrid.md,
            mc_steps=hybrid.mc_steps,
            md_steps=hybrid.md_steps,
            mc_energy_only=args.mc_energy_only,
            device=device,
        )
    else:
        hybrid.__class__ = TimedHybridMCMD
        hybrid._timing_device = device
        hybrid.block_timings = []

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    hybrid.md_only = args.md_only
    run_start = time.perf_counter()
    result = hybrid.run(batch, n_blocks=args.n_blocks)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    run_seconds = time.perf_counter() - run_start
    process_seconds = time.perf_counter() - process_start

    block_records = [t for t in hybrid.block_timings if t["block"] >= 0]
    mc_times = [t["mc_seconds"] for t in block_records]
    md_times = [t["md_seconds"] for t in block_records]
    handoff_times = [t["handoff_seconds"] for t in block_records]
    md_compute_times = [t["md_compute_seconds"] for t in block_records]
    mc_sync_times = [t["mc_synchronize_seconds"] for t in block_records]
    phase_timing = {
        "mc": _phase_stats(mc_times),
        "md": _phase_stats(md_times),
        "handoff": _phase_stats(handoff_times),
        "handoff_md_compute": _phase_stats(md_compute_times),
        "handoff_mc_synchronize": _phase_stats(mc_sync_times),
        "warmup_handoff_seconds": hybrid.block_timings[0]["handoff_seconds"],
        "sum_of_phases_seconds": sum(mc_times) + sum(md_times) + sum(handoff_times),
    }

    final_energies_eV = [float(value) for value in result.energy.flatten()]
    final_energy_eV = sum(final_energies_eV) / len(
        final_energies_eV
    )  # mean over walkers
    n_atoms = int(result.num_nodes) // args.n_walkers  # per walker

    metrics = {
        "backend": "nvalchemi_toolkit",
        "n_atoms": n_atoms,
        "n_blocks": args.n_blocks,
        "temperature_K": args.temperature_k,
        "delta_mu_eV": args.delta_mu_ev,
        "species": list(campaign.SPECIES),
        "checkpoint": campaign.CHECKPOINT,
        "task_name": campaign.TASK,
        "inference_settings": args.inference_settings,
        "inference_settings_resolved": describe_inference_settings(inference_settings),
        "mc_energy_only": args.mc_energy_only,
        "md_steps_per_block": campaign.MD_STEPS_PER_BLOCK,
        "mc_step_fraction": campaign.MC_STEP_FRACTION,
        "dt_fs": campaign.DT_FS,
        "thermostat_time_fs": campaign.THERMOSTAT_TIME_FS,
        "barostat_time_fs": campaign.BAROSTAT_TIME_FS,
        "model_load_seconds": model_load_seconds,
        "hybrid_run_wall_seconds": run_seconds,
        "hybrid_run_wall_seconds_per_block": run_seconds / args.n_blocks,
        "process_wall_seconds_from_model_load": process_seconds,
        "mode": "md_only" if args.md_only else "hybrid",
        "sampler": None if args.md_only else args.sampler,
        "unlike_pairs_only": (not args.species_blind_proposals)
        if args.sampler == "kawasaki"
        else None,
        "n_walkers": args.n_walkers,
        "batch_width": args.n_walkers,
        "gpu_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else None,
        "gpu_total_memory_GB": (
            torch.cuda.get_device_properties(device).total_memory / 1024**3
            if device.type == "cuda"
            else None
        ),
        "final_energies_eV": final_energies_eV,
        "final_energy_eV": final_energy_eV,
        "final_energy_eV_per_atom": final_energy_eV / n_atoms,
        "mc_acceptance": hybrid.mc.stats.acceptance,
        "mc_accepted": hybrid.mc.stats.accepted,
        "mc_attempted": hybrid.mc.stats.attempted,
        "mc_step_count": hybrid.mc.step_count,
        "md_step_count": hybrid.md.step_count,
    }
    if device.type == "cuda":
        metrics["peak_gpu_memory_allocated_GB"] = (
            torch.cuda.max_memory_allocated(device) / 1024**3
        )
        metrics["peak_gpu_memory_reserved_GB"] = (
            torch.cuda.max_memory_reserved(device) / 1024**3
        )
    metrics["phase_timing"] = phase_timing

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    (args.output.parent / "block_timings.json").write_text(
        json.dumps(hybrid.block_timings, indent=2) + "\n"
    )
    print(json.dumps(metrics, indent=2))
    print(
        f"[benchmark] {n_atoms} atoms, {args.n_blocks} blocks: "
        f"hybrid.run={run_seconds:.2f}s, model_load={model_load_seconds:.2f}s, "
        f"final_energy={final_energy_eV:.4f} eV ({final_energy_eV / n_atoms:.6f} eV/atom)"
    )
    print(
        "[benchmark] phase timing (mean s/block): "
        f"mc={phase_timing['mc']['mean_seconds']:.3f}, "
        f"md={phase_timing['md']['mean_seconds']:.3f}, "
        f"handoff={phase_timing['handoff']['mean_seconds']:.3f} "
        f"(md_compute={phase_timing['handoff_md_compute']['mean_seconds']:.3f}, "
        f"mc_synchronize={phase_timing['handoff_mc_synchronize']['mean_seconds']:.3f})"
    )


if __name__ == "__main__":
    main()
