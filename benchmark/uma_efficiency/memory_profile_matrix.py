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
"""GPU memory-vs-batch-width profiling matrix for UMA workloads.

Measures peak GPU memory as a function of batch width (independent walkers
sharing one model call) for every combination of

  kernel              in {npt, nvt, kawasaki, sgc, hybrid_sgc_npt}
  n_atoms             in run_campaign.SIZE_REPEATS (500 / 1372 / 2048 by default)
  inference_settings  one preset or key=value spec per invocation

npt/nvt/kawasaki keep each walker's composition fixed; sgc/hybrid_sgc_npt
change it every step, so they must not use merge_mole or compile (see
SGC_KERNELS / _sgc_safe below and "UMA settings for SGC and SGC-NPT" in
docs/userguide/dynamics_simulations.md). An invocation whose settings enable
either skips those kernels rather than measuring a setting nobody should run.

This is a memory characterization, not a physics run: MD and MC steps per block
default to 10, because peak memory is reached within the first few steps.

Uses ``nvalchemi.scheduling.SimulationBatchPlanner`` -- the profiler
run_campaign.py's ``select_batch_width()`` uses -- so the numbers match the
widths real campaigns select. ``_BlockAdapter`` adapts
``BaseDynamics.run(batch, n_steps=...)`` to the planner's
``runner.run(batch, n_blocks=...)`` contract; HybridMCMD implements it natively.

One invocation profiles one inference setting (the model load is the expensive
fixed cost) across every requested kernel x n_atoms, each swept over
run_campaign.BATCH_WIDTH_CANDIDATES[n_atoms] by default.

Output (under --output-dir):
  memory_profile_matrix.json -- by kernel then n_atoms: each width's status,
                                peak reserved GB, walker throughput, and the
                                fitted linear model (resident_GB +
                                per_walker_GB) where at least 2 widths ran.
  memory_profile_matrix.csv  -- the same, one row per (kernel, n_atoms, width).

    python benchmark/uma_efficiency/memory_profile_matrix.py \
        --inference-settings batch \
        --kernels npt nvt kawasaki sgc hybrid_sgc_npt \
        --n-atoms-list 500 1372 2048 \
        --device cuda --output-dir memory_profile_matrix/batch
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

# run_campaign.py (workload builders, size tables) is in the sibling benchmark folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hybrid_sgc_npt"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_campaign as campaign  # noqa: E402 -- needs the sys.path insert above
from _uma_inference import (
    resolve as resolve_inference_settings,  # noqa: E402 -- sibling module
)
from benchmark_npt_md_only_single_point import (
    _build_state_matching_fcc_random_box,  # noqa: E402
)

from nvalchemi.data import Batch  # noqa: E402
from nvalchemi.dynamics.integrators.npt import NPT  # noqa: E402
from nvalchemi.dynamics.integrators.nvt_nose_hoover import NVTNoseHoover  # noqa: E402
from nvalchemi.hybrid import HybridMCMD  # noqa: E402
from nvalchemi.mc import SGC, Kawasaki  # noqa: E402
from nvalchemi.models.uma import UMAWrapper  # noqa: E402
from nvalchemi.scheduling import SimulationBatchPlanner  # noqa: E402

KERNELS = ("npt", "nvt", "kawasaki", "sgc", "hybrid_sgc_npt")
# Kernels that never change atomic_numbers (composition-consistent) vs. ones
# that do every step (composition-inconsistent) -- see docs/userguide/dynamics_simulations.md.
COMPOSITION_CONSISTENT = {
    "npt": True,
    "nvt": True,
    "kawasaki": False,
    "sgc": False,
    "hybrid_sgc_npt": False,
}
# SGC changes atomic composition every step, so SGC-containing kernels must not
# run with merge_mole (it assumes a fixed composition; "default"/"turbo" enable
# it, and showed ~22 GB vs ~7 GB reserved at 2048 atoms with memory still
# growing) or compile. Anything else is allowed -- "batch", or run_campaign.py's
# INFERENCE_SETTINGS spec (merge/compile off, checkpointing off, tf32 on; see
# the toolkit's "UMA settings for SGC and SGC-NPT").
# A disallowed request skips the kernel (not an error) rather than producing a
# number nobody should run with.
SGC_KERNELS = {"sgc", "hybrid_sgc_npt"}


def _sgc_safe(settings) -> bool:
    """True unless the settings merge MoLE weights or compile (both SGC-unsafe)."""
    if isinstance(settings, str):
        return settings == "batch"  # the other presets enable merge_mole + compile
    return not (
        getattr(settings, "merge_mole", False) or getattr(settings, "compile", False)
    )


class _BlockAdapter:
    """Adapts BaseDynamics.run(batch, n_steps=...) to runner.run(batch, n_blocks=...)."""

    def __init__(self, dynamics, steps_per_block: int) -> None:
        self.dynamics = dynamics
        self.steps_per_block = steps_per_block

    def run(self, batch: Batch, n_blocks: int) -> Batch:
        """Run *n_blocks* blocks of ``steps_per_block`` dynamics steps."""
        for _ in range(n_blocks):
            self.dynamics.run(batch, n_steps=self.steps_per_block)
        return batch


def _build_batch(
    template, temperature_k: float, width: int, device: torch.device
) -> Batch:
    return Batch.from_data_list(
        [
            _build_state_matching_fcc_random_box(
                template,
                composition_seed=campaign.SEED,
                velocity_seed=campaign.SEED + i,
                temperature_k=temperature_k,
                device=device,
            )
            for i in range(width)
        ]
    )


def _make_factory(
    kernel: str,
    model,
    template,
    temperature_k: float,
    md_steps_per_block: int,
    mc_steps_per_block: int,
):
    """Return workload_factory(width, device) -> (runner, batch) for `kernel`."""

    def npt_factory(width, device):
        batch = _build_batch(template, temperature_k, width, device)
        npt = NPT(
            model=model,
            dt=campaign.DT_FS,
            temperature=torch.full((width,), temperature_k, device=device),
            pressure=torch.full((width,), campaign.PRESSURE_EV_PER_A3, device=device),
            thermostat_time=campaign.THERMOSTAT_TIME_FS,
            barostat_time=campaign.BAROSTAT_TIME_FS,
            pressure_coupling="isotropic",
        )
        return _BlockAdapter(npt, md_steps_per_block), batch

    def nvt_factory(width, device):
        batch = _build_batch(template, temperature_k, width, device)
        nvt = NVTNoseHoover(
            model=model,
            dt=campaign.DT_FS,
            temperature=torch.full((width,), temperature_k, device=device),
            thermostat_time=campaign.THERMOSTAT_TIME_FS,
        )
        return _BlockAdapter(nvt, md_steps_per_block), batch

    def kawasaki_factory(width, device):
        batch = _build_batch(template, temperature_k, width, device)
        kawasaki = Kawasaki(
            model=model,
            temperature=temperature_k,
            cutoff=3.40,
            random_seed=campaign.SEED,
        )
        return _BlockAdapter(kawasaki, mc_steps_per_block), batch

    def sgc_factory(width, device):
        batch = _build_batch(template, temperature_k, width, device)
        sgc = SGC(
            model=model,
            temperature=temperature_k,
            species=campaign.SPECIES,
            chemical_potentials={campaign.SPECIES[0]: 0.0, campaign.SPECIES[1]: 0.5},
            random_seed=campaign.SEED,
        )
        return _BlockAdapter(sgc, mc_steps_per_block), batch

    def hybrid_factory(width, device):
        batch = _build_batch(template, temperature_k, width, device)
        sgc = SGC(
            model=model,
            temperature=torch.full((width,), temperature_k, device=device),
            species=campaign.SPECIES,
            chemical_potentials={
                campaign.SPECIES[0]: torch.zeros(width, device=device),
                campaign.SPECIES[1]: torch.full((width,), 0.5, device=device),
            },
            random_seed=campaign.SEED,
        )
        npt = NPT(
            model=model,
            dt=campaign.DT_FS,
            temperature=torch.full((width,), temperature_k, device=device),
            pressure=torch.full((width,), campaign.PRESSURE_EV_PER_A3, device=device),
            thermostat_time=campaign.THERMOSTAT_TIME_FS,
            barostat_time=campaign.BAROSTAT_TIME_FS,
            pressure_coupling="isotropic",
        )
        return HybridMCMD(
            mc=sgc, md=npt, mc_steps=mc_steps_per_block, md_steps=md_steps_per_block
        ), batch

    return {
        "npt": npt_factory,
        "nvt": nvt_factory,
        "kawasaki": kawasaki_factory,
        "sgc": sgc_factory,
        "hybrid_sgc_npt": hybrid_factory,
    }[kernel]


def main() -> None:
    """Command-line entry point: measure peak memory over kernels, sizes and widths."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--inference-settings",
        required=True,
        help="A fairchem preset (default/turbo/batch) or a key=value InferenceSettings spec.",
    )
    parser.add_argument(
        "--kernels", nargs="+", default=list(KERNELS), choices=list(KERNELS)
    )
    parser.add_argument(
        "--n-atoms-list", type=int, nargs="+", default=sorted(campaign.SIZE_REPEATS)
    )
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=None,
        help="Override campaign.BATCH_WIDTH_CANDIDATES[n_atoms] for every n_atoms swept.",
    )
    parser.add_argument(
        "--warmup-blocks", type=int, default=campaign.PROFILE_WARMUP_BLOCKS
    )
    parser.add_argument(
        "--measured-blocks", type=int, default=campaign.PROFILE_MEASURED_BLOCKS
    )
    parser.add_argument(
        "--md-steps-per-block",
        type=int,
        default=10,
        help="Memory characterization only -- deliberately NOT campaign.MD_STEPS_PER_BLOCK (50).",
    )
    parser.add_argument(
        "--mc-steps-per-block",
        type=int,
        default=10,
        help="Memory characterization only -- deliberately NOT MC_STEP_FRACTION-scaled.",
    )
    parser.add_argument("--temperature-k", type=float, default=1800.0)
    parser.add_argument(
        "--memory-fraction", type=float, default=campaign.BATCH_MEMORY_FRACTION
    )
    parser.add_argument(
        "--throughput-fraction", type=float, default=campaign.BATCH_THROUGHPUT_FRACTION
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "memory_profile_matrix.json"
    csv_path = args.output_dir / "memory_profile_matrix.csv"

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

    planner = SimulationBatchPlanner(
        memory_fraction=args.memory_fraction,
        throughput_fraction=args.throughput_fraction,
    )

    csv_file = csv_path.open("w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(
        [
            "kernel",
            "composition_consistent",
            "n_atoms",
            "inference_settings",
            "batch_width",
            "status",
            "peak_reserved_GB",
            "walker_blocks_per_second",
            "atoms_per_walker",
            "error",
        ]
    )

    results: dict[str, dict[str, dict]] = {}
    for kernel in args.kernels:
        results[kernel] = {}
        if kernel in SGC_KERNELS and not _sgc_safe(inference_settings):
            print(
                f"[memory_profile_matrix] skipping kernel={kernel} under "
                f"inference_settings={args.inference_settings!r}: SGC-containing kernels must not "
                "use merge_mole or compile, see docs/userguide/dynamics_simulations.md",
                flush=True,
            )
            for n_atoms in args.n_atoms_list:
                results[kernel][str(n_atoms)] = {
                    "skipped": True,
                    "reason": "SGC-containing kernels must not use merge_mole/compile; not a memory/speed "
                    "tradeoff, see docs/userguide/dynamics_simulations.md",
                }
            continue
        for n_atoms in args.n_atoms_list:
            repeats = campaign.SIZE_REPEATS[n_atoms]
            template = campaign.build_ase_structure(
                campaign.TEMPLATE_SYMBOL,
                campaign.CRYSTAL_STRUCTURE,
                campaign.LATTICE_A_ANG,
                repeats,
                cubic=campaign.CONVENTIONAL_CELL,
            )
            widths = args.widths or list(campaign.BATCH_WIDTH_CANDIDATES[n_atoms])
            factory = _make_factory(
                kernel,
                model,
                template,
                args.temperature_k,
                args.md_steps_per_block,
                args.mc_steps_per_block,
            )
            print(
                f"[memory_profile_matrix] kernel={kernel} n_atoms={n_atoms} widths={widths}",
                flush=True,
            )
            measurements = planner.profile(
                factory,
                widths,
                device=device,
                warmup_blocks=args.warmup_blocks,
                measured_blocks=args.measured_blocks,
            )
            for m in measurements:
                writer.writerow(
                    [
                        kernel,
                        COMPOSITION_CONSISTENT[kernel],
                        n_atoms,
                        args.inference_settings,
                        m.batch_width,
                        m.status,
                        f"{m.peak_reserved_bytes / 1024**3:.4f}"
                        if m.peak_reserved_bytes is not None
                        else "",
                        f"{m.walker_blocks_per_second:.6f}"
                        if m.walker_blocks_per_second is not None
                        else "",
                        m.atoms_per_walker if m.atoms_per_walker is not None else "",
                        (m.error or "").replace("\n", " ")[:200],
                    ]
                )
            csv_file.flush()

            model_estimate = None
            ok = [m for m in measurements if m.status == "ok"]
            if len(ok) >= 2:
                fit = SimulationBatchPlanner.infer_memory_model(measurements)
                model_estimate = {
                    "resident_GB": fit.model_resident_bytes / 1024**3,
                    "per_walker_GB": fit.bytes_per_walker / 1024**3,
                }
            results[kernel][str(n_atoms)] = {
                "composition_consistent": COMPOSITION_CONSISTENT[kernel],
                "widths_requested": widths,
                "measurements": [
                    {
                        "batch_width": m.batch_width,
                        "status": m.status,
                        "peak_reserved_GB": m.peak_reserved_bytes / 1024**3
                        if m.peak_reserved_bytes is not None
                        else None,
                        "walker_blocks_per_second": m.walker_blocks_per_second,
                        "atoms_per_walker": m.atoms_per_walker,
                        "error": m.error,
                    }
                    for m in measurements
                ],
                "linear_memory_model": model_estimate,
            }
    csv_file.close()
    process_seconds = time.perf_counter() - process_start

    output = {
        "backend": "nvalchemi_toolkit",
        "scope": "GPU memory-vs-batch-width profiling matrix -- see docs/userguide/dynamics_simulations.md",
        "checkpoint": campaign.CHECKPOINT,
        "task_name": campaign.TASK,
        "inference_settings": args.inference_settings,
        "temperature_K": args.temperature_k,
        "md_steps_per_block": args.md_steps_per_block,
        "mc_steps_per_block": args.mc_steps_per_block,
        "warmup_blocks": args.warmup_blocks,
        "measured_blocks": args.measured_blocks,
        "memory_fraction": args.memory_fraction,
        "throughput_fraction": args.throughput_fraction,
        "model_load_seconds": model_load_seconds,
        "process_wall_seconds": process_seconds,
        "results": results,
        "csv": str(csv_path),
    }
    json_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({k: v for k, v in output.items() if k != "results"}, indent=2))
    print(
        f"[memory_profile_matrix] done: model_load={model_load_seconds:.2f}s, total={process_seconds:.2f}s"
    )


if __name__ == "__main__":
    main()
