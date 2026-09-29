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
"""Wall-time / memory / final-energy benchmark for NPT molecular dynamics alone.

Drives ``nvalchemi.dynamics.integrators.npt.NPT`` directly (no Monte Carlo, no
HybridMCMD) on an ``--n-atoms`` Au-Pt fcc cell. Every physical parameter
(CHECKPOINT, TASK, SPECIES, LATTICE_A_ANG, DT_FS, THERMOSTAT_TIME_FS,
BAROSTAT_TIME_FS, PRESSURE_EV_PER_A3, PT_FRACTION) except the inference
settings is read from benchmark/hybrid_sgc_npt/run_campaign.py.

Positions come from run_campaign.py's ``build_ase_structure()``. Compositions
are assigned by ``_build_state_matching_fcc_random_box()``: a CPU
``torch.Generator`` seeded with ``--composition-seed`` draws a
``torch.randperm`` and the first round(fractions[0] * n_atoms) indices get
species[0] (Au), the rest species[1] (Pt). The batched MC benchmarks reuse this
helper so their replicas are reproducible from the seed alone; a
composition_fingerprint (sha256 of the atomic numbers) is written to
metrics.json so two runs can be checked for identical starting structures.
Velocities are Maxwell-Boltzmann with an independent ``--velocity-seed``.

    python benchmark/uma_efficiency/benchmark_npt_md_only_single_point.py \
        --n-atoms 2048 --n-steps 5000 --temperature-k 1200.0 \
        --device cuda --output npt_md_only/metrics.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# run_campaign.py (workload builders, size tables) is in the sibling benchmark folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hybrid_sgc_npt"))

import run_campaign as campaign  # noqa: E402 -- needs the sys.path insert above

from nvalchemi.data import AtomicData, Batch  # noqa: E402
from nvalchemi.dynamics.integrators.npt import NPT  # noqa: E402
from nvalchemi.models.uma import UMAWrapper  # noqa: E402


def _composition_fingerprint(atomic_numbers: torch.Tensor) -> str:
    numbers = atomic_numbers.detach().to("cpu").numpy().astype(np.int64)
    return hashlib.sha256(numbers.tobytes()).hexdigest()[:16]


def _build_state_matching_fcc_random_box(
    template,
    *,
    composition_seed: int,
    velocity_seed: int,
    temperature_k: float,
    device: torch.device,
) -> AtomicData:
    """Build one AtomicData with the seeded-randperm composition described in
    the module docstring and Maxwell-Boltzmann velocities (the same formula as
    run_campaign.py's ``_walker()``).
    """

    data = AtomicData.from_atoms(template, device=device)
    n_atoms = data.num_nodes

    cpu = torch.device("cpu")
    composition_generator = torch.Generator(device=cpu).manual_seed(composition_seed)
    shuffled = torch.randperm(n_atoms, device=cpu, generator=composition_generator)
    fractions = (1.0 - campaign.PT_FRACTION, campaign.PT_FRACTION)
    counts = [round(fraction * n_atoms) for fraction in fractions]
    counts[-1] += n_atoms - sum(counts)
    numbers_cpu = torch.empty(n_atoms, dtype=torch.long)
    offset = 0
    for number, count in zip(campaign.SPECIES, counts, strict=True):
        numbers_cpu[shuffled[offset : offset + count]] = number
        offset += count
    data.atomic_numbers = numbers_cpu.to(device)

    data.atomic_masses = None
    data.use_default_masses()
    velocity_generator = torch.Generator(device=device).manual_seed(velocity_seed)
    velocity_std = torch.sqrt(
        torch.as_tensor(campaign.KB_EV * temperature_k, device=device)
        / data.atomic_masses
    )
    data.velocities = (
        torch.randn((n_atoms, 3), device=device, generator=velocity_generator)
        * velocity_std[:, None]
    )
    data.velocities -= data.velocities.mean(dim=0, keepdim=True)

    data.forces = torch.zeros_like(data.positions)
    data.energy = torch.zeros(1, 1, device=device)
    data.stress = torch.zeros(1, 3, 3, device=device)
    return data


def main() -> None:
    """Command-line entry point: time NPT MD alone at one state point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n-atoms", type=int, default=2048, choices=sorted(campaign.SIZE_REPEATS)
    )
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--temperature-k", type=float, default=1200.0)
    parser.add_argument(
        "--composition-seed",
        type=int,
        default=2026090102,
        help="Seed of the initial composition (see the module docstring).",
    )
    parser.add_argument(
        "--velocity-seed",
        type=int,
        default=20260901,
        help="Seed of the Maxwell-Boltzmann velocities.",
    )
    parser.add_argument("--inference-settings", default="turbo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_npt_md_only_single_point/metrics.json"),
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    process_start = time.perf_counter()

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

    model_load_start = time.perf_counter()
    model = UMAWrapper.from_checkpoint(
        campaign.CHECKPOINT,
        task_name=campaign.TASK,
        device=str(device),
        inference_settings=args.inference_settings,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    model_load_seconds = time.perf_counter() - model_load_start

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    data = _build_state_matching_fcc_random_box(
        template,
        composition_seed=args.composition_seed,
        velocity_seed=args.velocity_seed,
        temperature_k=args.temperature_k,
        device=device,
    )
    composition_fingerprint = _composition_fingerprint(data.atomic_numbers)
    batch = Batch.from_data_list([data])

    npt = NPT(
        model=model,
        dt=campaign.DT_FS,
        temperature=torch.tensor([args.temperature_k], device=device),
        pressure=torch.tensor([campaign.PRESSURE_EV_PER_A3], device=device),
        thermostat_time=campaign.THERMOSTAT_TIME_FS,
        barostat_time=campaign.BAROSTAT_TIME_FS,
        pressure_coupling="isotropic",
    )

    run_start = time.perf_counter()
    with npt:
        npt.compute(batch)
        npt.run(batch, n_steps=args.n_steps)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    run_seconds = time.perf_counter() - run_start
    process_seconds = time.perf_counter() - process_start

    final_energy_eV = float(batch.energy.flatten()[0])
    n_atoms = int(batch.num_nodes)

    metrics = {
        "backend": "nvalchemi_toolkit",
        "scope": "MD-only: NPT integrator run directly, no HybridMCMD/SGC sampler in the loop",
        "n_atoms": n_atoms,
        "n_steps": args.n_steps,
        "temperature_K": args.temperature_k,
        "species": list(campaign.SPECIES),
        "checkpoint": campaign.CHECKPOINT,
        "task_name": campaign.TASK,
        "inference_settings": args.inference_settings,
        "dt_fs": campaign.DT_FS,
        "thermostat_time_fs": campaign.THERMOSTAT_TIME_FS,
        "barostat_time_fs": campaign.BAROSTAT_TIME_FS,
        "composition_fingerprint": composition_fingerprint,
        "composition_seed": args.composition_seed,
        "velocity_seed": args.velocity_seed,
        "model_load_seconds": model_load_seconds,
        "md_run_wall_seconds": run_seconds,
        "md_run_wall_seconds_per_step": run_seconds / args.n_steps,
        "process_wall_seconds_from_model_load": process_seconds,
        "final_energy_eV": final_energy_eV,
        "final_energy_eV_per_atom": final_energy_eV / n_atoms,
        "md_step_count": npt.step_count,
    }
    if device.type == "cuda":
        metrics["peak_gpu_memory_allocated_GB"] = (
            torch.cuda.max_memory_allocated(device) / 1024**3
        )
        metrics["peak_gpu_memory_reserved_GB"] = (
            torch.cuda.max_memory_reserved(device) / 1024**3
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(
        f"[benchmark] {n_atoms} atoms, {args.n_steps} MD steps: "
        f"md_run={run_seconds:.2f}s, model_load={model_load_seconds:.2f}s, "
        f"final_energy={final_energy_eV:.4f} eV ({final_energy_eV / n_atoms:.6f} eV/atom)"
    )


if __name__ == "__main__":
    main()
