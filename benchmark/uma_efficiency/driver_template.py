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
"""Template driver: batched MC, MD or hybrid MC-MD with nvalchemi-toolkit + FairChem UMA.

Adapt the marked sections (system, schedule, observables). Everything else encodes
measured lessons -- read the comments before changing it. Smoke-test with
--n-blocks 2 before a long submission.

    python driver.py --kind kawasaki-npt --structure start.xyz --width 4 \
        --inference-settings "compile=false,merge_mole=true,tf32=true,activation_checkpointing=false" \
        --temperature-k 800 900 1000 1100 --n-blocks 200 --output run/metrics.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from ase.io import read, write

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import DynamicsStage
from nvalchemi.dynamics.integrators.npt import NPT
from nvalchemi.dynamics.integrators.nvt_nose_hoover import NVTNoseHoover
from nvalchemi.hooks import WrapPeriodicHook
from nvalchemi.hooks.periodic import wrap_positions_into_cell
from nvalchemi.hybrid import HybridMCMD
from nvalchemi.mc import SGC, VCSGC, Kawasaki
from nvalchemi.models.uma import UMAWrapper

KB_EV = 8.617333262e-5
DT_FS, THERMOSTAT_FS, BAROSTAT_FS = 3.0, 100.0, 1000.0  # run_campaign.py's values
PRESSURE_EV_PER_A3 = 1.01325 / 1.602176634e6  # 1 atm


def build_walker(
    atoms, seed: int, temperature_k: float, device: torch.device
) -> AtomicData:
    """One walker from an ASE structure: masses from species, Maxwell-Boltzmann velocities."""
    data = AtomicData.from_atoms(atoms, device=device)
    data.atomic_masses = None
    data.use_default_masses()  # MC samplers keep these in step with species from here on
    generator = torch.Generator(device=device).manual_seed(seed)
    std = torch.sqrt(
        torch.as_tensor(KB_EV * temperature_k, device=device) / data.atomic_masses
    )
    data.velocities = (
        torch.randn((data.num_nodes, 3), device=device, generator=generator)
        * std[:, None]
    )
    data.velocities -= data.velocities.mean(dim=0, keepdim=True)
    # Preallocate outputs: MD integrators and HybridMCMD write into these in place.
    data.forces = torch.zeros_like(data.positions)
    data.energy = torch.zeros(1, 1, device=device)
    data.stress = torch.zeros(1, 3, 3, device=device)
    return data


def wrap_into_cells(batch: Batch) -> None:
    """Fold positions into their cells once, before the first force call
    (run_campaign.py's _wrap_batch_positions: the NPT hook only fires after MD steps)."""
    cell = batch.cell
    pbc = getattr(batch, "pbc", None)
    if pbc is None:
        pbc = torch.ones((batch.num_graphs, 3), dtype=torch.bool, device=batch.device)
    if cell.dim() == 4:
        cell = cell.squeeze(1)
    if pbc.dim() == 3:
        pbc = pbc.squeeze(1)
    with torch.no_grad():
        wrap_positions_into_cell(
            batch.positions, cell, pbc.to(torch.bool), batch.batch_idx
        )


def main() -> None:
    """Command-line entry point: build the batch, sampler or integrator and run it."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kind",
        required=True,
        choices=[
            "kawasaki",
            "sgc",
            "vcsgc",
            "npt",
            "nvt",
            "kawasaki-npt",
            "sgc-npt",
            "vcsgc-npt",
        ],
    )
    ap.add_argument("--structure", required=True, type=Path)
    ap.add_argument(
        "--width",
        type=int,
        default=1,
        help="walkers batched in one GPU call (from plan_run.py)",
    )
    ap.add_argument(
        "--inference-settings",
        required=True,
        help="preset name or key=value spec (from plan_run.py)",
    )
    ap.add_argument("--checkpoint", default="uma-s-1p2")
    ap.add_argument("--task", default="omat")
    # One value, or one per walker: a batch can hold a (T, dmu) grid, not just replicas.
    ap.add_argument("--temperature-k", type=float, nargs="+", required=True)
    ap.add_argument("--n-blocks", type=int, default=100)
    ap.add_argument(
        "--mc-steps", type=int, help="MC steps per block (default round(0.2 * n_atoms))"
    )
    ap.add_argument("--md-steps", type=int, default=50)
    ap.add_argument(
        "--species",
        type=int,
        nargs="+",
        help="SGC/VCSGC/Kawasaki species (atomic numbers)",
    )
    ap.add_argument(
        "--delta-mu-ev",
        type=float,
        nargs="+",
        default=[0.0],
        help="SGC: mu(species[1]) - mu(species[0]); one value or one per walker",
    )
    ap.add_argument(
        "--kawasaki-cutoff",
        type=float,
        default=3.4,
        help="first-RDF-minimum-ish swap radius, A",
    )
    ap.add_argument(
        "--kappa",
        type=float,
        default=1.0,
        help="VCSGC constraint stiffness, eV (intensive)",
    )
    ap.add_argument(
        "--target-concentration", type=float, help="VCSGC target fraction of species[1]"
    )
    ap.add_argument(
        "--reference-exchange-potential",
        type=float,
        help="VCSGC: calibrated mu(species[1]) - mu(species[0]) at this T, eV. Required for an MLIP.",
    )
    ap.add_argument("--seed", type=int, default=20260927)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path, default=Path("run/metrics.json"))
    args = ap.parse_args()
    device = torch.device(args.device)

    def per_walker(values: list[float], name: str) -> torch.Tensor:
        if len(values) not in (1, args.width):
            raise SystemExit(
                f"{name}: give one value or --width={args.width} values, got {len(values)}"
            )
        return torch.as_tensor(
            values * (args.width if len(values) == 1 else 1),
            dtype=torch.float32,
            device=device,
        )

    temperatures = per_walker(args.temperature_k, "--temperature-k")
    has_mc = args.kind in ("kawasaki", "sgc", "vcsgc") or "-" in args.kind
    has_md = args.kind in ("npt", "nvt") or "-" in args.kind

    model = UMAWrapper.from_checkpoint(
        args.checkpoint,
        task_name=args.task,
        device=str(device),
        inference_settings=args.inference_settings,
    )  # spec strings accepted

    atoms = read(args.structure)
    # Walker i gets seed + i: distinct velocities and MC streams per walker.
    batch = Batch.from_data_list(
        [
            build_walker(atoms, args.seed + i, float(temperatures[i]), device)
            for i in range(args.width)
        ]
    )
    wrap_into_cells(batch)  # start inside the cell (see WrapPeriodicHook below)
    mc_steps = args.mc_steps or max(1, round(0.2 * len(atoms)))

    mc = md = None
    if has_mc:
        mc_kind = args.kind.split("-")[0]
        if mc_kind == "kawasaki":
            # unlike_pairs_only=True (default): every step a real swap, exact MH correction.
            mc = Kawasaki(
                model=model,
                temperature=temperatures,
                cutoff=args.kawasaki_cutoff,
                random_seed=args.seed,
            )
        elif mc_kind == "sgc":
            a, b = args.species
            mc = SGC(
                model=model,
                temperature=temperatures,
                species=[a, b],
                chemical_potentials={
                    a: torch.zeros_like(temperatures),
                    b: per_walker(args.delta_mu_ev, "--delta-mu-ev"),
                },
                random_seed=args.seed,
            )
        else:
            # Without the calibrated reference, an MLIP's per-element energy offsets (eV) swamp
            # the constraint (at most 2*kappa) and the walker runs to one end member.
            if (
                args.reference_exchange_potential is None
                or args.target_concentration is None
            ):
                raise SystemExit(
                    "VCSGC needs --target-concentration and --reference-exchange-potential"
                )
            mc = VCSGC(
                model=model,
                temperature=temperatures,
                species=args.species,
                kappa=args.kappa,
                target_concentration=args.target_concentration,
                reference_exchange_potential=args.reference_exchange_potential,
                random_seed=args.seed,
            )
    if has_md:
        ensemble = args.kind.split("-")[-1]
        if ensemble == "npt":
            md = NPT(
                model=model,
                dt=DT_FS,
                temperature=temperatures,
                pressure=torch.full((args.width,), PRESSURE_EV_PER_A3, device=device),
                thermostat_time=THERMOSTAT_FS,
                barostat_time=BAROSTAT_FS,
                pressure_coupling="isotropic",
                # Without per-step wrapping, atoms that diffuse > ~1 cell apart lose their
                # minimum image in fairchem's graph and can collapse onto each other.
                hooks=[
                    WrapPeriodicHook(frequency=1, stage=DynamicsStage.AFTER_POST_UPDATE)
                ],
            )
        else:
            md = NVTNoseHoover(
                model=model,
                dt=DT_FS,
                temperature=temperatures,
                thermostat_time=THERMOSTAT_FS,
                hooks=[
                    WrapPeriodicHook(frequency=1, stage=DynamicsStage.AFTER_POST_UPDATE)
                ],
            )

    start = time.perf_counter()
    if mc is not None and md is not None:
        # mc_energy_only: MC blocks skip the forces/stress backward (1.3-2.1x per MC step);
        # MD keeps full outputs. For a custom loop, call hybrid.run_mc_block(), never mc.run().
        hybrid = HybridMCMD(
            mc=mc, md=md, mc_steps=mc_steps, md_steps=args.md_steps, mc_energy_only=True
        )
        hybrid.run(batch, n_blocks=args.n_blocks)
    elif mc is not None:
        model.model_config.active_outputs = {
            "energy"
        }  # MC-only: energies are all MC reads
        mc.run(batch, n_steps=args.n_blocks * mc_steps)
    else:
        if args.kind == "nvt":
            model.model_config.active_outputs = {
                "energy",
                "forces",
            }  # NVT needs no stress
        with md:
            md.compute(batch)
            md.run(batch, n_steps=args.n_blocks * args.md_steps)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - start

    # --- observables: adapt ------------------------------------------------------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for i, walker in enumerate(batch.to_data_list()):
        final = atoms.copy()
        final.set_atomic_numbers(walker.atomic_numbers.cpu().numpy())
        final.set_positions(walker.positions.detach().cpu().numpy())
        if walker.cell is not None:
            final.set_cell(walker.cell.reshape(3, 3).detach().cpu().numpy())
        write(args.output.parent / f"final_w{i}.xyz", final)
    metrics = {
        "kind": args.kind,
        "width": args.width,
        "temperature_k": temperatures.tolist(),
        "inference_settings": args.inference_settings,
        "n_atoms": len(atoms),
        "n_blocks": args.n_blocks,
        "mc_steps_per_block": mc_steps if has_mc else 0,
        "md_steps_per_block": args.md_steps if has_md else 0,
        "wall_seconds": wall,
        "final_energy_eV": batch.energy.flatten().tolist(),
        "mc_acceptance": mc.stats.acceptance if mc is not None else None,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_gpu_memory_reserved_GiB": torch.cuda.max_memory_reserved(device) / 2**30
        if device.type == "cuda"
        else None,
    }
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
