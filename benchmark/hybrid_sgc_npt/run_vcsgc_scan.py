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
"""Hybrid VC-SGC-NPT walkers at fixed target compositions, for the Au-Pt miscibility gap.

Each walker samples exp{-beta [U + PV + N kappa (c - c0)^2 - N (dmu_ref) c]} (Sadigh et al.,
PRB 85, 184203 (2012); ``nvalchemi.mc.VCSGC`` with ``reference_exchange_potential``), alternating
transmutation MC blocks with NPT MD exactly like the SGC-NPT scan. The constraint holds the mean
Pt fraction near c0 even inside the miscibility gap, where plain SGC jumps to one side, and

    dmu(c_bar) = mu_Pt - mu_Au = dmu_ref + 2 kappa (c0 - c_bar)

is the slope of the fixed-composition Gibbs energy g(c) at the sampled mean. A grid of c0 across
the gap therefore gives g(c) by integration, and the common tangent of g(c) gives dmu_coex and
both coexisting compositions, with vibrations and relaxation included and without any
pure-element free-energy anchor (``vcsgc_analysis.py``).

Walkers are independent, so any number share one batch (one per c0). Starting states:
- ``slab`` (default inside ``--slab-range``): Pt on the lowest-z planes, Au above, i.e. already
  phase-separated, so the walker does not have to nucleate a second phase with transmutations;
- ``random``: a homogeneous random alloy at c0 (default outside ``--slab-range``, where the
  equilibrium state is one dilute phase). Running a few gap compositions from ``random`` too is
  the hysteresis check: both starts must give the same dmu(c_bar).
The cell starts at the Vegard lattice constant from the pure-element calibration.

Resumable: every ``--chunk-blocks`` blocks each walker's state goes to ``<out>/states`` and its
per-block series (c, U/atom, V/atom) to ``<out>/<run_id>.series.json``; a resubmission continues
each walker where it stopped, until ``--n-blocks``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from run_campaign import (
    BAROSTAT_TIME_FS,
    CHECKPOINT,
    CRYSTAL_STRUCTURE,
    DT_FS,
    INFERENCE_SETTINGS,
    KB_EV,
    PRESSURE_EV_PER_A3,
    SEED,
    SIZE_REPEATS,
    SPECIES,
    TASK,
    THERMOSTAT_TIME_FS,
    _cell_volumes,
    _npt_wrap_hooks,
    _pt_fraction_per_graph,
    _wrap_batch_positions,
    build_ase_structure,
)

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.integrators.npt import NPT
from nvalchemi.hybrid import HybridMCMD
from nvalchemi.mc import VCSGC
from nvalchemi.models.uma import UMAWrapper
from nvalchemi.scheduling import FinalStateStore

# Au, Pt lattice constants at 700 K from the scan's calibration; only the starting cell.
DEFAULT_A = {79: 4.2221, 78: 3.9978}


def run_id_for(n_atoms: int, temperature: float, kappa: float, c0: float, init: str) -> str:
    return f"atoms{n_atoms}.T{temperature:g}.vcsgc.k{kappa:g}.c{c0:.3f}.{init}"


def initial_state(
    n_atoms: int, temperature: float, c0: float, init: str, lattice: dict, seed: int, device
) -> AtomicData:
    """Fresh walker: slab or random Pt placement at round(c0 N), Vegard cell, MB velocities."""
    a = (1 - c0) * lattice[SPECIES[0]] + c0 * lattice[SPECIES[1]]
    atoms = build_ase_structure("Au", CRYSTAL_STRUCTURE, a, SIZE_REPEATS[n_atoms], cubic=True)
    data = AtomicData.from_atoms(atoms, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n_pt = round(c0 * n_atoms)
    if init == "slab":
        z = torch.as_tensor(atoms.get_scaled_positions(wrap=True)[:, 2])
        jitter = torch.rand(n_atoms, generator=generator, dtype=z.dtype) * 1e-3  # random within a plane
        pt_sites = torch.argsort(z + jitter)[:n_pt]
    else:
        pt_sites = torch.randperm(n_atoms, generator=generator)[:n_pt]
    numbers = torch.full((n_atoms,), SPECIES[0], dtype=data.atomic_numbers.dtype)
    numbers[pt_sites] = SPECIES[1]
    data.atomic_numbers = numbers.to(device)
    data.atomic_masses = None
    data.use_default_masses()
    std = torch.sqrt(torch.as_tensor(KB_EV * temperature, device=device) / data.atomic_masses)
    noise = torch.randn((n_atoms, 3), generator=generator).to(device)
    data.velocities = noise * std[:, None]
    data.velocities -= data.velocities.mean(dim=0, keepdim=True)
    data.forces = torch.zeros_like(data.positions)
    data.energy = torch.zeros(1, 1, device=device)
    data.stress = torch.zeros(1, 3, 3, device=device)
    return data


def build_hybrid(model, n_graphs, temperature, c0s, kappa, dmu_ref, mc_steps, md_steps, seed, device):
    temps = torch.full((n_graphs,), float(temperature), device=device)
    vc = VCSGC(
        model=model,
        temperature=temps,
        species=SPECIES,
        kappa=float(kappa),
        target_concentration=torch.tensor(c0s, dtype=torch.float64),
        concentration_species=SPECIES[1],
        reference_exchange_potential=float(dmu_ref),
        random_seed=seed,
    )
    npt = NPT(
        model=model,
        dt=DT_FS,
        temperature=temps,
        pressure=torch.full((n_graphs,), PRESSURE_EV_PER_A3, device=device),
        thermostat_time=THERMOSTAT_TIME_FS,
        barostat_time=BAROSTAT_TIME_FS,
        pressure_coupling="isotropic",
        hooks=_npt_wrap_hooks(),
    )
    return HybridMCMD(mc=vc, md=npt, mc_steps=mc_steps, md_steps=md_steps, mc_energy_only=True)


def run_chunk(hybrid: HybridMCMD, batch: Batch, n_blocks: int, n_atoms: int) -> dict:
    """``n_blocks`` hybrid blocks; per-graph c, U/atom, V/atom after each block."""
    n = batch.num_graphs
    series = {"c": [[] for _ in range(n)], "u": [[] for _ in range(n)], "v": [[] for _ in range(n)]}
    with hybrid.md:
        hybrid.md.compute(batch)
        hybrid.mc.synchronize(batch)
        for _ in range(n_blocks):
            hybrid.run_mc_block(batch)
            hybrid.md.compute(batch)
            hybrid.md.run(batch, n_steps=hybrid.md_steps)
            hybrid.mc.synchronize(batch)
            c = _pt_fraction_per_graph(batch, SPECIES[1], n)
            u = (batch.energy.detach().reshape(-1).double() / n_atoms).cpu().tolist()
            v = (_cell_volumes(batch).detach().double() / n_atoms).cpu().tolist()
            for i in range(n):
                series["c"][i].append(c[i])
                series["u"][i].append(u[i])
                series["v"][i].append(v[i])
    return series


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="checkpoint root for this VC-SGC campaign")
    ap.add_argument("--c0", type=float, nargs="+", required=True, help="target Pt fractions, one walker each")
    ap.add_argument("--temperature-k", type=float, default=700.0)
    ap.add_argument("--kappa", type=float, default=0.5, help="eV, intensive (VCSGC docstring)")
    ap.add_argument("--n-atoms", type=int, default=500)
    ap.add_argument("--n-blocks", type=int, default=300)
    ap.add_argument("--chunk-blocks", type=int, default=25, help="blocks between checkpoints")
    ap.add_argument("--mc-step-fraction", type=float, default=0.6)
    ap.add_argument("--md-steps-per-block", type=int, default=50)
    ap.add_argument("--init", choices=["auto", "slab", "random"], default="auto")
    ap.add_argument("--slab-range", type=float, nargs=2, default=(0.08, 0.92),
                    help="--init auto: slab start for c0 inside this range, random outside")
    ap.add_argument("--reference-energies", type=Path,
                    help="auto_reference_energies.json from the SGC scan (dmu_ref and lattice constants)")
    ap.add_argument("--delta-mu-ref", type=float, help="overrides the reference file's dmu_ref (eV)")
    ap.add_argument("--inference-settings", default=INFERENCE_SETTINGS)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    T = args.temperature_k
    lattice = dict(DEFAULT_A)
    dmu_ref = args.delta_mu_ref
    if args.reference_energies:
        ref = json.loads(args.reference_energies.read_text())["reference"][f"{T:g}"]
        lattice = {SPECIES[0]: ref["Au"]["lattice_constant_a_ang"], SPECIES[1]: ref["Pt"]["lattice_constant_a_ang"]}
        dmu_ref = ref["delta_mu_ref_eV"] if dmu_ref is None else dmu_ref
    if dmu_ref is None:
        ap.error("pass --reference-energies or --delta-mu-ref")

    args.out.mkdir(parents=True, exist_ok=True)
    store = FinalStateStore(args.out / "states")
    walkers = []
    for c0 in args.c0:
        init = args.init
        if init == "auto":
            init = "slab" if args.slab_range[0] <= c0 <= args.slab_range[1] else "random"
        rid = run_id_for(args.n_atoms, T, args.kappa, c0, init)
        spath = args.out / f"{rid}.series.json"
        series = json.loads(spath.read_text()) if spath.is_file() else None
        if series is None:
            series = dict(
                run_id=rid, temperature_K=T, c0=c0, kappa=args.kappa, init=init, n_atoms=args.n_atoms,
                delta_mu_ref_eV=dmu_ref, mc_step_fraction=args.mc_step_fraction,
                md_steps_per_block=args.md_steps_per_block, inference_settings=args.inference_settings,
                checkpoint=CHECKPOINT, task=TASK, c=[], u=[], v=[], acceptance=[],
            )
        elif (series["kappa"], series["delta_mu_ref_eV"], series["inference_settings"]) != (
            args.kappa, dmu_ref, args.inference_settings
        ):
            raise SystemExit(f"{rid}: existing series was made with different kappa/dmu_ref/settings")
        walkers.append(dict(rid=rid, c0=c0, init=init, path=spath, series=series))

    model = UMAWrapper.from_checkpoint(
        CHECKPOINT, task_name=TASK, device=str(device), inference_settings=args.inference_settings
    )
    mc_steps = max(1, round(args.mc_step_fraction * args.n_atoms))
    print(
        f"[vcsgc] T={T:g} K kappa={args.kappa} eV dmu_ref={dmu_ref:.5f} eV, {len(walkers)} walkers "
        f"{[(w['c0'], w['init'], len(w['series']['c'])) for w in walkers]}, {mc_steps} MC + "
        f"{args.md_steps_per_block} MD per block, settings {args.inference_settings!r}, out {args.out}",
        flush=True,
    )

    while True:
        active = [w for w in walkers if len(w["series"]["c"]) < args.n_blocks]
        if not active:
            break
        # Walkers that finished drop out; the rest advance by the same chunk.
        done = min(len(w["series"]["c"]) for w in active)
        chunk = min(args.chunk_blocks, args.n_blocks - done)
        active = [w for w in active if len(w["series"]["c"]) == done]
        states = [
            store.load(w["rid"], device=device) if store.exists(w["rid"]) and done > 0
            else initial_state(args.n_atoms, T, w["c0"], w["init"], lattice, SEED + int(1000 * w["c0"]), device)
            for w in active
        ]
        batch = Batch.from_data_list(states)
        _wrap_batch_positions(batch)
        # Seed per chunk, so a resumed walker does not replay the random stream it started with.
        hybrid = build_hybrid(
            model, len(active), T, [w["c0"] for w in active], args.kappa, dmu_ref,
            mc_steps, args.md_steps_per_block, SEED + done, device,
        )
        start = time.perf_counter()
        out = run_chunk(hybrid, batch, chunk, args.n_atoms)
        elapsed = time.perf_counter() - start
        acceptance = hybrid.mc.stats.acceptance
        for i, (w, final) in enumerate(zip(active, batch.to_data_list())):
            store.save(w["rid"], final)
            s = w["series"]
            for key in ("c", "u", "v"):
                s[key].extend(out[key][i])
            s["acceptance"].append(dict(blocks=len(s["c"]), acceptance=acceptance))
            tmp = w["path"].with_suffix(".tmp")
            tmp.write_text(json.dumps(s) + "\n")
            tmp.replace(w["path"])
        print(
            f"[vcsgc] blocks {done}->{done + chunk} width {len(active)}: {elapsed:.0f} s "
            f"({len(active) * chunk / elapsed:.4f} walker-blocks/s), acceptance {acceptance:.4f}; "
            + ", ".join(
                f"c0={w['c0']:.3f}: c={sum(w['series']['c'][-chunk:]) / chunk:.4f}" for w in active
            ),
            flush=True,
        )
        del hybrid, batch
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
    print("[vcsgc] complete", flush=True)


if __name__ == "__main__":
    main()
