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
"""Classical configurational Gibbs free energy of a pure fcc element under UMA.

This supplies the free-energy anchors F_Au and F_Pt that a relaxed-lattice (hybrid SGC-NPT)
phase-boundary analysis cannot get from the scan itself. Three GPU stages, each saved to
``<out>/<El>.json`` as soon as it finishes so a resubmission skips it:

1. ``lattice``: E(a) of the perfect crystal on a +-1.5 % grid, fitted for the static lattice
   constant a0 and energy e0 (per atom).
2. ``hessian``: force constants at a0 by central finite differences. One atom is displaced by
   +-delta along x, y, z; every other column follows from lattice translations (all fcc sites
   are equivalent in a supercell of the primitive cell). Done twice (two deltas) as a check, with
   TF32 matmuls switched OFF so force differences are not lost to TF32 rounding.
3. ``ladder``: NPT MD at a temperature ladder up to the target, recording <U>/atom and <V>/atom.
   Uses the SGC runs' own inference settings, so the integrated anharmonic enthalpy is that of
   the potential the phase-boundary runs sample.

``pure_free_energy_analysis.py`` (numpy only, runs on a laptop) then combines them:

    g(T) = g_harm(T) - T * integral_0^T [h(T') - h_harm(T')] / T'^2 dT'      (Gibbs-Helmholtz)

with g_harm = e0 + P v0 + (kT / 2N) sum_i ln(lambda_i / (2 pi kT)) over the 3N-3 nonzero
eigenvalues of the force-constant matrix, h_harm = e0 + P v0 + (3N-3)/(2N) kT. Everything is
CLASSICAL and CONFIGURATIONAL (no masses, no kinetic term) because that is what the SGC
transmutation acceptance samples: it compares potential energies at fixed positions.

    python pure_free_energy.py --element Au --out <dir> [--temperatures 100 150 ... 700]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk
from pure_free_energy_analysis import (
    assemble_force_constants,
    fit_lattice,
    harmonic_summary,
)
from run_campaign import (
    BAROSTAT_TIME_FS,
    CHECKPOINT,
    CRYSTAL_STRUCTURE,
    DT_FS,
    EQUILIBRATION_WINDOW_BLOCKS,
    KB_EV,
    PRESSURE_EV_PER_A3,
    SIZE_REPEATS,
    TASK,
    THERMOSTAT_TIME_FS,
    _batch_means_se,
    _cell_volumes,
    _equilibration_gate,
    _npt_wrap_hooks,
    _pure_element_endpoint,
)

from nvalchemi.data import Batch
from nvalchemi.dynamics.integrators.npt import NPT
from nvalchemi.models.uma import UMAWrapper

# Pure-element runs have one fixed composition per task, so MoLE merging is valid and ~2x
# faster for NPT (GPU_MEMORY_GUIDE.md / nvalchemi-uma-submission); merged and unmerged energies
# agree to ueV/atom. Compile stays off: NPT changes the edge count every step.
DEFAULT_SETTINGS = (
    "compile=false,merge_mole=true,tf32=true,activation_checkpointing=false"
)
LATTICE_STRAINS = (-0.015, -0.010, -0.005, 0.0, 0.005, 0.010, 0.015)
DELTAS_ANG = (0.01, 0.02)
DEFAULT_TEMPERATURES = tuple(float(t) for t in range(100, 701, 50))


# ----------------------------------------------------------------------------- GPU stages
def _template(symbol: str, a: float, n_atoms: int):
    return (
        bulk(symbol, crystalstructure=CRYSTAL_STRUCTURE, a=a, cubic=True)
        * SIZE_REPEATS[n_atoms]
    )


def _evaluate(model, templates, device):
    """Energies (eV/atom) and forces (eV/A) of static structures, one batch."""
    data = [_pure_element_endpoint(t, 1.0, 0, device) for t in templates]
    batch = Batch.from_data_list(data)
    npt = NPT(
        model=model,
        dt=DT_FS,
        temperature=torch.full((len(templates),), 1.0, device=device),
        pressure=torch.full((len(templates),), PRESSURE_EV_PER_A3, device=device),
        thermostat_time=THERMOSTAT_TIME_FS,
        barostat_time=BAROSTAT_TIME_FS,
        pressure_coupling="isotropic",
    )
    with npt:
        npt.compute(batch)
    n = len(templates[0])
    energies = batch.energy.detach().reshape(-1).double().cpu().numpy() / n
    forces = batch.forces.detach().double().cpu().numpy().reshape(len(templates), n, 3)
    stress = batch.stress.detach().double().cpu().numpy().reshape(len(templates), 3, 3)
    return energies, forces, stress


def stage_lattice(model, symbol, n_atoms, a_guess, device) -> dict:
    a = np.array([a_guess * (1 + s) for s in LATTICE_STRAINS])
    e, f, s = _evaluate(model, [_template(symbol, ai, n_atoms) for ai in a], device)
    a0, _, curv = fit_lattice(a, e)
    e0, f0, s0 = _evaluate(model, [_template(symbol, a0, n_atoms)], device)
    return dict(
        a_grid=a.tolist(),
        e_grid=e.tolist(),
        a0=a0,
        e0=float(e0[0]),
        v0=a0**3 / 4,
        d2e_da2=curv,
        max_force_at_a0=float(np.abs(f0).max()),
        stress_at_a0=s0[0].tolist(),
    )


def stage_hessian(model, symbol, n_atoms, a0, device, chunk: int) -> dict:
    """Force constants with TF32 matmuls disabled (restored afterwards)."""
    flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        base = _template(symbol, a0, n_atoms)
        frac = base.get_scaled_positions(wrap=True)
        jobs = [(d, ax, sgn) for d in DELTAS_ANG for ax in range(3) for sgn in (+1, -1)]
        structures = []
        for d, ax, sgn in jobs:
            t = base.copy()
            pos = t.get_positions()
            pos[0, ax] += sgn * d
            t.set_positions(pos)
            structures.append(t)
        forces = np.concatenate(
            [
                _evaluate(model, structures[k : k + chunk], device)[1]
                for k in range(0, len(structures), chunk)
            ]
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags
        torch.set_float32_matmul_precision(precision)
    out = {"deltas": {}}
    for d in DELTAS_ANG:
        column = np.zeros((n_atoms, 3, 3))
        for ax in range(3):
            fp = forces[jobs.index((d, ax, +1))]
            fm = forces[jobs.index((d, ax, -1))]
            column[:, :, ax] = -(fp - fm) / (2 * d)
        phi = assemble_force_constants(column, frac)
        out["deltas"][f"{d:g}"] = dict(column=column.tolist(), **harmonic_summary(phi))
    out["fractional_positions"] = frac.tolist()
    return out


def stage_ladder(
    model, symbol, n_atoms, a0, temperatures, n_blocks, md_steps, width, device, done
):
    """NPT at each temperature not yet in ``done``; yields the updated per-T results after each
    batch (<U>/atom, <V>/atom with batch-means SEs over the last two thirds of the run)."""
    results = dict(done)
    todo = [t for t in temperatures if f"{t:g}" not in results]
    for k in range(0, len(todo), width):
        temps = todo[k : k + width]
        data = [
            _pure_element_endpoint(_template(symbol, a0, n_atoms), t, 1000 + i, device)
            for i, t in enumerate(temps)
        ]
        batch = Batch.from_data_list(data)
        npt = NPT(
            model=model,
            dt=DT_FS,
            temperature=torch.tensor(temps, device=device),
            pressure=torch.full((len(temps),), PRESSURE_EV_PER_A3, device=device),
            thermostat_time=THERMOSTAT_TIME_FS,
            barostat_time=BAROSTAT_TIME_FS,
            pressure_coupling="isotropic",
            hooks=_npt_wrap_hooks(),
        )
        u_series = [[] for _ in temps]
        v_series = [[] for _ in temps]
        start = time.perf_counter()
        with npt:
            npt.compute(batch)
            for _ in range(n_blocks):
                npt.run(batch, n_steps=md_steps)
                energies = batch.energy.detach().reshape(-1).double().cpu()
                volumes = _cell_volumes(batch).detach().double().cpu()
                for i in range(len(temps)):
                    u_series[i].append(float(energies[i]) / n_atoms)
                    v_series[i].append(float(volumes[i]) / n_atoms)
        elapsed = time.perf_counter() - start
        for i, t in enumerate(temps):
            prod_u = u_series[i][n_blocks // 3 :]
            prod_v = v_series[i][n_blocks // 3 :]
            results[f"{t:g}"] = dict(
                T=t,
                u_mean=statistics.fmean(prod_u),
                u_se=_batch_means_se(prod_u),
                v_mean=statistics.fmean(prod_v),
                v_se=_batch_means_se(prod_v),
                n_production_blocks=len(prod_u),
                u_gate=_equilibration_gate(u_series[i], EQUILIBRATION_WINDOW_BLOCKS),
                v_gate=_equilibration_gate(v_series[i], EQUILIBRATION_WINDOW_BLOCKS),
                u_series=u_series[i],
                v_series=v_series[i],
            )
        print(
            f"[ladder] {symbol} T={temps}: {elapsed:.0f} s for {n_blocks} blocks x {md_steps} steps, width {len(temps)}; "
            + ", ".join(f"{t:g}K U={results[f'{t:g}']['u_mean']:.5f}" for t in temps),
            flush=True,
        )
        del npt, batch
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
        yield results


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--element", required=True, choices=["Au", "Pt"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-atoms", type=int, default=500)
    ap.add_argument(
        "--temperatures", type=float, nargs="+", default=list(DEFAULT_TEMPERATURES)
    )
    ap.add_argument(
        "--n-blocks",
        type=int,
        default=300,
        help="NPT blocks per ladder temperature (first third discarded)",
    )
    ap.add_argument("--md-steps-per-block", type=int, default=50)
    ap.add_argument(
        "--batch-width", type=int, default=7, help="ladder temperatures per NPT batch"
    )
    ap.add_argument(
        "--hessian-chunk",
        type=int,
        default=4,
        help="displaced structures per force batch",
    )
    ap.add_argument("--inference-settings", default=DEFAULT_SETTINGS)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.element}.json"
    state = json.loads(path.read_text()) if path.is_file() else {}
    meta = dict(
        element=args.element,
        checkpoint=CHECKPOINT,
        task=TASK,
        n_atoms=args.n_atoms,
        pressure_ev_per_a3=PRESSURE_EV_PER_A3,
        kb_ev=KB_EV,
        inference_settings=args.inference_settings,
        hessian_tf32=False,
        md_steps_per_block=args.md_steps_per_block,
        dt_fs=DT_FS,
    )
    if (
        state
        and state.get("meta", {}).get("inference_settings") != args.inference_settings
    ):
        raise SystemExit(
            f"{path} was made with other inference settings; use a fresh --out"
        )
    state["meta"] = meta

    def save() -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1) + "\n")
        tmp.replace(path)

    model = UMAWrapper.from_checkpoint(
        CHECKPOINT,
        task_name=TASK,
        device=str(device),
        inference_settings=args.inference_settings,
    )
    print(
        f"[pure-G] {args.element}: settings {args.inference_settings!r}, out {path}",
        flush=True,
    )

    if "lattice" not in state:
        a_guess = bulk(
            args.element, crystalstructure=CRYSTAL_STRUCTURE, cubic=True
        ).cell[0, 0]
        state["lattice"] = stage_lattice(
            model, args.element, args.n_atoms, a_guess, device
        )
        save()
    lat = state["lattice"]
    print(
        f"[lattice] a0={lat['a0']:.4f} A e0={lat['e0']:.6f} eV/atom max|F|={lat['max_force_at_a0']:.2e}",
        flush=True,
    )

    if "hessian" not in state:
        state["hessian"] = stage_hessian(
            model, args.element, args.n_atoms, lat["a0"], device, args.hessian_chunk
        )
        save()
    for d, h in state["hessian"]["deltas"].items():
        print(
            f"[hessian] delta={d} A: mean ln(lambda)={h['mean_ln_lambda']} lambda in [{h['lambda_min']:.4f}, "
            f"{h['lambda_max']:.3f}] eV/A^2, {h['n_negative']} negative, acoustic {h['acoustic_eigenvalues']}",
            flush=True,
        )

    ladder = state.get("ladder", {})
    for ladder in stage_ladder(
        model,
        args.element,
        args.n_atoms,
        lat["a0"],
        sorted(args.temperatures),
        args.n_blocks,
        args.md_steps_per_block,
        args.batch_width,
        device,
        ladder,
    ):
        state["ladder"] = ladder
        save()
    print(
        f"[pure-G] {args.element}: complete ({len(state.get('ladder', {}))} ladder temperatures)",
        flush=True,
    )


if __name__ == "__main__":
    main()
