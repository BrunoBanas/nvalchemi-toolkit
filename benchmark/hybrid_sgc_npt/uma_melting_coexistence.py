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
"""Melting point of a pure fcc metal under UMA by solid-liquid coexistence.

One process = one element at one temperature, as a single-graph batch (no
multi-walker batching). Run a temperature grid as a Slurm array and bracket T_m
with ``uma_melting_analysis.py``.

Why coexistence, not heating: a perfect crystal heated in MD superheats by
10-20 % before it melts, so a heating ramp overestimates T_m. With a
solid|liquid interface already present there is no nucleation barrier: below
T_m the crystal grows into the liquid, above T_m it melts back.

Stages (same UMA checkpoint/task/inference settings as run_campaign.py):

1. **Solid NPT.** Anisotropic NPT of an ``nx x nx x nz`` conventional fcc
   supercell at T, to get the equilibrium lateral lattice constant.
2. **Liquid.** A copy of the equilibrated solid is melted at fixed lateral
   dimensions (NVT Langevin, box stretched along z by ``--liquid-z-strain``):
   ``--melt-ps`` at ``--melt-temperature-k``, then ``--liquid-eq-ps`` at T. It
   is checked to be < 5 % solid-like before use (the melt is extended if not).
3. **Stitch.** Solid and liquid slabs are stacked along z (interfaces normal to
   z, ``--gap-ang`` of space at each), then relaxed ``--clamp-ps`` in NVT at T with
   dt/2 and a force clamp to remove the initial close contacts.
4. **Production.** Anisotropic NPT at T and ~1 atm, so the liquid density and
   the z length relax freely while x/y stay locked to the crystal. Every
   ``--snapshot-ps`` the solid-like atom fraction (ten Wolde-Frenkel q6 bonds)
   is computed and the frame appended to an extxyz trajectory. The run stops
   early once the box is fully solid or fully liquid for 3 snapshots.

Every integrator carries ``WrapPeriodicHook``: nvalchemi's UMA adapter passes
positions to fairchem unwrapped, and a diffusing liquid otherwise loses
interactions once atoms drift beyond one periodic image (see
``_npt_wrap_hooks`` in run_campaign.py and the BUG note in
nvalchemi/models/uma.py).

Alloy mode (``--delta-mu-ev``): every stage after the hot melt is hybrid SGC-MD at
the absolute Pt chemical potential delta_mu (Au = 0, the run_campaign.py
convention): before each ``--log-every-steps`` MD chunk, ``--mc-step-fraction`` x N
SGC transmutation trials (energy-only). Solid and liquid then each take their own
equilibrium composition at (T, delta_mu), and the interface moves toward the
phase with the lower SGC potential phi = G - delta_mu x, so the T where it stops
moving is a point of the solid-liquid coexistence line, with the solid-like and
liquid-like Pt fractions (``x_pt_solid``/``x_pt_liquid`` in summary.json) as the
solidus and liquidus compositions. ``--element`` is then the host the cells are
built from (Au for the Au-rich branch, Pt for the Pt-rich one); merge_mole must
be off because the composition changes. uma_melting_bisect.py --delta-mu-ev
bisects T at fixed delta_mu.

Outputs in ``--out-dir/<element>/T<temperature>/``: ``series.csv`` (time,
potential energy/atom, temperature, box lengths, solid fraction),
``trajectory.extxyz``, ``snapshots.npz``, ``summary.json``.

Run on a GPU node:
    python benchmark/hybrid_sgc_npt/uma_melting_coexistence.py \\
        --element Au --temperature-k 1300 --out-dir <output_root>/uma_melting
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk
from ase.io import write as ase_write

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_campaign import (  # noqa: E402
    BAROSTAT_TIME_FS,
    CHECKPOINT,
    INFERENCE_SETTINGS,
    KB_EV,
    PRESSURE_EV_PER_A3,
    TASK,
    THERMOSTAT_TIME_FS,
    _npt_wrap_hooks,
    _wrap_batch_positions,
)

from run_campaign import SPECIES  # noqa: E402

from nvalchemi.data import AtomicData, Batch
from nvalchemi.hybrid import HybridMCMD
from nvalchemi.mc import SGC
from nvalchemi.dynamics.hooks import MaxForceClampHook
from nvalchemi.dynamics.integrators.npt import NPT
from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
from nvalchemi.models.uma import UMAWrapper

# Experimental T_m (K) and a hot-melt temperature that liquefies a 1000-atom
# slab within a few ps; only used as defaults.
EXPERIMENTAL_TM_K = {"Au": 1337.0, "Pt": 2041.0}
DEFAULT_MELT_TEMPERATURE_K = {"Au": 2800.0, "Pt": 4000.0}
LANGEVIN_FRICTION_PER_FS = 0.01
SOLID_FRACTION_DONE = (0.05, 0.95)


# ----------------------------------------------------------------------------- structure analysis
def _sph_harm_l6(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Y_6m for m = -6..6, stacked on the last axis (SciPy old/new API)."""
    try:
        from scipy.special import sph_harm_y  # SciPy >= 1.15

        return np.stack([sph_harm_y(6, m, theta, phi) for m in range(-6, 7)], axis=-1)
    except ImportError:
        from scipy.special import sph_harm

        return np.stack([sph_harm(m, 6, phi, theta) for m in range(-6, 7)], axis=-1)


def solid_like_mask(positions: np.ndarray, cell: np.ndarray, k: int = 12) -> np.ndarray:
    """ten Wolde-Frenkel crystallinity: an atom is solid-like when >= 7 of its 12
    nearest neighbours share a normalised q6 vector with dot product > 0.5.

    Calibrated on this workflow's own data: a hot fcc lattice with MSD 0.2 A^2
    gives 0.99, and the liquid Au-rich hybrid-campaign seeds give 0.00-0.01.
    """
    frac = positions @ np.linalg.inv(cell)
    d = frac[None, :, :] - frac[:, None, :]
    d -= np.round(d)
    vec = d @ cell
    r = np.linalg.norm(vec, axis=2)
    np.fill_diagonal(r, np.inf)
    idx = np.argsort(r, axis=1)[:, :k]
    nb = np.take_along_axis(vec, idx[:, :, None], axis=1)
    rr = np.linalg.norm(nb, axis=2)
    theta = np.arccos(np.clip(nb[..., 2] / rr, -1.0, 1.0))
    phi = np.arctan2(nb[..., 1], nb[..., 0])
    q = _sph_harm_l6(theta, phi).mean(axis=1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    bonds = np.real((q[:, None, :] * np.conj(q[idx])).sum(axis=2)) > 0.5
    return bonds.sum(axis=1) >= 7


# ----------------------------------------------------------------------------- state helpers
def _data_from_arrays(
    numbers: np.ndarray,
    positions: np.ndarray,
    cell: np.ndarray,
    velocities: np.ndarray | None,
    temperature_k: float,
    seed: int,
    device: torch.device,
) -> AtomicData:
    """Single-system AtomicData with forces/energy/stress preallocated."""
    atoms = Atoms(numbers=numbers, positions=positions, cell=cell, pbc=True)
    data = AtomicData.from_atoms(atoms, device=device)
    data.atomic_masses = None
    data.use_default_masses()
    if velocities is None:
        generator = torch.Generator(device=device).manual_seed(seed)
        std = torch.sqrt(
            torch.as_tensor(KB_EV * temperature_k, device=device) / data.atomic_masses
        )
        v = (
            torch.randn((len(numbers), 3), device=device, generator=generator)
            * std[:, None]
        )
    else:
        v = torch.as_tensor(velocities, dtype=data.positions.dtype, device=device)
    data.velocities = v - v.mean(dim=0, keepdim=True)
    data.forces = torch.zeros_like(data.positions)
    data.energy = torch.zeros(1, 1, device=device)
    data.stress = torch.zeros(1, 3, 3, device=device)
    return data


def _snapshot(batch: Batch) -> dict[str, np.ndarray]:
    """Host copy of the single system's state."""
    return {
        "numbers": batch.atomic_numbers.detach().cpu().numpy().astype(int),
        "positions": batch.positions.detach().cpu().double().numpy(),
        "velocities": batch.velocities.detach().cpu().double().numpy(),
        "masses": batch.atomic_masses.detach().cpu().double().numpy(),
        "cell": batch.cell.detach().reshape(-1, 3, 3)[0].cpu().double().numpy(),
        "energy": float(batch.energy.detach().reshape(-1)[0].cpu()),
    }


def _temperature(s: dict[str, np.ndarray]) -> float:
    return float(
        (s["masses"][:, None] * s["velocities"] ** 2).sum()
        / (3 * len(s["masses"]) * KB_EV)
    )


def _npt(
    model: UMAWrapper, temperature_k: float, dt_fs: float, device: torch.device
) -> NPT:
    return NPT(
        model=model,
        dt=dt_fs,
        temperature=torch.tensor([temperature_k], device=device),
        pressure=torch.full((1, 3), PRESSURE_EV_PER_A3, device=device),
        thermostat_time=THERMOSTAT_TIME_FS,
        barostat_time=BAROSTAT_TIME_FS,
        pressure_coupling="anisotropic",
        hooks=_npt_wrap_hooks(),
    )


def _nvt(
    model: UMAWrapper,
    temperature_k: float,
    dt_fs: float,
    seed: int,
    device: torch.device,
    max_force: float | None = None,
) -> NVTLangevin:
    hooks: list = list(_npt_wrap_hooks())
    if max_force is not None:
        hooks.insert(0, MaxForceClampHook(max_force=max_force))
    return NVTLangevin(
        model=model,
        dt=dt_fs,
        temperature=torch.tensor([temperature_k], device=device),
        friction=LANGEVIN_FRICTION_PER_FS,
        random_seed=seed,
        hooks=hooks,
    )


def _sgc(
    model: UMAWrapper,
    temperature_k: float,
    delta_mu_ev: float,
    seed: int,
    device: torch.device,
) -> SGC:
    return SGC(
        model=model,
        temperature=torch.tensor([temperature_k], device=device),
        species=SPECIES,
        chemical_potentials={
            SPECIES[0]: torch.zeros(1, device=device),
            SPECIES[1]: torch.tensor([delta_mu_ev], device=device),
        },
        random_seed=seed,
    )


def _integrate(
    dynamics, batch: Batch, n_steps: int, chunk: int, on_chunk=None, mc=None, mc_steps=0
) -> None:
    """Run ``n_steps`` in chunks, calling ``on_chunk(steps_done)`` after each.
    ``on_chunk`` returning True stops early. With ``mc`` (an SGC sampler), each
    chunk is preceded by an ``mc_steps`` SGC block (hybrid MC-MD, the same loop
    as run_campaign._run_hybrid_with_observables)."""
    _wrap_batch_positions(batch)
    hybrid = (
        None
        if mc is None or mc_steps < 1
        else HybridMCMD(mc=mc, md=dynamics, mc_steps=mc_steps, md_steps=chunk, mc_energy_only=True)
    )
    with dynamics:
        dynamics.compute(batch)
        if hybrid is not None:
            hybrid.mc.synchronize(batch)
        done = 0
        while done < n_steps:
            k = min(chunk, n_steps - done)
            if hybrid is not None:
                hybrid.run_mc_block(batch)
                if hasattr(hybrid, "prepare_md_block"):
                    hybrid.prepare_md_block(batch)
                dynamics.compute(batch)
            dynamics.run(batch, n_steps=k)
            if hybrid is not None:
                hybrid.mc.synchronize(batch)
            done += k
            if on_chunk is not None and on_chunk(done):
                return


def _steps(ps: float, dt_fs: float) -> int:
    return max(1, int(round(ps * 1000.0 / dt_fs)))


# ----------------------------------------------------------------------------- main
def main() -> None:
    """Run one solid-liquid coexistence simulation and record its solid fraction."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--element", choices=sorted(EXPERIMENTAL_TM_K), required=True)
    ap.add_argument("--temperature-k", type=float, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument(
        "--nx", type=int, default=5, help="conventional cells along x and y (default 5)"
    )
    ap.add_argument(
        "--nz",
        type=int,
        default=5,
        help="conventional cells along z PER PHASE (default 5 -> 1000 atoms)",
    )
    ap.add_argument("--dt-fs", type=float, default=2.0)
    ap.add_argument("--solid-eq-ps", type=float, default=5.0)
    ap.add_argument("--melt-temperature-k", type=float, default=None)
    ap.add_argument("--melt-ps", type=float, default=5.0)
    ap.add_argument("--liquid-eq-ps", type=float, default=3.0)
    ap.add_argument(
        "--liquid-z-strain",
        type=float,
        default=0.05,
        help="z stretch of the liquid slab (liquid is less dense)",
    )
    ap.add_argument("--gap-ang", type=float, default=1.0)
    ap.add_argument("--clamp-ps", type=float, default=1.0)
    ap.add_argument(
        "--clamp-max-force",
        type=float,
        default=5.0,
        help="eV/A, interface relaxation only",
    )
    ap.add_argument("--production-ps", type=float, default=50.0)
    ap.add_argument("--snapshot-ps", type=float, default=0.5)
    ap.add_argument("--log-every-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument(
        "--delta-mu-ev",
        type=float,
        default=None,
        help="alloy mode: absolute Pt chemical potential (Au = 0) for hybrid SGC-MD in every "
        "stage after the hot melt; omit for a pure element",
    )
    ap.add_argument(
        "--mc-step-fraction",
        type=float,
        default=0.1,
        help="alloy mode: SGC trials per MD chunk, as a fraction of the atom count",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--inference-settings",
        default=INFERENCE_SETTINGS,
        help="UMA settings; a pure element has fixed composition, so merge_mole=true is valid "
        "(~2x faster MD); add activation_checkpointing=true for large cells on small cards",
    )
    args = ap.parse_args()

    device = torch.device(args.device)
    T, el, dt = args.temperature_k, args.element, args.dt_fs
    melt_T = args.melt_temperature_k or max(DEFAULT_MELT_TEMPERATURE_K[el], 1.8 * T)
    dmu = args.delta_mu_ev
    if dmu is not None and "merge_mole=true" in args.inference_settings.replace(" ", ""):
        raise SystemExit("--delta-mu-ev changes the composition: use merge_mole=false")
    out = args.out_dir / el / f"T{T:g}"
    out.mkdir(parents=True, exist_ok=True)
    t_wall = time.perf_counter()
    print(
        f"[setup] {el} T={T:g} K (exp. T_m {EXPERIMENTAL_TM_K[el]:g} K) nx={args.nx} nz/phase={args.nz} "
        f"dt={dt} fs melt_T={melt_T:g} K checkpoint={CHECKPOINT} task={TASK} device={device} "
        f"(single-graph batch, no multi-walker batching)"
        + ("" if dmu is None else f" SGC delta_mu={dmu:+.4f} eV, MC trials/chunk frac={args.mc_step_fraction}"),
        flush=True,
    )

    model = UMAWrapper.from_checkpoint(
        CHECKPOINT,
        task_name=TASK,
        device=str(device),
        inference_settings=args.inference_settings,
    )

    # --- 1. solid NPT
    solid0 = bulk(el, "fcc", cubic=True) * (args.nx, args.nx, args.nz)
    data = _data_from_arrays(
        solid0.numbers,
        solid0.positions,
        np.array(solid0.cell),
        None,
        T,
        args.seed,
        device,
    )
    batch = Batch.from_data_list([data])
    cells: list[np.ndarray] = []

    def _solid_log(done: int) -> bool:
        s = _snapshot(batch)
        cells.append(np.diag(s["cell"]).copy())
        if done % (10 * args.log_every_steps) == 0:
            L = np.diag(s["cell"])
            print(
                f"[solid] {done * dt / 1000:6.2f} ps  E/atom={s['energy'] / len(s['masses']):.4f}  "
                f"T={_temperature(s):6.0f}  L=({L[0]:.3f},{L[1]:.3f},{L[2]:.3f})",
                flush=True,
            )
        return False

    def _mc(seed: int) -> dict:
        """Hybrid SGC arguments for _integrate in alloy mode (none for a pure element)."""
        if dmu is None:
            return {}
        n = len(batch.atomic_numbers)
        return dict(mc=_sgc(model, T, dmu, seed, device), mc_steps=max(1, round(args.mc_step_fraction * n)))

    _integrate(
        _npt(model, T, dt, device),
        batch,
        _steps(args.solid_eq_ps, dt),
        args.log_every_steps,
        _solid_log,
        **_mc(args.seed + 100),
    )
    L_avg = np.mean(cells[len(cells) // 2 :], axis=0)
    solid = _snapshot(batch)
    a_lat = float(L_avg[:2].mean() / args.nx)
    f_solid = float(solid_like_mask(solid["positions"], solid["cell"]).mean())
    print(
        f"[solid] equilibrated: a={a_lat:.4f} A (x/y), Lz/nz={L_avg[2] / args.nz:.4f} A, solid-like={f_solid:.2f}",
        flush=True,
    )
    if f_solid < 0.5:
        print(
            f"[solid] WARNING: the pure crystal is only {f_solid:.0%} solid-like after {args.solid_eq_ps} ps -- "
            f"it melted without an interface, so T={T:g} K is above the superheating limit.",
            flush=True,
        )

    # Put the solid on the averaged box (affine), so both slabs share exact x/y.
    solid_cell = np.diag(L_avg)
    solid_pos = solid["positions"] @ np.linalg.inv(solid["cell"]) @ solid_cell

    # --- 2. liquid at fixed lateral box
    liq_cell = np.diag([L_avg[0], L_avg[1], L_avg[2] * (1.0 + args.liquid_z_strain)])
    liq_pos = solid_pos @ np.linalg.inv(solid_cell) @ liq_cell
    data = _data_from_arrays(
        solid["numbers"], liq_pos, liq_cell, None, melt_T, args.seed + 1, device
    )
    batch = Batch.from_data_list([data])
    melt_dt = min(dt, 1.0)
    for attempt in range(3):
        _integrate(
            _nvt(model, melt_T, melt_dt, args.seed + 2 + attempt, device),
            batch,
            _steps(args.melt_ps, melt_dt),
            args.log_every_steps,
        )
        s = _snapshot(batch)
        f_liq = float(solid_like_mask(s["positions"], s["cell"]).mean())
        print(
            f"[liquid] melt attempt {attempt + 1} at {melt_T:g} K: solid-like={f_liq:.3f}",
            flush=True,
        )
        if f_liq < 0.05:
            break
    else:
        raise RuntimeError(
            f"liquid slab still {f_liq:.0%} solid-like after 3 melts at {melt_T:g} K"
        )
    _integrate(
        _nvt(model, T, dt, args.seed + 10, device),
        batch,
        _steps(args.liquid_eq_ps, dt),
        args.log_every_steps,
        **_mc(args.seed + 110),
    )
    liquid = _snapshot(batch)
    f_liq = float(solid_like_mask(liquid["positions"], liquid["cell"]).mean())
    print(
        f"[liquid] quenched to {T:g} K: solid-like={f_liq:.3f}, T={_temperature(liquid):.0f} K",
        flush=True,
    )

    # --- 3. stitch solid | liquid along z
    def _wrapped(pos: np.ndarray, cell: np.ndarray) -> np.ndarray:
        fr = pos @ np.linalg.inv(cell)
        return (fr - np.floor(fr)) @ cell

    # fcc (001) planes sit at z = k*a/2, one of them exactly on z = 0. Shift by a/4
    # so the outermost solid planes are a/4 inside the slab at BOTH faces; otherwise
    # the top of the liquid meets the z = 0 plane across the periodic boundary at
    # only ~gap distance.
    layer = solid_cell[2, 2] / args.nz / 2.0
    sp = _wrapped(solid_pos + np.array([0.0, 0.0, layer / 2.0]), solid_cell)
    lp = _wrapped(liquid["positions"], liquid["cell"])
    lp[:, 2] += solid_cell[2, 2] + args.gap_ang
    box = np.diag(
        [L_avg[0], L_avg[1], solid_cell[2, 2] + liquid["cell"][2, 2] + 2 * args.gap_ang]
    )
    numbers = np.concatenate([solid["numbers"], liquid["numbers"]])
    positions = np.concatenate([sp, lp])
    velocities = np.concatenate([solid["velocities"], liquid["velocities"]])
    n_solid0 = len(sp)
    data = _data_from_arrays(
        numbers, positions, box, velocities, T, args.seed + 20, device
    )
    batch = Batch.from_data_list([data])
    _integrate(
        _nvt(model, T, dt / 2, args.seed + 21, device, max_force=args.clamp_max_force),
        batch,
        _steps(args.clamp_ps, dt / 2),
        args.log_every_steps,
    )
    s = _snapshot(batch)
    mask = solid_like_mask(s["positions"], s["cell"])
    print(
        f"[stitch] {len(numbers)} atoms, box=({box[0, 0]:.2f},{box[1, 1]:.2f},{box[2, 2]:.2f}) A, "
        f"solid-like after relaxation={mask.mean():.3f} (solid slab alone {n_solid0 / len(numbers):.3f})",
        flush=True,
    )

    # --- 4. production NPT
    series_path, traj_path = out / "series.csv", out / "trajectory.extxyz"
    traj_path.unlink(missing_ok=True)
    rows: list[list[float]] = []
    snaps: dict[str, list] = {
        "time_ps": [],
        "positions": [],
        "cell": [],
        "solid_mask": [],
        "pt": [],
    }
    snap_every = max(
        args.log_every_steps,
        int(round(_steps(args.snapshot_ps, dt) / args.log_every_steps))
        * args.log_every_steps,
    )
    state = {
        "f_s": float(mask.mean()),
        "streak": 0,
        "verdict": "coexisting (undecided at end of run)",
    }

    def _prod_log(done: int) -> bool:
        s = _snapshot(batch)
        t_ps = done * dt / 1000.0
        L = np.diag(s["cell"])
        if done % snap_every == 0:
            m = solid_like_mask(s["positions"], s["cell"])
            state["f_s"] = float(m.mean())
            snaps["time_ps"].append(t_ps)
            snaps["positions"].append(s["positions"].astype(np.float32))
            snaps["cell"].append(s["cell"])
            snaps["solid_mask"].append(m)
            snaps["pt"].append(s["numbers"] == SPECIES[1])
            frame = Atoms(
                numbers=s["numbers"], positions=s["positions"], cell=s["cell"], pbc=True
            )
            frame.arrays["solid_like"] = m.astype(int)
            ase_write(traj_path, frame, format="extxyz", append=True)
            np.savez_compressed(
                out / "snapshots.npz", **{k: np.asarray(v) for k, v in snaps.items()}
            )
            lo, hi = SOLID_FRACTION_DONE
            done_state = (
                "melted"
                if state["f_s"] < lo
                else "frozen"
                if state["f_s"] > hi
                else None
            )
            state["streak"] = (
                state["streak"] + 1
                if done_state and done_state == state.get("last")
                else (1 if done_state else 0)
            )
            state["last"] = done_state
            print(
                f"[prod] {t_ps:6.2f} ps  E/atom={s['energy'] / len(s['masses']):.4f}  T={_temperature(s):6.0f}  "
                f"L=({L[0]:.2f},{L[1]:.2f},{L[2]:.2f})  solid-like={state['f_s']:.3f}"
                + ("" if dmu is None else f"  x_Pt solid/liquid={(s['numbers'][m] == SPECIES[1]).mean():.3f}/"
                   f"{(s['numbers'][~m] == SPECIES[1]).mean() if (~m).any() else float('nan'):.3f}"),
                flush=True,
            )
            if state["streak"] >= 3:
                state["verdict"] = f"fully {done_state} (stopped early)"
        rows.append(
            [t_ps, s["energy"] / len(s["masses"]), _temperature(s), *L, state["f_s"]]
        )
        return state["streak"] >= 3

    _integrate(
        _npt(model, T, dt, device),
        batch,
        _steps(args.production_ps, dt),
        args.log_every_steps,
        _prod_log,
        **_mc(args.seed + 120),
    )
    with series_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "time_ps",
                "epot_per_atom_ev",
                "temperature_k",
                "lx_ang",
                "ly_ang",
                "lz_ang",
                "solid_fraction",
            ]
        )
        w.writerows(rows)

    t = np.asarray(snaps["time_ps"])
    f = np.array([m.mean() for m in snaps["solid_mask"]])
    early = t <= min(10.0, t.max()) if len(t) else t
    rate = (
        float(np.polyfit(t[early], f[early], 1)[0])
        if early.sum() >= 3
        else float("nan")
    )
    final = snaps["solid_mask"][-1].mean() if snaps["solid_mask"] else float("nan")
    # Phase compositions over the second half of the snapshots (solid-like vs liquid-like atoms).
    half = slice(len(snaps["pt"]) // 2, None)
    def _x(sel: bool) -> float:
        num = sum(int((p & (m == sel)).sum()) for p, m in zip(snaps["pt"][half], snaps["solid_mask"][half]))
        den = sum(int((m == sel).sum()) for m in snaps["solid_mask"][half])
        return num / den if den else float("nan")
    x_solid, x_liquid = _x(True), _x(False)
    x_total = float(np.mean([p.mean() for p in snaps["pt"][half]])) if snaps["pt"] else float("nan")
    verdict = state["verdict"]
    if verdict.startswith("coexisting"):
        verdict = (
            "solid growing" if rate > 0 else "solid shrinking" if rate < 0 else verdict
        )
    summary = dict(
        element=el,
        temperature_k=T,
        experimental_tm_k=EXPERIMENTAL_TM_K[el],
        checkpoint=CHECKPOINT,
        task=TASK,
        inference_settings=args.inference_settings,
        n_atoms=int(len(numbers)),
        nx=args.nx,
        nz_per_phase=args.nz,
        dt_fs=dt,
        solid_lattice_a_ang=a_lat,
        solid_eq_solid_fraction=f_solid,
        liquid_solid_fraction=f_liq,
        solid_fraction_initial=float(mask.mean()),
        solid_fraction_final=float(final),
        solid_fraction_rate_per_ps_first_10ps=rate,
        production_ps_run=float(t.max()) if len(t) else 0.0,
        verdict=verdict,
        delta_mu_ev=dmu,
        mc_step_fraction=args.mc_step_fraction if dmu is not None else 0.0,
        x_pt_solid=x_solid,
        x_pt_liquid=x_liquid,
        x_pt_total=x_total,
        wall_minutes=(time.perf_counter() - t_wall) / 60,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[done] {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
