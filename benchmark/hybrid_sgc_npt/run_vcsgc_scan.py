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

Walkers come from ``--plan`` (a JSON list, run in that order) or ``--c0``. Each has a c0, a
starting state and its own block count:

- ``slab``: two slabs along z in lever-rule proportions, Pt-rich at low z. With
  ``--seed-phase-x X_ALPHA X_GAMMA`` each slab already holds its phase's measured solute fraction
  (randomly placed), so the walker starts close to two-phase equilibrium instead of from pure
  slabs whose solute content transmutations would have to build up. Without it the slabs are pure.
- ``random``: a homogeneous random alloy at c0 (hysteresis check against ``slab``).
- ``checkpoint``: an equilibrated state from another run (e.g. the SGC-NPT scan), given as
  ``"state": "<path>.pt"``; c0 defaults to that state's composition, so the walker starts at its
  target, already relaxed.

Fresh walkers start at the Vegard lattice constant from the pure-element calibration.

The walkers run in serial batches of ``--batch-width`` (``auto``: 4 on cards with >= 75 GB, else
3 -- sized on peak allocated memory, see resolve_width; run with expandable_segments). Within a batch, walkers may have
different c0 and progress. Every ``--chunk-blocks`` blocks each walker's state goes to
``<out>/states`` and its per-block series (c, U/atom, V/atom) to ``<out>/<run_id>.series.json``;
a resubmission continues where the last job stopped, finishing partly-run walkers first.

Live equilibration gate: after every chunk, once a walker has ``min_blocks`` blocks, its
composition AND energy series are tested with the scan's gate (``run_campaign._equilibration_gate``:
the means of the last two ``EQUILIBRATION_WINDOW_BLOCKS``-block windows agree within twice their
combined standard error). A walker that passes ``--gate-passes`` consecutive checks has stopped
evolving and is stopped (``"stopped": "equilibrated"`` in its series); one that never passes
runs to its ``n_blocks`` cap (``"stopped": "cap"``) and is flagged by the analysis. A batch whose
walker stops early keeps running its other walkers below full width.
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
    EQUILIBRATION_WINDOW_BLOCKS,
    INFERENCE_SETTINGS,
    KB_EV,
    PRESSURE_EV_PER_A3,
    SEED,
    SIZE_REPEATS,
    SPECIES,
    TASK,
    THERMOSTAT_TIME_FS,
    _cell_volumes,
    _equilibration_gate,
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
INITS = ("slab", "random", "checkpoint")


def run_id_for(n_atoms: int, temperature: float, kappa: float, c0: float, init: str) -> str:
    tag = {"checkpoint": "ckpt"}.get(init, init)
    return f"atoms{n_atoms}.T{temperature:g}.vcsgc.k{kappa:g}.c{c0:.3f}.{tag}"


def load_state(path: str | Path, device) -> AtomicData:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    return AtomicData.model_validate(payload["state"] if "state" in payload else payload).to(device)


def state_composition(path: str | Path) -> float:
    data = load_state(path, "cpu")
    return float((data.atomic_numbers == SPECIES[1]).double().mean())


def slab_sites(z: torch.Tensor, c0: float, n_atoms: int, seed_x, generator) -> torch.Tensor:
    """Pt site indices for a two-slab start at overall Pt fraction round(c0 N) / N.

    ``seed_x = (x_alpha, x_gamma)``: the Pt-rich slab (lowest z) takes the lever-rule share
    (c0 - x_alpha) / (x_gamma - x_alpha) of the atoms; each slab then gets its phase's Pt fraction,
    placed at random within it, with the remainder put in the Au-rich slab so the total is exact.
    ``seed_x = None``: pure slabs (the lowest-z round(c0 N) sites are Pt).
    """
    n_pt = round(c0 * n_atoms)
    jitter = torch.rand(n_atoms, generator=generator, dtype=z.dtype) * 1e-3  # random within a plane
    order = torch.argsort(z + jitter)
    if seed_x is None:
        return order[:n_pt]
    xa, xg = seed_x
    frac_gamma = min(max((c0 - xa) / (xg - xa), 0.0), 1.0)
    n_gamma = round(frac_gamma * n_atoms)
    gamma, alpha = order[:n_gamma], order[n_gamma:]
    n_pt_gamma = min(round(xg * n_gamma), n_pt)
    n_pt_alpha = min(n_pt - n_pt_gamma, len(alpha))
    n_pt_gamma = n_pt - n_pt_alpha
    pick_g = gamma[torch.randperm(len(gamma), generator=generator)[:n_pt_gamma]]
    pick_a = alpha[torch.randperm(len(alpha), generator=generator)[:n_pt_alpha]]
    return torch.cat([pick_g, pick_a])


def initial_state(
    n_atoms: int, temperature: float, c0: float, init: str, lattice: dict, seed: int, device, seed_x=None
) -> AtomicData:
    """Fresh walker: slab or random Pt placement at round(c0 N), Vegard cell, MB velocities."""
    a = (1 - c0) * lattice[SPECIES[0]] + c0 * lattice[SPECIES[1]]
    atoms = build_ase_structure("Au", CRYSTAL_STRUCTURE, a, SIZE_REPEATS[n_atoms], cubic=True)
    data = AtomicData.from_atoms(atoms, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if init == "slab":
        z = torch.as_tensor(atoms.get_scaled_positions(wrap=True)[:, 2])
        pt_sites = slab_sites(z, c0, n_atoms, seed_x, generator)
    else:
        pt_sites = torch.randperm(n_atoms, generator=generator)[: round(c0 * n_atoms)]
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


def resolve_width(arg: str, device) -> int:
    if arg != "auto":
        return int(arg)
    if device.type != "cuda":
        return 2
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    # Peak ALLOCATED SGC-NPT memory is ~1.1 + 5.2 GiB per 500-atom walker (6.3 / 11.5 / 21.8 GiB at
    # width 1 / 2 / 4); the 48.7 GiB *reserved* at width 4 is allocator caching, which
    # expandable_segments removes. So 3 fits a 40 GB card and 4 an 80 GB one.
    return 4 if total >= 75 else 3


def plan_entries(args) -> list[dict]:
    if args.plan:
        entries = json.loads(args.plan.read_text())
    else:
        entries = [{"c0": c, "init": args.init} for c in args.c0]
    out = []
    for e in entries:
        init = e.get("init", "auto")
        if init == "checkpoint":
            c0 = e.get("c0", state_composition(e["state"]))
        else:
            c0 = float(e["c0"])
            if init == "auto":
                init = "slab" if args.slab_range[0] <= c0 <= args.slab_range[1] else "random"
        if init not in INITS:
            raise SystemExit(f"unknown init {init!r} in plan entry {e}")
        out.append(dict(
            c0=float(c0), init=init, state=e.get("state"),
            n_blocks=int(e.get("n_blocks", args.n_blocks)),
            min_blocks=int(e.get("min_blocks", args.min_blocks)),
        ))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="checkpoint root for this VC-SGC campaign")
    walkers_arg = ap.add_mutually_exclusive_group(required=True)
    walkers_arg.add_argument("--plan", type=Path,
                             help='JSON list of {"c0", "init": slab|random|checkpoint|auto, "state", "n_blocks", "min_blocks"}, run in order')
    walkers_arg.add_argument("--c0", type=float, nargs="+", help="target Pt fractions, one walker each (all --init)")
    ap.add_argument("--init", choices=["auto", "slab", "random"], default="auto", help="with --c0")
    ap.add_argument("--slab-range", type=float, nargs=2, default=(0.08, 0.92),
                    help="init auto: slab start for c0 inside this range, random outside")
    ap.add_argument("--seed-phase-x", type=float, nargs=2, metavar=("X_ALPHA", "X_GAMMA"),
                    help="Pt fraction of the Au-rich and Pt-rich slabs in a slab start (default: pure slabs)")
    ap.add_argument("--temperature-k", type=float, default=700.0)
    ap.add_argument("--kappa", type=float, default=0.5, help="eV, intensive (VCSGC docstring)")
    ap.add_argument("--n-atoms", type=int, default=500)
    ap.add_argument("--n-blocks", type=int, default=300,
                    help="default cap on blocks per walker; the gate usually stops it earlier (plan entries may override)")
    ap.add_argument("--min-blocks", type=int, default=100,
                    help="default blocks before the gate may stop a walker (plan entries may override)")
    ap.add_argument("--gate-passes", type=int, default=2,
                    help="consecutive passing gate checks (one per chunk) needed to stop a walker")
    ap.add_argument("--chunk-blocks", type=int, default=25, help="blocks between checkpoints")
    ap.add_argument("--batch-width", default="auto", help="walkers per batch, or auto (4 on >= 75 GB cards, else 3)")
    ap.add_argument("--mc-step-fraction", type=float, default=0.6)
    ap.add_argument("--md-steps-per-block", type=int, default=50)
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
    for e in plan_entries(args):
        rid = run_id_for(args.n_atoms, T, args.kappa, e["c0"], e["init"])
        spath = args.out / f"{rid}.series.json"
        series = json.loads(spath.read_text()) if spath.is_file() else None
        if series is None:
            series = dict(
                run_id=rid, temperature_K=T, c0=e["c0"], kappa=args.kappa, init=e["init"], start_state=e["state"],
                seed_phase_x=args.seed_phase_x if e["init"] == "slab" else None, n_atoms=args.n_atoms,
                delta_mu_ref_eV=dmu_ref, mc_step_fraction=args.mc_step_fraction,
                md_steps_per_block=args.md_steps_per_block, inference_settings=args.inference_settings,
                checkpoint=CHECKPOINT, task=TASK, c=[], u=[], v=[], acceptance=[], gate=[], stopped=None,
            )
        elif (series["kappa"], series["delta_mu_ref_eV"], series["inference_settings"]) != (
            args.kappa, dmu_ref, args.inference_settings
        ):
            raise SystemExit(f"{rid}: existing series was made with different kappa/dmu_ref/settings")
        if any(w["rid"] == rid for w in walkers):
            raise SystemExit(f"duplicate walker {rid} in the plan")
        walkers.append(dict(rid=rid, path=spath, series=series, **e))

    width = resolve_width(args.batch_width, device)
    model = UMAWrapper.from_checkpoint(
        CHECKPOINT, task_name=TASK, device=str(device), inference_settings=args.inference_settings
    )
    mc_steps = max(1, round(args.mc_step_fraction * args.n_atoms))
    total = sum(w["n_blocks"] for w in walkers)  # upper bound: the gate stops most walkers earlier
    print(
        f"[vcsgc] T={T:g} K kappa={args.kappa} eV dmu_ref={dmu_ref:.5f} eV, {len(walkers)} walkers, "
        f"{sum(len(w['series']['c']) for w in walkers)}/{total} walker-blocks done, batch width {width}, "
        f"{mc_steps} MC + {args.md_steps_per_block} MD per block, seed phases {args.seed_phase_x}, "
        f"settings {args.inference_settings!r}, out {args.out}",
        flush=True,
    )
    for w in walkers:
        print(f"[vcsgc]   {w['rid']}  blocks {len(w['series']['c'])}/{w['n_blocks']} (min {w['min_blocks']})  "
              f"start {w['state'] or w['init']}  {w['series'].get('stopped') or ''}", flush=True)

    def finished(w) -> bool:
        return w["series"].get("stopped") is not None or len(w["series"]["c"]) >= w["n_blocks"]

    def fresh(w) -> bool:
        # Fresh-built states carry fewer fields than saved ones (e.g. MC bookkeeping), so the two
        # kinds are never mixed in one Batch.
        return len(w["series"]["c"]) == 0 and w["init"] != "checkpoint"

    while True:
        todo = [w for w in walkers if not finished(w)]
        if not todo:
            break
        # Partly-run walkers first (a resubmission finishes them), then plan order.
        todo.sort(key=lambda w: len(w["series"]["c"]) == 0)
        kind = fresh(todo[0])
        active = [w for w in todo if fresh(w) == kind][:width]
        chunk = min(args.chunk_blocks, *(w["n_blocks"] - len(w["series"]["c"]) for w in active))
        states = []
        for w in active:
            done = len(w["series"]["c"])
            if done > 0:
                states.append(store.load(w["rid"], device=device))
            elif w["init"] == "checkpoint":
                states.append(load_state(w["state"], device))
            else:
                seed_x = args.seed_phase_x if w["init"] == "slab" else None
                states.append(initial_state(args.n_atoms, T, w["c0"], w["init"], lattice,
                                            SEED + int(1000 * w["c0"]), device, seed_x))
        batch = Batch.from_data_list(states)
        _wrap_batch_positions(batch)
        # A new MC stream per chunk, so resumed walkers never replay an earlier stream.
        done_total = sum(len(w["series"]["c"]) for w in walkers)
        hybrid = build_hybrid(
            model, len(active), T, [w["c0"] for w in active], args.kappa, dmu_ref,
            mc_steps, args.md_steps_per_block, SEED + done_total, device,
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
            n_done = len(s["c"])
            if n_done >= max(w["min_blocks"], 2 * EQUILIBRATION_WINDOW_BLOCKS):
                gc = _equilibration_gate(s["c"], EQUILIBRATION_WINDOW_BLOCKS)
                ge = _equilibration_gate(s["u"], EQUILIBRATION_WINDOW_BLOCKS)
                passed = bool(gc.get("resolved")) and bool(ge.get("resolved"))
                s.setdefault("gate", []).append(dict(
                    blocks=n_done, passed=passed,
                    c_difference=gc.get("difference"), c_combined_se=gc.get("combined_standard_error"),
                    u_difference=ge.get("difference"), u_combined_se=ge.get("combined_standard_error"),
                ))
                recent = s["gate"][-args.gate_passes:]
                if len(recent) == args.gate_passes and all(g["passed"] for g in recent):
                    s["stopped"] = "equilibrated"
            if s.get("stopped") is None and n_done >= w["n_blocks"]:
                s["stopped"] = "cap"
            tmp = w["path"].with_suffix(".tmp")
            tmp.write_text(json.dumps(s) + "\n")
            tmp.replace(w["path"])
        done_total += len(active) * chunk
        for w in active:
            if w["series"].get("stopped"):
                g = w["series"]["gate"][-1] if w["series"].get("gate") else None
                print(
                    f"[vcsgc] stop {w['rid']} after {len(w['series']['c'])} blocks: {w['series']['stopped']}"
                    + (f" (last gate: dc={g['c_difference']:+.4f} vs 2SE {2 * g['c_combined_se']:.4f}, "
                       f"dU={g['u_difference']:+.5f} vs 2SE {2 * g['u_combined_se']:.5f})" if g and g["c_difference"] is not None else ""),
                    flush=True,
                )
        print(
            f"[vcsgc] {done_total}/{total} walker-blocks; chunk of {chunk} at width {len(active)}: {elapsed:.0f} s "
            f"({len(active) * chunk / elapsed:.4f} walker-blocks/s), acceptance {acceptance:.4f}; "
            + ", ".join(
                f"c0={w['c0']:.3f} ({w['init']}, {len(w['series']['c'])}/{w['n_blocks']}"
                f"{', gate pass' if w['series'].get('gate') and w['series']['gate'][-1]['passed'] else ''}): "
                f"c={sum(w['series']['c'][-chunk:]) / chunk:.4f}"
                for w in active
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
