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
"""Gated SGC / SGC-NPT delta_mu ladders started from existing checkpoints.

Refines a delta_mu scan where the first pass was too coarse: each *chain* is one walker that
starts from a saved state (e.g. a scan checkpoint) and steps through its own list of delta_mu
values. At every step it runs until the composition AND energy series pass the scan's
equilibration gate (``run_campaign._equilibration_gate``: last two 25-block windows agree within
2 combined SE) on ``gate_passes`` consecutive checks, with at least ``min_blocks`` and at most
``max_blocks`` blocks, then moves to the next delta_mu warm-started from where it is. All chains
of a plan run side by side in one batch (one walker each, possibly at different delta_mu and
step); a chain that finishes drops out and the batch narrows.

Plan (JSON):
    {"label": "...", "temperature_k": 700, "md_steps_per_block": 0, "mc_step_fraction": 0.2,
     "min_blocks": 100, "max_blocks": 300, "gate_passes": 2, "chunk_blocks": 25,
     "reference_energies": "<auto_reference_energies.json>",   # delta_mu_ref for "mu_excess"
     "chains": [{"name": "Afine", "branch": "Arich", "start": "<state>.pt",
                 "mu_excess": [-0.045, -0.0425, ...]}, ...]}

Output, one directory per chain under ``--out``: a ``*.equilibration.json`` per finished step in
run_campaign's format (so ``sgc_phase_boundary.py`` reads a chain's directory directly; the
run_id carries the ``branch`` label and step ``dmuN``), the step's final state in
``<out>/states``, and ``progress.json`` (current step, its per-block series and gate history).
Resumable: a resubmission continues every chain from its last 25-block chunk.

Each chunk draws a fresh MC random stream (``SEED`` + chunk count), so a resumed or long step
never replays the proposals and acceptance draws of an earlier chunk.

Optional plan keys:
    "volume_relax": {"bulk_modulus_ev_per_a3": 1.2, "tolerance_ev_per_a3": 2e-4, "max_iterations": 4}
        Rigid-lattice SGC with a relaxed lattice constant (use with ``md_steps_per_block: 0``):
        after every block each walker's cell and positions are scaled together until P ~ 0
        (``run_campaign.relax_cell_volume``), so atoms stay on ideal sites but the lattice constant
        follows the composition. The per-block lattice constant is recorded (``a_series``).
    "static_references": true
        Before the ladders, relax pure Au and pure Pt the same way and write their static energies
        (eV/atom) to ``<out>/static_references.json``, and as bare ``{"<T>": {"A": E_Au, "B": E_Pt}}``
        to ``<out>/static_pure_free_energies.json`` (sgc_phase_boundary.py's ``--pure-free-energies``
        format): with no vibrations and no
        configurational entropy in a pure crystal these ARE the pure-end free energies, so the
        phi integration can be anchored exactly.
    "stop_after_jump": {"dx": 0.25, "extra_steps": 2}
        A chain whose step-to-step x changes by more than ``dx`` (it fell onto the other phase)
        runs ``extra_steps`` more steps (overlap for the hysteresis bracket), then stops.
A chain's "start" may be ``"pure:Au"`` / ``"pure:Pt"`` (a fresh pure crystal on the template
lattice) instead of a checkpoint path.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from run_campaign import (
    CHECKPOINT,
    CONVENTIONAL_CELL,
    CRYSTAL_STRUCTURE,
    EQUILIBRATION_WINDOW_BLOCKS,
    INFERENCE_SETTINGS,
    LATTICE_A_ANG,
    MC_ENERGY_ONLY,
    PRESSURE_EV_PER_A3,
    SEED,
    SIZE_REPEATS,
    SPECIES,
    TASK,
    TEMPLATE_SYMBOL,
    _equilibration_gate,
    _run_hybrid_with_observables,
    build_ase_structure,
    lattice_constant_per_graph,
    make_workload,
    relax_cell_volume,
)

from nvalchemi.data import AtomicData
from nvalchemi.models.uma import UMAWrapper
from nvalchemi.scheduling import FinalStateStore, RunSpec


def load_state(path: str | Path, device) -> AtomicData:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    return AtomicData.model_validate(payload["state"] if "state" in payload else payload).to(device)


PURE_START = {"pure:Au": 0.0, "pure:Pt": 1.0}  # chain "start" -> Pt fraction of a fresh crystal


def step_run_id(n_atoms: int, T: float, chain: dict, step: int, mu: float) -> str:
    return f"atoms{n_atoms}.T{T:g}.{chain['branch']}.{chain['name']}.dmu{step}.mu{mu:.5f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-atoms", type=int, default=500)
    ap.add_argument("--inference-settings", default=INFERENCE_SETTINGS)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    plan = json.loads(args.plan.read_text())
    T = float(plan["temperature_k"])
    md_steps = int(plan["md_steps_per_block"])
    mc_fraction = float(plan["mc_step_fraction"])
    min_blocks, max_blocks = int(plan["min_blocks"]), int(plan["max_blocks"])
    gate_passes, chunk = int(plan.get("gate_passes", 2)), int(plan.get("chunk_blocks", EQUILIBRATION_WINDOW_BLOCKS))
    ref = json.loads(Path(plan["reference_energies"]).read_text())["reference"][f"{T:g}"]["delta_mu_ref_eV"]
    device = torch.device(args.device)
    volume_relax = plan.get("volume_relax")
    stop_after_jump = plan.get("stop_after_jump")

    args.out.mkdir(parents=True, exist_ok=True)
    store = FinalStateStore(args.out / "states")
    chains = []
    for c in plan["chains"]:
        cdir = args.out / c["name"]
        cdir.mkdir(exist_ok=True)
        ppath = cdir / "progress.json"
        mus = [ref + float(e) for e in c["mu_excess"]]
        prog = json.loads(ppath.read_text()) if ppath.is_file() else dict(
            name=c["name"], branch=c["branch"], start=c["start"], mu=mus, step=0,
            x=[], e=[], a=[], gate=[], chunks=0, finished=[], settings=args.inference_settings,
        )
        if prog["mu"] != mus or prog["settings"] != args.inference_settings:
            raise SystemExit(f"{c['name']}: plan or settings differ from {ppath}; use a fresh --out")
        chains.append(dict(cfg=c, dir=cdir, path=ppath, prog=prog))

    template = build_ase_structure(TEMPLATE_SYMBOL, CRYSTAL_STRUCTURE, LATTICE_A_ANG, SIZE_REPEATS[args.n_atoms],
                                   cubic=CONVENTIONAL_CELL)
    model = UMAWrapper.from_checkpoint(CHECKPOINT, task_name=TASK, device=str(device),
                                       inference_settings=args.inference_settings)

    def workload(runs, parents):
        """make_workload plus, for volume_relax plans, a cell hook logging each block's lattice constant."""
        hybrid, batch = make_workload(model, template, tuple(runs), tuple(parents), device, md_steps, mc_fraction)
        if not volume_relax:
            return hybrid, batch, None, []
        log: list[tuple[list[float], float]] = []

        def hook(b):
            pressure = relax_cell_volume(hybrid.md, b, **volume_relax)
            log.append((lattice_constant_per_graph(b), float(pressure.abs().max())))

        return hybrid, batch, hook, log

    static_path = args.out / "static_references.json"
    if plan.get("static_references") and not static_path.is_file():
        if not volume_relax:
            raise SystemExit("static_references needs volume_relax (the references must be relaxed the same way)")
        refs = [RunSpec(run_id=f"static_{sym}", temperature_k=T, pressure_ev_per_a3=PRESSURE_EV_PER_A3,
                        chemical_potentials_ev={SPECIES[0]: 0.0, SPECIES[1]: ref}, species=SPECIES,
                        batch_group="ladder", metadata={"pt_fraction": frac})
                for sym, frac in (("Au", 0.0), ("Pt", 1.0))]
        hybrid, batch, hook, log = workload(refs, (None, None))
        with hybrid.md:
            hybrid.md.compute(batch)
            for _ in range(5):  # from the template lattice constant: a few Newton passes
                hook(batch)
        e = [float(v) / args.n_atoms for v in batch.energy.flatten().tolist()]
        a, p_max = log[-1]
        static = {f"{T:g}": {"A": e[0], "B": e[1]},
                  "lattice_constant_a_ang": {"Au": a[0], "Pt": a[1]}, "max_abs_pressure_ev_per_a3": p_max,
                  "delta_mu_ref_static_eV": e[1] - e[0], "delta_mu_ref_plan_eV": ref,
                  "note": "static, volume-relaxed pure fcc energies per atom on ideal sites; with no "
                          "vibrations these are the exact pure-end free energies of this model"}
        static_path.write_text(json.dumps(static, indent=1) + "\n")
        # bare {"T": {"A", "B"}} copy for sgc_phase_boundary.py --pure-free-energies
        (args.out / "static_pure_free_energies.json").write_text(json.dumps({f"{T:g}": static[f"{T:g}"]}) + "\n")
        print(f"[ladder] static references: E_Au={e[0]:.5f} (a={a[0]:.4f}) E_Pt={e[1]:.5f} (a={a[1]:.4f}) "
              f"eV/atom, static dmu_ref={e[1] - e[0]:.5f} vs plan dmu_ref {ref:.5f}; |P|max {p_max:.1e}", flush=True)
        del hybrid, batch

    print(
        f"[ladder] {plan['label']}: T={T:g} K dmu_ref={ref:.5f} eV md_steps/block={md_steps} "
        f"mc_step_fraction={mc_fraction} blocks/step {min_blocks}-{max_blocks} (gate x2 windows of "
        f"{EQUILIBRATION_WINDOW_BLOCKS}, {gate_passes} passes), mc_energy_only={MC_ENERGY_ONLY}, "
        f"settings {args.inference_settings!r}, volume_relax={volume_relax}, stop_after_jump={stop_after_jump}",
        flush=True,
    )
    for ch in chains:
        p = ch["prog"]
        print(f"[ladder]   {p['name']} ({p['branch']}): step {p['step']}/{len(p['mu'])}, "
              f"{len(p['x'])} blocks into it; start {p['start']}", flush=True)

    def save(ch) -> None:
        tmp = ch["path"].with_suffix(".tmp")
        tmp.write_text(json.dumps(ch["prog"]) + "\n")
        tmp.replace(ch["path"])

    n_chunks = sum(ch["prog"]["chunks"] for ch in chains)
    while True:
        active = [ch for ch in chains if ch["prog"]["step"] < len(ch["prog"]["mu"]) and not ch["prog"].get("done")]
        if not active:
            break
        runs, parents = [], []
        for ch in active:
            p = ch["prog"]
            current = f"{p['name']}.current"
            if store.exists(current):
                parents.append(store.load(current, device=device))
            else:
                parents.append(None if p["start"] in PURE_START else load_state(p["start"], device))
            runs.append(RunSpec(
                run_id=f"{p['name']}.s{p['step']}", temperature_k=T, pressure_ev_per_a3=PRESSURE_EV_PER_A3,
                chemical_potentials_ev={SPECIES[0]: 0.0, SPECIES[1]: p["mu"][p["step"]]},
                species=SPECIES, batch_group="ladder", metadata={"pt_fraction": PURE_START.get(p["start"], 0.5)},
            ))
        hybrid, batch, hook, a_log = workload(runs, parents)
        hybrid.mc._random_seed = SEED + 7919 * (n_chunks + 1)  # fresh stream per chunk (generator is created lazily)
        start = time.perf_counter()
        result, xs, es = _run_hybrid_with_observables(hybrid, batch, chunk, len(active), SPECIES[1], cell_hook=hook)
        a_blocks = [a for a, _ in a_log[1:]]  # entry 0 is the relaxation before the first block
        elapsed = time.perf_counter() - start
        acceptance = hybrid.mc.stats.acceptance
        n_chunks += 1
        notes = []
        for i, (ch, final) in enumerate(zip(active, result.to_data_list())):
            p = ch["prog"]
            p["chunks"] += 1
            store.save(f"{p['name']}.current", final)
            p["x"].extend(b[i] for b in xs)
            p["e"].extend(b[i] for b in es)
            p.setdefault("a", []).extend(b[i] for b in a_blocks)
            n = len(p["x"])
            gx, ge = {}, {}
            if n >= max(min_blocks, 2 * EQUILIBRATION_WINDOW_BLOCKS):
                gx = _equilibration_gate(p["x"], EQUILIBRATION_WINDOW_BLOCKS)
                ge = _equilibration_gate(p["e"], EQUILIBRATION_WINDOW_BLOCKS)
                p["gate"].append(dict(blocks=n, passed=bool(gx.get("resolved")) and bool(ge.get("resolved"))))
            recent = p["gate"][-gate_passes:]
            equilibrated = len(recent) == gate_passes and all(g["passed"] for g in recent)
            if equilibrated or n >= max_blocks:
                mu = p["mu"][p["step"]]
                rid = step_run_id(args.n_atoms, T, p, p["step"], mu)
                record = {
                    "run_id": rid, "temperature_K": T,
                    "chemical_potentials_ev": {"Au": 0.0, "Pt": mu},
                    "delta_mu_ref_eV": ref, "delta_mu_excess_eV": mu - ref,
                    "n_blocks": n, "window_blocks": EQUILIBRATION_WINDOW_BLOCKS,
                    "composition_gate": gx, "energy_gate": ge,
                    "resolved": bool(equilibrated), "stopped": "equilibrated" if equilibrated else "cap",
                    "gate_history": p["gate"], "acceptance_last_chunk": acceptance,
                    "md_steps_per_block": md_steps, "mc_step_fraction": mc_fraction,
                    "chain": p["name"], "start_state": p["start"],
                    "x_series": p["x"], "e_series": p["e"],
                    "volume_relax": volume_relax, "a_series": p["a"],
                    "a_mean_last_window": (sum(p["a"][-EQUILIBRATION_WINDOW_BLOCKS:]) / len(p["a"][-EQUILIBRATION_WINDOW_BLOCKS:])
                                           if p["a"] else None),
                }
                (ch["dir"] / f"{rid}.equilibration.json").write_text(json.dumps(record) + "\n")
                store.save(rid, final)
                p["finished"].append(dict(step=p["step"], mu_excess=mu - ref, x=gx.get("mean_last_window"),
                                          blocks=n, stopped=record["stopped"]))
                notes.append(f"{p['name']} step {p['step']} (dmu_ex {mu - ref:+.4f}) done after {n} blocks, "
                             f"{record['stopped']}: x={gx.get('mean_last_window', float('nan')):.4f}")
                if stop_after_jump and len(p["finished"]) >= 2 and "jump_step" not in p:
                    x0, x1 = p["finished"][-2]["x"], p["finished"][-1]["x"]
                    if x0 is not None and x1 is not None and abs(x1 - x0) > float(stop_after_jump["dx"]):
                        p["jump_step"] = p["step"]
                        notes.append(f"{p['name']} jumped x {x0:.3f} -> {x1:.3f} at step {p['step']}; "
                                     f"{int(stop_after_jump['extra_steps'])} more step(s) then stop")
                if "jump_step" in p and p["step"] - p["jump_step"] >= int(stop_after_jump["extra_steps"]):
                    p["done"] = True
                p["step"] += 1
                p["x"], p["e"], p["a"], p["gate"] = [], [], [], []
            save(ch)
        print(
            f"[ladder] chunk {n_chunks}: width {len(active)}, {chunk} blocks in {elapsed:.0f} s "
            f"({len(active) * chunk / elapsed:.4f} walker-blocks/s), acceptance {acceptance:.4f}"
            + (f", a {min(a_blocks[-1]):.4f}-{max(a_blocks[-1]):.4f} A, |P|max {a_log[-1][1]:.1e}" if a_blocks else "")
            + "; "
            + ", ".join(f"{ch['prog']['name']}@{ch['prog']['step']}: x={ch['prog']['x'][-1] if ch['prog']['x'] else float('nan'):.4f}"
                        for ch in active),
            flush=True,
        )
        for note in notes:
            print(f"[ladder]   {note}", flush=True)
        del hybrid, batch, result
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
    print(f"[ladder] {plan['label']}: complete", flush=True)


if __name__ == "__main__":
    main()
