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
"""One-GPU phase-diagram campaign: every kind of walker in one pool that keeps the GPU full.

A campaign (JSON spec) mixes four kinds of work, all at any number of temperatures:

``ladders``  gated delta_mu ladders with continuation (hysteresis-sensitive scans): one walker
             steps through its own delta_mu list, warm-started from step to step.
``points``   fan-out: one independent walker per delta_mu, all started from the same parent
             state (single-phase regions, where continuation is optional).
``vcsgc``    variance-constrained walkers at fixed target composition c0 (inside a gap).
``traces``   boundary tracing (eq. 29 of van de Walle & Asta 2002) with ``boundary_tracer.py``,
             faithful to their section 3.3 when paired with a retrace: an upward trace finds the
             end of the two-phase region with recentering, then a trace with
             ``"retrace_from": "<upward trace>"`` restarts at its highest point whose gap is still
             >= ``retrace_min_gap`` (0.15) and integrates back DOWN (the stable direction) to its
             own t_stop, reporting how well it closes on the upward trace's start;
             including its Fig. 6 recentering (sweeps run only the moved phase's replicas; give
             x_a0 / x_g0 so a mis-centred start is recentered too). Each tracer runs in its own
             thread and submits its walkers to the same pool, so several traces and any other
             work share the GPU.

All walkers use ``nvalchemi.mc.VCSGC``: an SGC walker is VC-SGC with kappa = 0 and phi = -dmu
(identical move for move, test/mc/test_vcsgc.py), so ladder, fan-out, trace and VC-SGC walkers can
share one batch. Walkers can share a batch whenever they use the same MC/MD block structure (a
*block class*, e.g. hybrid SGC-NPT vs fixed-lattice SGC); temperature, delta_mu, kappa and c0 are
per walker.

Scheduling. Every chunk (25 blocks) the driver fills a batch of up to the class width. If a trace
walker is ready its class runs (trace chains are serial, so they get priority); otherwise the
class that fills the largest fraction of its width runs. Within a class: trace walkers, then
walkers already part-way through a step, then the rest in spec order. Slots are refilled as soon
as walkers finish, so a batch only narrows when nothing else of its class is ready. Every step is gated: a walker stays
at its (T, dmu) until its Pt fraction AND energy pass run_campaign's equilibration gate on
``gate_passes`` consecutive checks (min..max blocks), then reports and moves on.

Width. Per class: the largest width whose estimated peak ALLOCATED memory fits 85 % of the card
(peak reserved is much larger without expandable segments: efficiency_hybrid measured
6.3/11.5/21.8 GiB allocated but 9.0/17.8/48.7 GiB reserved at width 1/2/4), capped by the class's
``max_width``. Run with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True so reserved tracks
allocated. A CUDA out-of-memory error lowers that class's width by one and repeats the chunk from
the last committed states (nothing is lost).

Resumable: every walker's spec, progress and latest state are saved after each chunk; traces
resume from their own trace.json. A resubmission continues exactly where the last job stopped.

Trace runs also check structure: every walker's final state must keep a solid-like fraction
(atoms with exactly 12 neighbours inside the fcc first shell) of at least ``min_solid_fraction``
(0.5); otherwise that phase is reported with ``phase_ok=False`` and the tracer stops there with
"a new phase appeared" (e.g. melting of the Au-rich phase). Every accepted trace point's
structures are written to ``traces/<name>/structures/`` (.pt checkpoint + .extxyz per walker,
indexed in ``index.csv``) as soon as the point is accepted.

Outputs under ``--out`` (all in the formats the existing analysis tools read):
  ladders/<name>/*.equilibration.json     -> sgc_phase_boundary.py
  points/<name>/*.equilibration.json      -> sgc_phase_boundary.py
  vcsgc/<name>/*.series.json              -> vcsgc_analysis.py
  traces/<name>/trace.json                -> boundary_tracer.py report
  traces/<name>/structures/               accepted points: step<NN>_T<T>_<phase>_r<k>.{pt,extxyz}
  traces/<name>/closure.json              retrace only: arrival vs the upward trace's start
  throughput.csv (one row per chunk), walkers/<id>.json, states/

    python run_phase_campaign.py --spec campaign.json --out <dir> [--smoke]

Spec strings may use environment variables ($RUN_ROOT/...). Keys starting with "_" are ignored, so
"_comment" fields and parked items (e.g. "_traces") can live in the spec.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import threading
import time
import traceback
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from run_campaign import (
    BAROSTAT_TIME_FS,
    CHECKPOINT,
    DT_FS,
    EQUILIBRATION_WINDOW_BLOCKS,
    INFERENCE_SETTINGS,
    PRESSURE_EV_PER_A3,
    SEED,
    SPECIES,
    TASK,
    THERMOSTAT_TIME_FS,
    NvalchemiTraceEngine,
    _equilibration_gate,
    _npt_wrap_hooks,
    _wrap_batch_positions,
    compute_reference_energies,
)
from run_vcsgc_scan import initial_state, load_state, run_chunk

from nvalchemi.data import Batch
from nvalchemi.dynamics.integrators.npt import NPT
from nvalchemi.hybrid import HybridMCMD
from nvalchemi.mc import VCSGC
from nvalchemi.scheduling import FinalStateStore

# Peak allocated GiB at 500 atoms (efficiency matrix, A100, eager_unmerged + energy-only MC):
# hybrid SGC-NPT 6.3 / 11.5 / 21.8 at width 1 / 2 / 4; SGC only 1.9 / 2.8 / 4.5 / 7.9 / 11.3 at
# width 1 / 2 / 4 / 8 / 12. Linear fits, scaled with atom count.
MEM_BASE_GIB = 1.1
MEM_PER_WALKER_GIB_500 = {"md": 5.2, "mc_only": 0.85}
MEMORY_FRACTION = 0.85
PRIORITY = {"trace": 0, "ladder": 2, "point": 3, "vcsgc": 3}


@dataclass
class Walker:
    """One chain of (T, dmu) or (T, c0) steps, persisted under walkers/<wid>.json."""

    wid: str
    kind: str
    cls: str
    T: float
    mus: list[float]
    kappa: float = 0.0
    c0: float | None = None
    ref: float = 0.0
    start: dict = field(default_factory=dict)
    min_blocks: int = 100
    max_blocks: int = 300
    gate_passes: int = 2
    out_dir: str = ""
    branch: str = "Arich"
    order: int = 0
    # progress
    step: int = 0
    x: list = field(default_factory=list)
    e: list = field(default_factory=list)
    v: list = field(default_factory=list)
    gate: list = field(default_factory=list)
    finished: list = field(default_factory=list)
    done: bool = False
    stopped: str | None = None

    @property
    def n_steps(self) -> int:
        return 1 if self.kind == "vcsgc" else len(self.mus)

    def coefficients(self) -> tuple[float, float]:
        """(phi, kappa) for VCSGC with reference_exchange_potential = 0."""
        if self.kind == "vcsgc":
            return -2.0 * self.kappa * self.c0 - self.ref, self.kappa
        return -self.mus[self.step], 0.0

    def to_json(self) -> dict:
        """All fields, for walkers/<wid>.json."""
        return {k: v for k, v in self.__dict__.items()}


class Pool:
    """Walker registry + scheduler state shared by the main loop and the tracer threads."""

    def __init__(self, out: Path, store: FinalStateStore):
        self.out, self.store = out, store
        self.walkers: dict[str, Walker] = {}
        self.cond = threading.Condition()
        (out / "walkers").mkdir(parents=True, exist_ok=True)
        self._order = 0

    def path(self, wid: str) -> Path:
        """Progress file of walker *wid*."""
        return self.out / "walkers" / f"{wid}.json"

    def add(self, w: Walker) -> Walker:
        """Register a walker, or return the persisted one with the same id (resume)."""
        with self.cond:
            if w.wid in self.walkers:
                return self.walkers[w.wid]
            p = self.path(w.wid)
            if p.is_file():
                saved = Walker(**json.loads(p.read_text()))
                if (saved.kind, saved.cls, saved.T, saved.mus, saved.kappa, saved.c0) != (
                    w.kind, w.cls, w.T, w.mus, w.kappa, w.c0
                ):
                    raise SystemExit(f"walker {w.wid}: spec differs from {p}; use a fresh --out")
                w = saved
            self._order += 1
            w.order = self._order
            self.walkers[w.wid] = w
            self.save(w)
            self.cond.notify_all()
            return w

    def save(self, w: Walker) -> None:
        """Atomically persist a walker's spec and progress."""
        tmp = self.path(w.wid).with_suffix(".tmp")
        tmp.write_text(json.dumps(w.to_json()) + "\n")
        tmp.replace(self.path(w.wid))

    def ready(self) -> list[Walker]:
        """Unfinished walkers, most urgent first (trace, part-way through a step, spec order)."""
        with self.cond:
            ws = [w for w in self.walkers.values() if not w.done]
        return sorted(ws, key=lambda w: (PRIORITY[w.kind], 0 if w.x else 1, w.order))

    def wait(self, wids: list[str]) -> list[Walker]:
        """Block (tracer thread) until all walkers *wids* are done; return them."""
        with self.cond:
            self.cond.wait_for(lambda: all(self.walkers[i].done for i in wids))
            return [self.walkers[i] for i in wids]


def solid_fraction(data) -> float:
    """Fraction of atoms with exactly 12 neighbours inside the fcc first shell (minimum image).

    The cutoff sits midway between the first and second fcc shells of the cell's own mean
    lattice constant. Solid fcc at 700 K gives about 0.85-1.0 (scan_700_hybrid_tf32eo: 0.85-0.96 for
    the Au-rich phase); a melt falls far below 0.5."""
    pos = data.positions.detach().cpu().double().numpy().reshape(-1, 3)
    cell = data.cell.detach().cpu().double().numpy().reshape(3, 3)
    n = len(pos)
    a = (4 * abs(np.linalg.det(cell)) / n) ** (1 / 3)
    rc = 0.5 * (1 / np.sqrt(2) + 1) * a
    frac = pos @ np.linalg.inv(cell)
    d = frac[:, None, :] - frac[None, :, :]
    d -= np.round(d)
    dist = np.linalg.norm(d @ cell, axis=-1)
    np.fill_diagonal(dist, np.inf)
    return float(np.mean((dist < rc).sum(axis=1) == 12))


def write_extxyz(data, path: Path, comment: str = "") -> None:
    """Minimal extended-XYZ writer (species, positions, cell) for OVITO / ASE."""
    from ase.data import chemical_symbols

    z = data.atomic_numbers.detach().cpu().numpy().reshape(-1)
    pos = data.positions.detach().cpu().double().numpy().reshape(-1, 3)
    cell = data.cell.detach().cpu().double().numpy().reshape(3, 3)
    lat = " ".join(f"{v:.8f}" for v in cell.reshape(-1))
    lines = [str(len(z)), f'Lattice="{lat}" Properties=species:S:1:pos:R:3 pbc="T T T" {comment}'.rstrip()]
    lines += [f"{chemical_symbols[int(zi)]} {x:.6f} {y:.6f} {w:.6f}" for zi, (x, y, w) in zip(z, pos)]
    path.write_text("\n".join(lines) + "\n")


def make_exporting_tracer(base, store: FinalStateStore, tdir: Path):
    """BoundaryTracer subclass that writes each accepted point's structures when it is saved."""

    class ExportingTracer(base):
        """Exports traces/<name>/structures/ after every trace.json save (idempotent)."""

        def _save(self):
            super()._save()
            sdir = tdir / "structures"
            sdir.mkdir(exist_ok=True)
            index = sdir / "index.csv"
            new = not index.is_file()
            with index.open("a") as fh:
                if new:
                    fh.write("step,T_K,mu_eV,mu_se_eV,phase,replica,x_run_mean,solid_fraction,recentered,file\n")
                for i, pt in enumerate(self.trace["points"]):
                    for key, phase in (("a", "alpha"), ("g", "gamma")):
                        states = pt.get(f"state_{key}") or []
                        states = [states] if isinstance(states, str) else states
                        for k, sid in enumerate(states):
                            stem = f"step{i:02d}_T{pt['T']:g}_{phase}_r{k}"
                            dst = sdir / f"{stem}.pt"
                            if dst.is_file() or not store.exists(sid):
                                continue
                            shutil.copyfile(store.path_for(sid), dst)
                            data = store.load(sid, device="cpu")
                            sf = solid_fraction(data)
                            write_extxyz(data, sdir / f"{stem}.extxyz",
                                         f'T={pt["T"]:g} mu={pt["mu"]:.6f} phase={phase} replica={k}')
                            fh.write(f"{i},{pt['T']:g},{pt['mu']:.6f},{pt['mu_se']:.6f},{phase},{k},"
                                     f"{pt[key]['x']:.5f},{sf:.3f},{bool(pt.get('recentered'))},{stem}.pt\n")

    return ExportingTracer


class PoolTraceEngine(NvalchemiTraceEngine):
    """boundary_tracer engine whose walkers run in the shared pool (called from a tracer thread).

    Reuses NvalchemiTraceEngine's replica bookkeeping (_states), observables (_observe: last-window
    means, batch-means SEs, drift, gate) and replica combination (_combine, incl. replica_split).
    Walker ids are deterministic in the tracer's tag, so an interrupted run resumes mid-step."""

    def __init__(self, pool, name, cls, replicas, min_blocks, max_blocks, gate_passes, log_path,
                 min_solid_fraction=0.5):
        self.min_solid_fraction = min_solid_fraction
        self.pool, self.name, self.cls = pool, name, cls
        self.replicas, self.z, self.min_jump = replicas, 2.576, 0.03
        self.min_blocks, self.max_blocks, self.gate_passes = min_blocks, max_blocks, gate_passes
        self.log_path = log_path

    def _submit(self, T, mu, tag, phases) -> list[Walker]:
        """Queue one walker per replica for each (phase name, states, branch) and wait for all."""
        wids = []
        for phase, states, branch in phases:
            for k, s in enumerate(self._states(states)):
                w = self.pool.add(Walker(
                    wid=f"{tag}.{phase}.r{k}", kind="trace", cls=self.cls, T=float(T), mus=[float(mu)],
                    start={"state_ref": s}, min_blocks=self.min_blocks, max_blocks=self.max_blocks,
                    gate_passes=self.gate_passes, branch=branch,
                ))
                wids.append(w.wid)
        return self.pool.wait(wids)

    def _crystal(self, ws: list[Walker], obs: dict) -> dict:
        """Add each replica's solid-like fraction; flag the phase if any replica lost its crystal."""
        sf = [solid_fraction(self.pool.store.load(f"{w.wid}.final", device="cpu")) for w in ws]
        obs = dict(obs, solid_fraction=[round(v, 3) for v in sf])
        if min(sf) < self.min_solid_fraction:
            obs.update(phase_ok=False, phase_note=f"solid-like fraction {min(sf):.2f} < {self.min_solid_fraction}")
        return obs

    def run_one(self, T, mu, state, phase, tag=""):
        """One phase's replicas only (used by the tracer's Fig. 6 recentering sweeps)."""
        tag = f"{self.name}.{tag}".replace("+", "")
        name, branch = ("alpha", "Arich") if phase == "a" else ("gamma", "Brich")
        ws = self._submit(T, mu, tag, ((name, state, branch),))
        obs = self._combine([self._observe(w.finished[-1]["x_series"], w.finished[-1]["e_series"]) for w in ws])
        obs = self._crystal(ws, obs)
        print(f"[trace-run] {tag}: T={T:g} mu={mu:.5f} {name} x={obs['x']:.4f}{obs['x_replicas']} "
              f"solid {obs['solid_fraction']} (recentering)",
              flush=True)
        return obs, [f"{w.wid}.final" for w in ws]

    def run(self, T, mu, state_a, state_g, tag=""):
        """Both phases' replicas at (T, mu), run in the shared pool; returns the tracer's observables."""
        tag = f"{self.name}.{tag}".replace("+", "")
        ws = self._submit(T, mu, tag, (("alpha", state_a, "Arich"), ("gamma", state_g, "Brich")))
        per_run = [self._observe(w.finished[-1]["x_series"], w.finished[-1]["e_series"]) for w in ws]
        obs, new_states = {}, {}
        for j, phase in enumerate(("alpha", "gamma")):
            sl = slice(j * self.replicas, (j + 1) * self.replicas)
            obs[phase] = self._crystal(ws[sl], self._combine(per_run[sl]))
            new_states[phase] = [f"{w.wid}.final" for w in ws[sl]]
        with self.log_path.open("a") as fh:
            fh.write(json.dumps(dict(tag=tag, T=T, mu=mu, alpha=obs["alpha"], gamma=obs["gamma"])) + "\n")
        a, g = obs["alpha"], obs["gamma"]
        print(f"[trace-run] {tag}: T={T:g} mu={mu:.5f} x_a={a['x']:.4f}{a['x_replicas']} "
              f"x_g={g['x']:.4f}{g['x_replicas']} E_a={a['E']:.4f} E_g={g['E']:.4f} "
              f"solid a {a['solid_fraction']} g {g['solid_fraction']}", flush=True)
        return a, g, new_states["alpha"], new_states["gamma"]


def references(spec: dict, model, temps, out: Path, device) -> dict:
    """dmu_ref and lattice constants per T: from the spec's files, else calibrated once and cached."""
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "auto_reference_energies.json"
    table = {}
    for f in spec.get("reference_energies", []) + ([str(cache)] if cache.is_file() else []):
        for t, v in json.loads(Path(f).read_text())["reference"].items():
            table.setdefault(t, v)
    missing = sorted({float(t) for t in temps if f"{t:g}" not in table})
    if missing:
        print(f"[campaign] calibrating pure Au/Pt NPT references at {missing} K", flush=True)
        cal = compute_reference_energies(
            model, missing, spec["n_atoms"], n_blocks=150, md_steps_per_block=50,
            equilibration_window_blocks=EQUILIBRATION_WINDOW_BLOCKS, velocity_seed=SEED, device=device,
        )
        old = json.loads(cache.read_text()) if cache.is_file() else cal
        old["reference"].update(cal["reference"])
        cache.write_text(json.dumps(old, indent=2) + "\n")
        table.update(cal["reference"])
    return table


def build_walkers(spec: dict, ref: dict, pool: Pool) -> list[dict]:
    """Register ladder / point / vcsgc walkers; return the trace specs (run in threads)."""
    d = spec.get("defaults", {})

    def gated(item, kind):
        g = dict(min_blocks=d.get("min_blocks", 100), max_blocks=d.get("max_blocks", 300),
                 gate_passes=d.get("gate_passes", 2))
        g.update({k: item[k] for k in ("min_blocks", "max_blocks", "gate_passes") if k in item})
        return g

    def mu_list(item, T):
        if "mu_excess" in item:
            return [ref[f"{T:g}"]["delta_mu_ref_eV"] + float(m) for m in item["mu_excess"]]
        return [float(m) for m in item["mu"]]

    for it in spec.get("ladders", []):
        T = float(it["T"])
        pool.add(Walker(wid=f"ladder.{it['name']}", kind="ladder", cls=it["class"], T=T, mus=mu_list(it, T),
                        start=it["start"], branch=it.get("branch", "Arich"),
                        out_dir=f"ladders/{it['name']}", ref=ref[f"{T:g}"]["delta_mu_ref_eV"], **gated(it, "ladder")))
    for it in spec.get("points", []):
        T = float(it["T"])
        for k, mu in enumerate(mu_list(it, T)):
            pool.add(Walker(wid=f"point.{it['name']}.p{k}", kind="point", cls=it["class"], T=T, mus=[mu],
                            start=it["start"], branch=it.get("branch", "Arich"), out_dir=f"points/{it['name']}",
                            ref=ref[f"{T:g}"]["delta_mu_ref_eV"], **gated(it, "point")))
    for it in spec.get("vcsgc", []):
        T = float(it["T"])
        for c0 in it["c0"]:
            pool.add(Walker(wid=f"vcsgc.{it['name']}.c{c0:.3f}", kind="vcsgc", cls=it["class"], T=T, mus=[],
                            kappa=float(it["kappa"]), c0=float(c0), start=dict(it["start"], c0=c0),
                            out_dir=f"vcsgc/{it['name']}", ref=ref[f"{T:g}"]["delta_mu_ref_eV"], **gated(it, "vcsgc")))
    return spec.get("traces", [])


def initial(w: Walker, pool: Pool, ref: dict, n_atoms: int, device):
    """Starting AtomicData for a walker that has not run yet."""
    s = w.start
    if "state_ref" in s:
        r = s["state_ref"]
        return load_state(r, device) if r.endswith(".pt") and Path(r).is_file() else pool.store.load(r, device=device)
    if "state" in s:
        return load_state(s["state"], device)
    T = f"{w.T:g}"
    a = s.get("lattice_a")
    lattice = ({SPECIES[0]: a, SPECIES[1]: a} if a else
               {SPECIES[0]: ref[T]["Au"]["lattice_constant_a_ang"], SPECIES[1]: ref[T]["Pt"]["lattice_constant_a_ang"]})
    x0 = s.get("c0", s.get("x0", 0.5))
    seed_x = s.get("seed_phase_x")
    return initial_state(n_atoms, w.T, x0, s.get("fresh", "random"), lattice,
                         SEED + zlib.crc32(w.wid.encode()) % 100000, device, tuple(seed_x) if seed_x else None)


def finish_step(w: Walker, pool: Pool, n_atoms: int, equilibrated: bool, gx: dict, ge: dict,
                acceptance: float, final) -> None:
    """Record a finished step in the format of the matching analysis tool, then advance."""
    out = pool.out / w.out_dir if w.out_dir else None
    stopped = "equilibrated" if equilibrated else "cap"
    if w.kind in ("ladder", "point"):
        mu = w.mus[w.step]
        step_index = w.step if w.kind == "ladder" else int(w.wid.rsplit(".p", 1)[1])
        rid = f"atoms{n_atoms}.T{w.T:g}.{w.branch}.{w.wid.split('.', 1)[1]}.dmu{step_index}.mu{mu:.5f}"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{rid}.equilibration.json").write_text(json.dumps({
            "run_id": rid, "temperature_K": w.T, "chemical_potentials_ev": {"Au": 0.0, "Pt": mu},
            "delta_mu_ref_eV": w.ref, "delta_mu_excess_eV": mu - w.ref, "n_blocks": len(w.x),
            "window_blocks": EQUILIBRATION_WINDOW_BLOCKS, "composition_gate": gx, "energy_gate": ge,
            "resolved": bool(equilibrated), "stopped": stopped, "gate_history": w.gate,
            "acceptance_last_chunk": acceptance, "walker": w.wid, "x_series": w.x, "e_series": w.e,
        }) + "\n")
        pool.store.save(rid.replace("+", ""), final)
    w.finished.append(dict(step=w.step, blocks=len(w.x), stopped=stopped,
                           x=gx.get("mean_last_window"), e=ge.get("mean_last_window"),
                           x_series=list(w.x) if w.kind == "trace" else None,
                           e_series=list(w.e) if w.kind == "trace" else None))
    if w.kind == "trace":
        pool.store.save(f"{w.wid}.final", final)
    if w.kind == "vcsgc":
        w.stopped = stopped
    w.step += 1
    if w.step >= w.n_steps:
        w.done = True
    else:
        w.x, w.e, w.v, w.gate = [], [], [], []


def write_vcsgc_series(w: Walker, pool: Pool, n_atoms: int, cls: dict, settings: str, acc: list) -> None:
    """Write a VC-SGC walker's series in run_vcsgc_scan.py's format (read by vcsgc_analysis.py)."""
    out = pool.out / w.out_dir
    out.mkdir(parents=True, exist_ok=True)
    rid = f"atoms{n_atoms}.T{w.T:g}.vcsgc.k{w.kappa:g}.c{w.c0:.3f}.{w.start.get('fresh', 'ckpt')}"
    s = dict(run_id=rid, temperature_K=w.T, c0=w.c0, kappa=w.kappa, init=w.start.get("fresh", "checkpoint"),
             start_state=w.start.get("state"), seed_phase_x=w.start.get("seed_phase_x"), n_atoms=n_atoms,
             delta_mu_ref_eV=w.ref, mc_step_fraction=cls["mc_step_fraction"],
             md_steps_per_block=cls["md_steps_per_block"], inference_settings=settings, checkpoint=CHECKPOINT,
             task=TASK, c=w.x, u=w.e, v=w.v, acceptance=acc, gate=w.gate, stopped=w.stopped)
    (out / f"{rid}.series.json").write_text(json.dumps(s) + "\n")


def class_width(cls: dict, n_atoms: int, device) -> int:
    """Largest batch width whose estimated peak allocated memory fits the card, capped by max_width."""
    cap = int(cls.get("max_width", 4))
    if device.type != "cuda":
        return cap
    per = MEM_PER_WALKER_GIB_500["md" if cls["md_steps_per_block"] > 0 else "mc_only"] * n_atoms / 500
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    return max(1, min(cap, int((MEMORY_FRACTION * total - MEM_BASE_GIB) // per)))


def run(spec: dict, out: Path, model, device, settings: str) -> int:
    """Run (or resume) a whole campaign on one device; returns 0 on success, 1 if a trace failed."""
    out.mkdir(parents=True, exist_ok=True)
    n_atoms, chunk = int(spec["n_atoms"]), int(spec.get("chunk_blocks", EQUILIBRATION_WINDOW_BLOCKS))
    classes = spec["classes"]
    temps = {float(i["T"]) for k in ("ladders", "points", "vcsgc") for i in spec.get(k, [])}
    temps |= {float(t["t0"]) for t in spec.get("traces", []) if "t0" in t}   # a retrace has no t0 of its own
    ref = references(spec, model, temps, out, device)
    store = FinalStateStore(out / "states")
    pool = Pool(out, store)
    traces = build_walkers(spec, ref, pool)
    widths = {name: class_width(c, n_atoms, device) for name, c in classes.items()}
    print(f"[campaign] {spec.get('name', '')}: {len(pool.walkers)} walkers + {len(traces)} traces, "
          f"widths {widths}, chunk {chunk} blocks, settings {settings!r}", flush=True)

    # tracer threads (import here: boundary_tracer is vendored next to this script)
    from boundary_tracer import BoundaryTracer, TraceConfig

    errors: list[str] = []
    threads: dict[str, threading.Thread] = {}
    cfg_keys = ("mu0_se", "dt", "dt_min", "dt_max", "tol_mu", "min_gap", "max_corr", "max_gap_frac", "z", "min_jump",
                "max_steps", "recenter", "recenter_dmu", "recenter_max_runs", "max_recenter", "x_a0", "x_g0")
    names = {t["name"] for t in traces}
    for t in traces:
        src = t.get("retrace_from")
        if src and src not in names:
            raise SystemExit(f"trace {t['name']}: retrace_from {src!r} is not a trace in this spec")

    def retrace_start(t):
        """Highest point of the source trace whose gap is still >= retrace_min_gap (None if only the start)."""
        src = json.loads((out / "traces" / t["retrace_from"] / "trace.json").read_text())
        pts = src["points"]
        ok = [p for p in pts if p["g"]["x"] - p["a"]["x"] >= float(t.get("retrace_min_gap", 0.15))]
        top = max(ok, key=lambda p: p["T"]) if ok else None
        if top is None or top["T"] == pts[0]["T"]:
            return None, src
        return top, src

    for t in traces:
        tdir = out / "traces" / t["name"]
        tdir.mkdir(parents=True, exist_ok=True)
        engine = PoolTraceEngine(pool, f"trace.{t['name']}", t["class"], int(t.get("replicas", 1)),
                                 int(t.get("min_blocks", spec.get("defaults", {}).get("min_blocks", 100))),
                                 int(t.get("max_blocks", spec.get("defaults", {}).get("max_blocks", 300))),
                                 int(t.get("gate_passes", 2)), tdir / "runs.jsonl",
                                 float(t.get("min_solid_fraction", 0.5)))
        Tracer = make_exporting_tracer(BoundaryTracer, store, tdir)

        def target(engine=engine, t=t, tdir=tdir, Tracer=Tracer):
            try:
                if t.get("retrace_from"):
                    threads[t["retrace_from"]].join()
                    top, src = retrace_start(t)
                    if top is None:
                        print(f"[trace] {t['name']}: {t['retrace_from']} never got above its start with an open gap; "
                              "nothing to retrace", flush=True)
                        return
                    t0 = src["points"][0]
                    kw = {k: t[k] for k in cfg_keys if k in t}
                    kw.update(mu0_se=top["mu_se"], x_a0=top["a"]["x"], x_g0=top["g"]["x"])
                    cfg = TraceConfig(t0=top["T"], mu0=top["mu"], t_stop=float(t.get("t_stop", t0["T"])), **kw)
                    print(f"[trace] {t['name']}: retracing DOWN from {t['retrace_from']}'s top point T={top['T']:g} K "
                          f"mu={top['mu']:.5f}+-{top['mu_se']:.4f} (gap {top['g']['x'] - top['a']['x']:.3f}) "
                          f"to {cfg.t_stop:g} K", flush=True)
                    result = Tracer(cfg, engine, tdir / "trace.json").run(top["state_a"], top["state_g"])
                    end = min(result["points"], key=lambda p: abs(p["T"] - t0["T"]))
                    closure = dict(start_T=t0["T"], start_mu=t0["mu"], start_mu_se=t0["mu_se"], retrace_T=end["T"],
                                   retrace_mu=end["mu"], retrace_mu_se=end["mu_se"], dmu_meV=1e3 * (end["mu"] - t0["mu"]),
                                   x_alpha=[t0["a"]["x"], end["a"]["x"]], x_gamma=[t0["g"]["x"], end["g"]["x"]],
                                   stop_reason=result["stop_reason"])
                    (tdir / "closure.json").write_text(json.dumps(closure, indent=2) + "\n")
                    print(f"[trace] {t['name']}: closure at {end['T']:g} K: retrace mu {end['mu']:.5f}+-{end['mu_se']:.4f} "
                          f"vs start {t0['mu']:.5f}+-{t0['mu_se']:.4f} ({closure['dmu_meV']:+.2f} meV)", flush=True)
                else:
                    cfg = TraceConfig(**{k: t[k] for k in ("t0", "mu0", "t_stop")}, **{k: t[k] for k in cfg_keys if k in t})
                    result = Tracer(cfg, engine, tdir / "trace.json").run(t["alpha_state"], t["gamma_state"])
                print(f"[trace] {t['name']}: {result['status']}: {result['stop_reason']} "
                      f"({len(result['points'])} points)", flush=True)
            except Exception:  # surfaced at the end; the pool keeps serving the other work
                errors.append(f"trace {t['name']}:\n{traceback.format_exc()}")
            finally:
                with pool.cond:
                    pool.cond.notify_all()

        threads[t["name"]] = threading.Thread(target=target, name=t["name"], daemon=True)
    for th in threads.values():
        th.start()

    log = out / "throughput.csv"
    new_log = not log.is_file()
    n_chunks = sum(len(w.gate) for w in pool.walkers.values())
    acc_hist: dict[str, list] = {}
    with log.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if new_log:
            writer.writerow(["utc", "class", "width", "blocks", "seconds", "walker_blocks_per_s",
                             "peak_alloc_gib", "peak_reserved_gib", "acceptance", "walkers"])
        while True:
            ready = pool.ready()
            if not ready:
                if any(th.is_alive() for th in threads.values()):
                    with pool.cond:
                        pool.cond.wait(timeout=2.0)
                    continue
                break
            # Trace walkers first (their chains are serial); otherwise the class that fills the most
            # slots, so a lone walker of one class never runs at width 1 while another class waits.
            if ready[0].kind == "trace":
                cname = ready[0].cls
            else:
                fill = {c: min(sum(w.cls == c for w in ready), widths[c]) / widths[c] for c in {w.cls for w in ready}}
                cname = max(fill, key=lambda c: (fill[c], -min(w.order for w in ready if w.cls == c)))
            active = [w for w in ready if w.cls == cname][: widths[cname]]
            cls = classes[cname]
            states = []
            for w in active:
                started = w.x or w.step > 0 or w.finished
                states.append(store.load(w.wid, device=device) if started and store.exists(w.wid)
                              else initial(w, pool, ref, n_atoms, device))
            n_chunks += 1
            try:
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                batch = Batch.from_data_list(states, exclude_keys=["mc_accepted"])
                _wrap_batch_positions(batch)
                phis, kappas = zip(*(w.coefficients() for w in active))
                temps_t = torch.tensor([w.T for w in active], device=device)
                mc = VCSGC(model=model, temperature=temps_t, species=SPECIES, kappa=torch.tensor(kappas),
                           phi=torch.tensor(phis), concentration_species=SPECIES[1],
                           reference_exchange_potential=0.0, random_seed=SEED + 7919 * n_chunks)
                md = NPT(model=model, dt=DT_FS, temperature=temps_t,
                         pressure=torch.full((len(active),), PRESSURE_EV_PER_A3, device=device),
                         thermostat_time=THERMOSTAT_TIME_FS, barostat_time=BAROSTAT_TIME_FS,
                         pressure_coupling="isotropic", hooks=_npt_wrap_hooks())
                hybrid = HybridMCMD(mc=mc, md=md, mc_steps=max(1, round(cls["mc_step_fraction"] * n_atoms)),
                                    md_steps=int(cls["md_steps_per_block"]), mc_energy_only=True)
                start = time.perf_counter()
                series = run_chunk(hybrid, batch, chunk, n_atoms)
                elapsed = time.perf_counter() - start
            except torch.cuda.OutOfMemoryError:
                widths[cname] = max(1, len(active) - 1)
                print(f"[campaign] CUDA OOM at width {len(active)} ({cname}); width -> {widths[cname]}, "
                      "repeating the chunk from the last saved states", flush=True)
                n_chunks -= 1
                batch = hybrid = None
                torch.cuda.empty_cache()
                continue
            acceptance = mc.stats.acceptance
            finals = batch.to_data_list()
            alloc = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
            reserved = torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0
            notes = []
            with pool.cond:
                for i, (w, final) in enumerate(zip(active, finals)):
                    store.save(w.wid, final)
                    w.x.extend(series["c"][i])
                    w.e.extend(series["u"][i])
                    w.v.extend(series["v"][i])
                    acc_hist.setdefault(w.wid, []).append(dict(blocks=len(w.x), acceptance=acceptance))
                    n = len(w.x)
                    gx = ge = {}
                    if n >= max(w.min_blocks, 2 * EQUILIBRATION_WINDOW_BLOCKS):
                        gx = _equilibration_gate(w.x, EQUILIBRATION_WINDOW_BLOCKS)
                        ge = _equilibration_gate(w.e, EQUILIBRATION_WINDOW_BLOCKS)
                        w.gate.append(dict(blocks=n, passed=bool(gx.get("resolved")) and bool(ge.get("resolved"))))
                    recent = w.gate[-w.gate_passes:]
                    equilibrated = len(recent) == w.gate_passes and all(g["passed"] for g in recent)
                    if w.kind == "vcsgc":
                        if equilibrated or n >= w.max_blocks:
                            w.stopped = "equilibrated" if equilibrated else "cap"
                        write_vcsgc_series(w, pool, n_atoms, cls, settings, acc_hist[w.wid])
                    if equilibrated or n >= w.max_blocks:
                        label = w.mus[w.step] - w.ref if w.kind != "vcsgc" else w.c0
                        notes.append(f"{w.wid} step {w.step} ({'c0' if w.kind == 'vcsgc' else 'dmu_ex'} "
                                     f"{label:+.4f}) after {n} blocks, {'equilibrated' if equilibrated else 'cap'}: "
                                     f"x={gx.get('mean_last_window', float('nan')):.4f}")
                        finish_step(w, pool, n_atoms, equilibrated, gx, ge, acceptance, final)
                    pool.save(w)
                pool.cond.notify_all()
            writer.writerow([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), cname, len(active), chunk,
                             f"{elapsed:.1f}", f"{len(active) * chunk / elapsed:.5f}", f"{alloc:.2f}",
                             f"{reserved:.2f}", f"{acceptance:.4f}", " ".join(w.wid for w in active)])
            fh.flush()
            print(f"[campaign] chunk {n_chunks} {cname} width {len(active)}: {elapsed:.0f} s "
                  f"({len(active) * chunk / elapsed:.4f} walker-blocks/s, peak alloc {alloc:.1f} GiB, "
                  f"reserved {reserved:.1f}), acceptance {acceptance:.4f}; "
                  f"{sum(not w.done for w in pool.walkers.values())} walkers open", flush=True)
            for note in notes:
                print(f"[campaign]   {note}", flush=True)
            del batch, hybrid, mc, md, finals
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
    for th in threads.values():
        th.join()
    for err in errors:
        print(f"[campaign] ERROR {err}", flush=True)
    print(f"[campaign] {'complete' if not errors else 'finished with errors'}", flush=True)
    return 1 if errors else 0


def expand_env(obj):
    """Expand $VARS in every string of the spec (paths like $RUN_ROOT/...); fail on undefined ones."""
    if isinstance(obj, dict):
        return {k: expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env(v) for v in obj]
    if isinstance(obj, str) and "$" in obj:
        out = os.path.expandvars(obj)
        if "$" in out:
            raise SystemExit(f"undefined environment variable in spec string {obj!r}")
        return out
    return obj


def smoke_spec(spec: dict) -> dict:
    """The same campaign cut to one 50-block step per item (one trace step), to test a spec quickly."""
    spec = json.loads(json.dumps(spec))
    spec["defaults"] = dict(spec.get("defaults", {}), min_blocks=50, max_blocks=50)
    for key in ("ladders", "points", "vcsgc", "traces"):
        for it in spec.get(key, []):
            for k in ("min_blocks", "max_blocks"):
                it.pop(k, None)
            for k in ("mu_excess", "mu", "c0"):
                if k in it:
                    it[k] = it[k][:1]
            if key == "traces" and "t0" in it:     # a retrace inherits its start from the source trace
                step = it.get("dt_min", 5.0)
                it["t_stop"] = it["t0"] + (step if it["t_stop"] > it["t0"] else -step)
                it["dt"] = step
    return spec


def main() -> None:
    """Command-line entry point (UMA model)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--inference-settings", default=INFERENCE_SETTINGS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true",
                    help="check a spec end to end: every item cut to one 50-block step (use a separate --out)")
    args = ap.parse_args()
    from nvalchemi.models.uma import UMAWrapper

    spec = expand_env(json.loads(args.spec.read_text()))
    if args.smoke:
        spec = smoke_spec(spec)

    device = torch.device(args.device)
    model = UMAWrapper.from_checkpoint(CHECKPOINT, task_name=TASK, device=str(device),
                                       inference_settings=args.inference_settings)
    raise SystemExit(run(spec, args.out, model, device, args.inference_settings))


if __name__ == "__main__":
    main()
