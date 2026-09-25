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
# Vendored from the sgc-phase-boundary skill (scripts/boundary_tracer.py), version 1.0, sha256[:16]=dcb52dee8b7a7ff0.
# Do not edit here: change the skill's copy, rerun its tests, then re-vendor with
#   python boundary_tracer.py vendor <this file> --header <license header>
#!/usr/bin/env python3
"""Trace a two-phase coexistence line in T (van de Walle & Asta 2002, section 3.3).

Given ONE coexistence point (T0, dmu0) and one walker equilibrated in each phase there,
integrate eq. (29)

    d dmu / d beta = (E_g - E_a) / (beta (x_g - x_a)) - dmu / beta      (beta = 1/kT)

to follow dmu_coex(T), measuring E and x of both phases at every step. Each step needs only
two SGC runs at the SAME (T, dmu) -- one walker per phase, warm-started from the previous
step -- and the boundary compositions are simply the two walkers' x. No free-energy anchor
is needed after the starting point.

This module is ENGINE-AGNOSTIC: it owns the algorithm and the resumable trace file; the
simulation is supplied by an object with

    engine.run(T, mu, state_a, state_g, tag) -> (obs_a, obs_g, new_state_a, new_state_g)

where obs_* are dicts with keys x, x_se, E, E_se, drift (late-minus-early window mean of x),
resolved (bool), and states are JSON-serializable handles (e.g. checkpoint ids). Both
walkers must be run at the same T and dmu. E is per atom and must be the energy the SGC
acceptance uses (potential energy for rigid-lattice SGC; potential energy + PV for NPT
hybrid runs), in the same units and zero as dmu. x is the fraction of species B.

Algorithm (see references/theory.md, "Boundary tracing"):
  * predictor: Adams-Bashforth-2 in beta (Euler on the first step); corrector: trapezoid,
    re-running at the corrected dmu until it moves by less than --tol-mu (max --max-corr);
  * each walker is checked against its own history: a jump toward the other phase, a drift
    toward it, or a collapsed gap rejects the step, restores both walkers, and halves dT;
  * dT grows by 1.5x after clean steps (up to --dt-max); tracing stops at --t-stop, when dT
    falls below --dt-min (walkers keep transforming: near T_c, a spinodal, or a new phase),
    or when x_g - x_a drops below --min-gap (critical point approached).
Integrating toward LOWER T is stable (eq. 31: the gap widens); toward T_c it is not, which is
why step rejection and the gap/critical checks matter most going up.

Command line (analysis only; tracing itself is driven by an engine adapter):
    python boundary_tracer.py report trace_down.json [trace_up.json ...] --out dir
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

KB_EV = 8.617333262e-5
TRACER_VERSION = "1.0"


# ----------------------------------------------------------------------------- configuration
@dataclass
class TraceConfig:
    t0: float                       # starting temperature (K), a known coexistence point
    mu0: float                      # coexistence dmu at t0 (eV)
    t_stop: float                   # trace toward this temperature (either direction)
    dt: float = 50.0                # initial |dT| (K)
    dt_min: float = 5.0             # give up below this |dT|
    dt_max: float = 100.0           # never step more than this
    tol_mu: float = 0.002           # corrector convergence (eV)
    max_corr: int = 2               # corrector re-runs per step
    min_gap: float = 0.05           # x_g - x_a below this = critical point approached
    z: float = 2.576                # significance for jump/drift tests
    min_jump: float = 0.03          # smallest x change counted as leaving the phase
    max_steps: int = 60
    mu0_se: float = 0.0             # uncertainty of the starting dmu (propagated as an offset)
    max_gap_frac: float = 0.15      # cap |dT| so the gap shrinks by at most this fraction per step


# ----------------------------------------------------------------------------- physics
def slope(T: float, mu: float, a: dict, g: dict) -> tuple[float, float]:
    """Eq. (29) right-hand side and its 1-sigma error from the observables' SEs."""
    beta = 1.0 / (KB_EV * T)
    dx = g["x"] - a["x"]
    dE = g["E"] - a["E"]
    f = dE / (beta * dx) - mu / beta
    sE = math.hypot(a.get("E_se") or 0.0, g.get("E_se") or 0.0)
    sx = math.hypot(a.get("x_se") or 0.0, g.get("x_se") or 0.0)
    f_se = math.hypot(sE / (beta * dx), dE * sx / (beta * dx * dx))
    return f, f_se


def _predict_own(hist: list[dict], key: str, T: float) -> float | None:
    """Linear extrapolation in T from a walker's last (up to 3) accepted points."""
    pts = [(p["T"], p[key]["x"]) for p in hist[-3:]]
    if len(pts) < 2:
        return None
    n = len(pts)
    mt = sum(t for t, _ in pts) / n
    mx = sum(x for _, x in pts) / n
    stt = sum((t - mt) ** 2 for t, _ in pts)
    b = sum((t - mt) * (x - mx) for t, x in pts) / stt if stt > 0 else 0.0
    return mx + b * (T - mt)


def check_phases(cfg: TraceConfig, hist: list[dict], T: float, a: dict, g: dict) -> list[str]:
    """Reasons to reject a step: a walker left (or is leaving) its phase, or the gap closed."""
    problems = []
    gap = g["x"] - a["x"]
    for key, obs, toward in (("a", a, +1.0), ("g", g, -1.0)):   # alpha moves up toward gamma; gamma down
        se = obs.get("x_se") or 0.0
        drift = obs.get("drift") or 0.0
        if toward * drift > cfg.z * max(se, 1e-6) and abs(drift) > 0.5 * cfg.min_jump:
            problems.append(f"{'alpha' if key == 'a' else 'gamma'} still drifting toward the other phase "
                            f"(window drift {drift:+.4f})")
        pred = _predict_own(hist, key, T)
        if pred is not None:
            dev = obs["x"] - pred
            if toward * dev > max(cfg.z * se, cfg.min_jump) and abs(dev) > 0.25 * max(gap, 1e-9):
                problems.append(f"{'alpha' if key == 'a' else 'gamma'} jumped toward the other phase "
                                f"(x {obs['x']:.3f} vs own trend {pred:.3f})")
    if gap < cfg.min_gap:
        problems.append(f"gap x_g - x_a = {gap:.3f} < min_gap {cfg.min_gap}")
    return problems


# ----------------------------------------------------------------------------- tracer
class BoundaryTracer:
    """Resumable eq. (29) integrator. ``path`` holds the whole trace as JSON."""

    def __init__(self, cfg: TraceConfig, engine, path: str | Path, log=print):
        self.cfg, self.engine, self.path, self.log = cfg, engine, Path(path), log
        self.trace = dict(version=TRACER_VERSION, config=asdict(cfg), points=[], events=[], status="running",
                          stop_reason=None, next_dt=None)

    # -- persistence ---------------------------------------------------------------
    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.trace, indent=2, default=float))
        os.replace(tmp, self.path)

    def _event(self, **kw):
        self.trace["events"].append(kw)
        self.log(f"[trace] {kw}")

    # -- main loop -----------------------------------------------------------------
    def run(self, state_a, state_g):
        cfg = self.cfg
        if self.path.exists():                       # resume
            self.trace = json.loads(self.path.read_text())
            if self.trace["status"] != "running":
                self.log(f"[trace] {self.path} already finished: {self.trace['stop_reason']}")
                return self.trace
            self.log(f"[trace] resuming {self.path} at T={self.trace['points'][-1]['T'] if self.trace['points'] else cfg.t0}")
        pts = self.trace["points"]
        direction = 1.0 if cfg.t_stop > cfg.t0 else -1.0

        if not pts:                                  # anchor point: verify both walkers at (T0, mu0)
            a, g, sa, sg = self.engine.run(cfg.t0, cfg.mu0, state_a, state_g, tag=f"T{cfg.t0:g}.start")
            problems = [p for p in check_phases(cfg, [], cfg.t0, a, g) if "gap" in p or "drifting" in p]
            if problems:
                self.trace.update(status="failed", stop_reason="start point invalid: " + "; ".join(problems))
                self._save()
                return self.trace
            f, fse = slope(cfg.t0, cfg.mu0, a, g)
            pts.append(dict(T=cfg.t0, mu=cfg.mu0, mu_se=cfg.mu0_se, a=a, g=g, f=f, f_se=fse, dT=0.0,
                            corrector_runs=0, mu_pred=cfg.mu0, mu_corr=cfg.mu0, state_a=sa, state_g=sg))
            self.trace["next_dt"] = direction * abs(cfg.dt)
            self._save()
            self.log(f"[trace] start T={cfg.t0:g} mu={cfg.mu0:.4f} x_a={a['x']:.3f} x_g={g['x']:.3f} f={f:+.4f}")

        while self.trace["status"] == "running":
            p0 = pts[-1]
            dT = self.trace["next_dt"]
            if len(pts) - 1 >= cfg.max_steps:
                self._finish("max_steps reached"); break
            if (p0["T"] - cfg.t_stop) * direction >= -1e-9:
                self._finish("reached t_stop"); break
            T1 = p0["T"] + dT
            if (T1 - cfg.t_stop) * direction > 0:
                T1, dT = cfg.t_stop, cfg.t_stop - p0["T"]
            b0, b1 = 1 / (KB_EV * p0["T"]), 1 / (KB_EV * T1)
            db = b1 - b0
            if len(pts) >= 2:                        # Adams-Bashforth 2 in beta (variable step)
                pm = pts[-2]
                bm = 1 / (KB_EV * pm["T"])
                mu_pred = p0["mu"] + db * (p0["f"] + (p0["f"] - pm["f"]) * db / (2 * (b0 - bm)))
            else:
                mu_pred = p0["mu"] + db * p0["f"]

            mu_run, runs, accepted = mu_pred, 0, None
            sa, sg = p0["state_a"], p0["state_g"]
            while True:
                a, g, sa_new, sg_new = self.engine.run(T1, mu_run, sa, sg, tag=f"T{T1:g}.run{runs}")
                runs += 1
                problems = check_phases(cfg, pts, T1, a, g)
                if problems:
                    break
                f1, f1se = slope(T1, mu_run, a, g)
                mu_corr = p0["mu"] + 0.5 * (p0["f"] + f1) * db
                sa, sg = sa_new, sg_new             # continue from the new states if we re-run
                if abs(mu_corr - mu_run) <= cfg.tol_mu or runs > cfg.max_corr:
                    if abs(mu_corr - mu_run) > cfg.tol_mu:
                        self._event(T=T1, kind="corrector-not-converged", mu_run=mu_run, mu_corr=mu_corr)
                    accepted = (a, g, f1, f1se, mu_corr, sa_new, sg_new)
                    break
                mu_run = mu_corr

            if accepted is None:                     # a walker left its phase: reject, halve the step
                self._event(T=T1, kind="rejected", dT=dT, mu=mu_run, reasons=problems)
                nd = dT / 2.0
                if abs(nd) < cfg.dt_min:
                    crit = any("gap" in p for p in problems)
                    self._finish(("gap closed: critical point approached" if crit else
                                  "walkers keep transforming at the minimum step (near T_c, a spinodal, "
                                  "or a new phase)") + f" between {p0['T']:g} and {T1:g} K")
                    break
                self.trace["next_dt"] = nd
                self._save()
                continue

            a, g, f1, f1se, mu_corr, sa_new, sg_new = accepted
            # dmu uncertainty: start offset + trapezoid slope noise accumulated in quadrature
            step_se = 0.5 * abs(db) * math.hypot(p0["f_se"], f1se)
            pts.append(dict(T=T1, mu=mu_run, mu_se=math.hypot(p0["mu_se"], step_se), a=a, g=g, f=f1, f_se=f1se,
                            dT=dT, corrector_runs=runs, mu_pred=mu_pred, mu_corr=mu_corr,
                            state_a=sa_new, state_g=sg_new))
            smooth = abs(mu_corr - mu_run) < 0.25 * cfg.tol_mu and runs == 1
            nxt = min(abs(dT) * (1.5 if smooth else 1.0), cfg.dt_max)
            # Approaching T_c the gap closes like (T_c - T)^beta_c and x becomes hypersensitive to dmu:
            # cap the step so the gap shrinks by at most max_gap_frac (from the last two points' trend).
            q = pts[-2]
            gap1, gap0 = g["x"] - a["x"], q["g"]["x"] - q["a"]["x"]
            dgap_dT = (gap1 - gap0) / (T1 - q["T"])
            if direction * dgap_dT < 0:
                nxt = min(nxt, max(cfg.max_gap_frac * gap1 / abs(dgap_dT), cfg.dt_min))
            self.trace["next_dt"] = direction * nxt
            self.log(f"[trace] T={T1:g} mu={mu_run:.4f}±{pts[-1]['mu_se']:.4f} x_a={a['x']:.3f} x_g={g['x']:.3f} "
                     f"runs={runs} next dT={self.trace['next_dt']:+g}")
            self._save()
        return self.trace

    def _finish(self, reason):
        self.trace.update(status="finished", stop_reason=reason)
        self._save()
        self.log(f"[trace] finished: {reason}")


# ----------------------------------------------------------------------------- critical point estimate
def critical_estimate(points: list[dict], beta_c: float = 0.326) -> dict | None:
    """Fit gap ~ A (T_c - T)^beta_c on the points nearest the top of the trace.

    beta_c = 0.326 is the 3D Ising order-parameter exponent (use 0.5 for mean field). Only
    meaningful when the trace approached T_c (gap noticeably shrinking); treat as an estimate.
    """
    pts = sorted(points, key=lambda p: p["T"])
    use = [p for p in pts if p["g"]["x"] - p["a"]["x"] > 0][-4:]
    if len(use) < 3:
        return None
    ys = [(p["g"]["x"] - p["a"]["x"]) ** (1.0 / beta_c) for p in use]
    ts = [p["T"] for p in use]
    n = len(ts); mt = sum(ts) / n; my = sum(ys) / n
    stt = sum((t - mt) ** 2 for t in ts)
    b = sum((t - mt) * (y - my) for t, y in zip(ts, ys)) / stt if stt else 0.0
    if b >= 0:
        return None                              # gap not shrinking with T: no T_c in reach
    tc = mt - my / b
    return dict(T_c=tc, beta_c=beta_c, fitted_T=ts,
                x_c=0.5 * (use[-1]["a"]["x"] + use[-1]["g"]["x"]))


# ----------------------------------------------------------------------------- report
def report(paths: list[str], out: str, title: str | None = None, species=("A", "B"), beta_c: float = 0.326) -> dict:
    """Merge one or more traces (e.g. down and up from the same start) into a table + figure."""
    import csv
    traces = [json.loads(Path(p).read_text()) for p in paths]
    rows = {}
    for tr in traces:
        for p in tr["points"]:
            rows[round(p["T"], 6)] = p                 # the shared start point appears in each trace once
    pts = [rows[k] for k in sorted(rows)]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "traced_boundary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["T_K", "dmu_coex_eV", "dmu_se_eV", "x_alpha", "x_alpha_se", "x_gamma", "x_gamma_se",
                    "E_alpha_eV", "E_gamma_eV", "dmu_dbeta", "alpha_resolved", "gamma_resolved", "corrector_runs"])
        for p in pts:
            w.writerow([p["T"], p["mu"], p["mu_se"], p["a"]["x"], p["a"].get("x_se"), p["g"]["x"], p["g"].get("x_se"),
                        p["a"]["E"], p["g"]["E"], p["f"], p["a"].get("resolved"), p["g"].get("resolved"),
                        p["corrector_runs"]])
    crit = critical_estimate(pts, beta_c=beta_c)
    summary = dict(n_points=len(pts), T_range=[pts[0]["T"], pts[-1]["T"]] if pts else None,
                   stop_reasons=[tr.get("stop_reason") for tr in traces],
                   rejections=sum(1 for tr in traces for e in tr["events"] if e.get("kind") == "rejected"),
                   critical_estimate=crit)
    Path(out, "traced_boundary.json").write_text(json.dumps(summary, indent=2, default=float))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A, B = species
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    T = [p["T"] for p in pts]
    xa = [p["a"]["x"] for p in pts]; xg = [p["g"]["x"] for p in pts]
    ea = [2 * (p["a"].get("x_se") or 0) for p in pts]; eg = [2 * (p["g"].get("x_se") or 0) for p in pts]
    ax1.errorbar(xa, T, xerr=ea, color="#2a78d6", marker="o", ms=5, mec="white", lw=1.6, label=f"{A}-rich boundary")
    ax1.errorbar(xg, T, xerr=eg, color="#eb6834", marker="o", ms=5, mec="white", lw=1.6, label=f"{B}-rich boundary")
    for p in pts:
        ax1.plot([p["a"]["x"], p["g"]["x"]], [p["T"], p["T"]], color="0.8", lw=0.8, zorder=0)
    t0 = traces[0]["config"]["t0"]
    ax1.axhline(t0, color="0.5", ls=":", lw=1)
    ax1.text(0.01, t0, " start", va="bottom", fontsize=7, color="0.4")
    if crit and crit["T_c"] > max(T):
        ax1.plot([crit["x_c"]], [crit["T_c"]], marker="*", ms=12, color="0.3",
                 label=f"T_c estimate {crit['T_c']:.0f} K (fit, β_c = {beta_c:g})")
    ax1.set(xlim=(0, 1), xlabel=f"x_{B}", ylabel="T (K)", title=title or "Traced coexistence line (eq. 29)")
    ax1.legend(frameon=False, fontsize=8)
    ax2.errorbar(T, [p["mu"] for p in pts], yerr=[2 * p["mu_se"] for p in pts], color="#1baf7a", marker="o", ms=5,
                 mec="white", lw=1.6)
    ax2.set(xlabel="T (K)", ylabel=f"Δμ_coex = μ_{B} − μ_{A} (eV)", title="Coexistence chemical potential (bars 2σ)")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25, lw=0.5); ax.spines[["top", "right"]].set_visible(False); ax.title.set_fontsize(10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "traced_boundary.png"), dpi=160)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="merge trace JSON files into a table and a T-x figure")
    r.add_argument("traces", nargs="+")
    r.add_argument("--out", required=True)
    r.add_argument("--title", default=None)
    r.add_argument("--species", nargs=2, default=("A", "B"))
    r.add_argument("--beta-c", type=float, default=0.326,
                   help="order-parameter exponent for the T_c fit: 0.326 (3D Ising, real MC) or 0.5 (mean field)")
    v = sub.add_parser("vendor", help="copy this module into a simulation repo (optionally with its license header)")
    v.add_argument("dest", type=Path)
    v.add_argument("--header", type=Path, default=None, help="text file prepended verbatim (e.g. a repo's SPDX header)")
    args = ap.parse_args()
    if args.cmd == "report":
        s = report(args.traces, args.out, args.title, tuple(args.species), args.beta_c)
        print(json.dumps(s, indent=2, default=float))
    elif args.cmd == "vendor":
        vendor(args.dest, args.header)


def _canonical_source() -> str:
    """This module's source with any vendoring banner removed (so copies can be compared)."""
    text = Path(__file__).read_text()
    marker = "#!/usr/bin/env python3\n"
    return text[text.index(marker):] if marker in text else text


def vendor(dest: Path, header: Path | None = None) -> None:
    import hashlib
    src = _canonical_source()
    digest = hashlib.sha256(src.encode()).hexdigest()[:16]
    banner = (f"# Vendored from the sgc-phase-boundary skill (scripts/boundary_tracer.py), version {TRACER_VERSION}, "
              f"sha256[:16]={digest}.\n# Do not edit here: change the skill's copy, rerun its tests, then re-vendor with\n"
              f"#   python boundary_tracer.py vendor <this file> --header <license header>\n")
    text = (header.read_text().rstrip("\n") + "\n" if header else "") + banner + src
    Path(dest).write_text(text)
    print(f"wrote {dest} (version {TRACER_VERSION}, sha256[:16]={digest})")


if __name__ == "__main__":
    main()
