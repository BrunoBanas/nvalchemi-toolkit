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
"""End-to-end validation of boundary_tracer.py against an exact binodal.

The engine is the sub-regular mean-field solution of make_synthetic.py: each walker sits on
its own metastable branch and only jumps to the other branch past its spinodal, like a real
SGC walker, and reports noisy x and E. The exact coexistence line comes from the common
tangent on a dense grid. Checks:
  1. downward trace from 950 K reproduces x_alpha, x_gamma and dmu_coex at every step;
  2. upward trace stops near the exact T_c and the Ising/mean-field T_c estimate is close;
  3. a start dmu that is off by +4 meV does not grow while tracing downward (eq. 31);
  4-6. recentering (paper Fig. 6) with NucleationEngine, whose walkers switch phase as soon as
     dmu leaves a narrow window dmu_coex(T) - w_g .. dmu_coex(T) + w_a (nucleation-limited
     hysteresis, as in the 500-atom Au-Pt runs), not at the spinodals:
     4. a start dmu 10 meV off with w = 6 meV: the alpha walker transforms at once; recentering
        the start must land within 1.5 meV of dmu_coex and the trace must reach t_stop (and fail
        to start without recentering);
     5. asymmetric windows (3 / 9 meV): the recentered start is biased by (w_a - w_g)/2 = -3 meV,
        the documented limit of the midpoint rule;
     6. a trace that drifts (energies biased by 0.03 eV * x) out of a 4 meV window: recentering
        keeps every point within 4 meV and reaches t_stop, where the plain tracer stops early;
        also through an engine without run_one (the fallback path);
  7. an engine that reports phase_ok=False for the alpha walker above 1100 K (it "melts"): the
     upward trace must stop below 1100 K with "a new phase appeared", without recentering.

    python test_boundary_tracer.py [--out dir]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The tracer itself lives next to the nvalchemi engine adapter (run_campaign.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hybrid_sgc_npt"))
from boundary_tracer import (  # noqa: E402
    BoundaryTracer,
    TraceConfig,
    critical_estimate,
    report,
)
from make_synthetic import model, truth  # noqa: E402

L0, L1, EB = 0.20, 0.03, -0.5


def branch_roots(T):
    """Return the low- and high-x increasing branches of mu(x) at *T*."""
    x = np.linspace(1e-7, 1 - 1e-7, 200001)
    _, _, mu = model(L0, L1, EB, T, x)
    breaks = np.where(np.diff(mu) <= 0)[0]
    if len(breaks) == 0:
        return (x, mu), (x, mu)
    lo_end, hi_start = breaks[0] + 1, breaks[-1] + 1
    return (x[:lo_end], mu[:lo_end]), (x[hi_start:], mu[hi_start:])


class MeanFieldEngine:
    """Two walkers on metastable mean-field branches with measurement noise."""

    def __init__(self, noise_x=0.003, noise_E=0.0015, seed=0):
        self.rng = np.random.default_rng(seed)
        self.nx, self.nE = noise_x, noise_E
        self.calls = 0

    def _walker(self, T, mu, state):
        low, high = branch_roots(T)
        branch = state["branch"]
        if branch == "low" and mu > low[1][-1]:
            branch = "high"  # past the low branch's spinodal: jump
        if branch == "high" and mu < high[1][0]:
            branch = "low"
        xs, mus = low if branch == "low" else high
        x = float(np.interp(mu, mus, xs))
        E = float(model(L0, L1, EB, T, np.array([x]))[0][0])
        obs = dict(
            x=float(np.clip(x + self.rng.normal(0, self.nx), 1e-6, 1 - 1e-6)),
            x_se=self.nx,
            E=E + self.rng.normal(0, self.nE),
            E_se=self.nE,
            drift=0.0,
            resolved=True,
        )
        return obs, dict(branch=branch)

    def run(self, T, mu, sa, sg, tag=""):
        """Report both walkers' noisy observables at ``(T, mu)``, like an SGC engine."""
        self.calls += 1
        a, sa2 = self._walker(T, mu, sa)
        g, sg2 = self._walker(T, mu, sg)
        return a, g, sa2, sg2


_MU_C = {}


def mu_coex(T):
    """Exact dmu_coex at *T* (cached), or None above T_c."""
    key = round(float(T), 6)
    if key not in _MU_C:
        t = truth(L0, L1, EB, key)
        _MU_C[key] = t["mu_coex"] if t else None
    return _MU_C[key]


class NucleationEngine(MeanFieldEngine):
    """Walkers switch branch once dmu leaves [dmu_coex - w_g, dmu_coex + w_a] (or a spinodal)."""

    def __init__(self, w_a, w_g, ebias=0.0, single=True, **kw):
        super().__init__(**kw)
        self.w_a, self.w_g, self.ebias, self.single, self.one_calls = (
            w_a,
            w_g,
            ebias,
            single,
            0,
        )

    def __getattribute__(self, name):
        """Hide run_one when ``single`` is False, to exercise the tracer's fallback path."""
        if name == "run_one" and not object.__getattribute__(self, "single"):
            raise AttributeError(name)  # exercise the tracer's fallback path
        return object.__getattribute__(self, name)

    def _walker(self, T, mu, state):
        """One walker's observables: it switches branch once dmu leaves its window."""
        low, high = branch_roots(T)
        branch, mc = state["branch"], mu_coex(T)
        if branch == "low" and (mu > mc + self.w_a or mu > low[1][-1]):
            branch = "high"
        if branch == "high" and (mu < mc - self.w_g or mu < high[1][0]):
            branch = "low"
        xs, mus = low if branch == "low" else high
        x = float(np.interp(mu, mus, xs))
        E = float(model(L0, L1, EB, T, np.array([x]))[0][0]) + self.ebias * x
        obs = dict(
            x=float(np.clip(x + self.rng.normal(0, self.nx), 1e-6, 1 - 1e-6)),
            x_se=self.nx,
            E=E + self.rng.normal(0, self.nE),
            E_se=self.nE,
            drift=0.0,
            resolved=True,
        )
        return obs, dict(branch=branch)

    def run_one(self, T, mu, state, phase, tag=""):
        """Run a single walker (the optional engine method recentering uses)."""
        self.one_calls += 1
        return self._walker(T, mu, state)


def mu_errors(trace):
    """(T, dmu error, recentered?) for each traced point with an exact reference."""
    return [
        (p["T"], p["mu"] - mu_coex(p["T"]), bool(p.get("recentered")))
        for p in trace["points"]
        if mu_coex(p["T"]) is not None
    ]


def exact_tc():
    """Return the exact critical temperature by bisection on the existence of a gap."""
    lo, hi = 900.0, 1400.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if truth(L0, L1, EB, mid) else (lo, mid)
    return 0.5 * (lo + hi)


def compare(trace):
    """Return per-point errors of a trace against the exact coexistence line."""
    errs = []
    for p in trace["points"]:
        t = truth(L0, L1, EB, p["T"])
        if t:
            errs.append(
                (
                    p["T"],
                    p["mu"] - t["mu_coex"],
                    p["a"]["x"] - t["x_alpha"],
                    p["g"]["x"] - t["x_gamma"],
                )
            )
    return errs


def main():
    """Run the downward, upward, offset-start and recentering traces and check them against the truth."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tracer_selftest")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    for f in out.glob("*.json"):
        f.unlink()
    T0 = 950.0
    t0 = truth(L0, L1, EB, T0)
    tc = exact_tc()
    ok = True

    def quiet(*_):
        return None

    # 1. downward from the exact start point
    cfg = TraceConfig(t0=T0, mu0=t0["mu_coex"], t_stop=600.0, dt=50.0)
    tr = BoundaryTracer(cfg, MeanFieldEngine(seed=1), out / "down.json", log=quiet).run(
        dict(branch="low"), dict(branch="high")
    )
    e = compare(tr)
    worst = max(max(abs(d) for d in r[1:]) for r in e)
    print(f"[1] downward 950->600 K: {len(e)} points, stop='{tr['stop_reason']}'")
    for T, dmu, dxa, dxg in e:
        print(
            f"      T={T:6.1f}  dmu err {dmu * 1e3:+6.2f} meV   x_alpha err {dxa:+.4f}   x_gamma err {dxg:+.4f}"
        )
    ok &= worst < 0.01 and tr["stop_reason"] == "reached t_stop"
    print(f"      worst |error| = {worst:.4f}  -> {'PASS' if worst < 0.01 else 'FAIL'}")

    # 2. upward toward T_c
    cfg = TraceConfig(t0=T0, mu0=t0["mu_coex"], t_stop=1400.0, dt=50.0, dt_min=2.0)
    tr_up = BoundaryTracer(
        cfg, MeanFieldEngine(seed=2), out / "up.json", log=quiet
    ).run(dict(branch="low"), dict(branch="high"))
    top = max(p["T"] for p in tr_up["points"])
    crit = critical_estimate(
        tr_up["points"], beta_c=0.5
    )  # mean-field model -> exponent 1/2
    e_up = compare(tr_up)
    worst_up = max(
        max(abs(d) for d in r[1:3]) for r in e_up
    )  # mu and x_alpha (x_gamma steepens near T_c)
    print(
        f"[2] upward from 950 K: highest traced T {top:.1f} K, exact T_c {tc:.1f} K, stop='{tr_up['stop_reason']}'"
    )
    print(
        f"      T_c estimate (beta_c=0.5): {crit['T_c']:.1f} K"
        if crit
        else "      no T_c estimate"
    )
    good_up = (
        top > tc - 120 and top < tc and crit is not None and abs(crit["T_c"] - tc) < 40
    )
    ok &= good_up
    print(
        f"      worst |dmu|,|x_alpha| error on traced points = {worst_up:.4f}; -> {'PASS' if good_up else 'FAIL'}"
    )

    # 3. biased start: +4 meV
    cfg = TraceConfig(
        t0=T0, mu0=t0["mu_coex"] + 0.004, t_stop=600.0, dt=50.0, mu0_se=0.004
    )
    tr_b = BoundaryTracer(
        cfg, MeanFieldEngine(seed=3), out / "biased.json", log=quiet
    ).run(dict(branch="low"), dict(branch="high"))
    e_b = compare(tr_b)
    first, last = e_b[0][1], e_b[-1][1]
    stable = abs(last) <= abs(first) + 0.002
    ok &= stable
    print(
        f"[3] start +4 meV: dmu error {first * 1e3:+.2f} meV at {e_b[0][0]:.0f} K -> {last * 1e3:+.2f} meV at "
        f"{e_b[-1][0]:.0f} K  -> {'PASS (does not grow)' if stable else 'FAIL'}"
    )

    # 4. mis-centred start, narrow symmetric window
    base = dict(
        t0=T0,
        t_stop=600.0,
        dt=50.0,
        mu0_se=0.004,
        x_a0=t0["x_alpha"],
        x_g0=t0["x_gamma"],
    )
    cfg = TraceConfig(mu0=t0["mu_coex"] + 0.010, **base)
    tr4 = BoundaryTracer(
        cfg,
        NucleationEngine(0.006, 0.006, seed=4),
        out / "recenter_start.json",
        log=quiet,
    ).run(dict(branch="low"), dict(branch="high"))
    tr4n = BoundaryTracer(
        TraceConfig(mu0=t0["mu_coex"] + 0.010, recenter=False, **base),
        NucleationEngine(0.006, 0.006, seed=4),
        out / "recenter_start_off.json",
        log=quiet,
    ).run(dict(branch="low"), dict(branch="high"))
    e4 = mu_errors(tr4)
    good4 = (
        tr4["stop_reason"] == "reached t_stop"
        and bool(e4)
        and e4[0][2]
        and abs(e4[0][1]) < 0.0015
        and max(abs(d) for _, d, _ in e4) < 0.004
        and tr4n["status"] == "failed"
    )
    ok &= good4
    print(
        f"[4] start +10 meV, window +-6 meV: recentered start error {e4[0][1] * 1e3:+.2f} meV, worst "
        f"{max(abs(d) for _, d, _ in e4) * 1e3:.2f} meV over {len(e4)} points, stop='{tr4['stop_reason']}'; "
        f"without recentering: {tr4n['status']}  -> {'PASS' if good4 else 'FAIL'}"
    )

    # 5. asymmetric window: the midpoint is biased by (w_a - w_g)/2
    tr5 = BoundaryTracer(
        TraceConfig(mu0=t0["mu_coex"] + 0.010, **base),
        NucleationEngine(0.003, 0.009, seed=5),
        out / "recenter_asym.json",
        log=quiet,
    ).run(dict(branch="low"), dict(branch="high"))
    e5 = mu_errors(tr5)
    good5 = bool(e5) and e5[0][2] and abs(e5[0][1] - (0.003 - 0.009) / 2) < 0.0015
    ok &= good5
    print(
        f"[5] asymmetric window (+3 / -9 meV): recentered start error {e5[0][1] * 1e3:+.2f} meV vs expected bias "
        f"{(0.003 - 0.009) / 2 * 1e3:+.1f} meV  -> {'PASS' if good5 else 'FAIL'}"
    )

    # 6. a drifting trace (biased energies) leaves a 4 meV window mid-way
    res6 = {}
    for label, kw, eng in (
        ("recenter", {}, NucleationEngine(0.004, 0.004, ebias=0.03, seed=7)),
        (
            "plain",
            dict(recenter=False),
            NucleationEngine(0.004, 0.004, ebias=0.03, seed=7),
        ),
        (
            "fallback",
            {},
            NucleationEngine(0.004, 0.004, ebias=0.03, single=False, seed=7),
        ),
    ):
        tr6 = BoundaryTracer(
            TraceConfig(mu0=t0["mu_coex"], **base, **kw),
            eng,
            out / f"recenter_drift_{label}.json",
            log=quiet,
        ).run(dict(branch="low"), dict(branch="high"))
        e6 = mu_errors(tr6)
        res6[label] = (
            tr6,
            e6,
            min(T for T, _, _ in e6),
            max(abs(d) for _, d, _ in e6),
            tr6.get("n_recenter", 0),
        )
    good6 = (
        all(
            res6[k][0]["stop_reason"] == "reached t_stop"
            and res6[k][4] >= 1
            and res6[k][3] < 0.004
            for k in ("recenter", "fallback")
        )
        and res6["plain"][2] > 700
    )
    ok &= good6
    print(
        "[6] drifting trace, window +-4 meV: "
        + "; ".join(
            f"{k}: down to {v[2]:.0f} K, worst {v[3] * 1e3:.2f} meV, {v[4]} recentering(s)"
            for k, v in res6.items()
        )
        + f"  -> {'PASS' if good6 else 'FAIL'}"
    )

    # 7. a walker that melts (engine reports phase_ok=False): stop, no recentering
    class MeltingEngine(MeanFieldEngine):
        """Mean-field walkers, but alpha reports phase_ok=False above 1100 K."""

        def run(self, T, mu, sa, sg, tag=""):
            """As MeanFieldEngine.run, flagging alpha as melted above 1100 K."""
            a, g, sa2, sg2 = super().run(T, mu, sa, sg, tag)
            if T > 1100:
                a = dict(a, phase_ok=False, phase_note="solid fraction 0.1")
            return a, g, sa2, sg2

    cfg = TraceConfig(t0=T0, mu0=t0["mu_coex"], t_stop=1400.0, dt=50.0, dt_min=2.0)
    tr7 = BoundaryTracer(cfg, MeltingEngine(seed=8), out / "melt.json", log=quiet).run(
        dict(branch="low"), dict(branch="high")
    )
    top7 = max(p["T"] for p in tr7["points"])
    good7 = (
        top7 <= 1100
        and "new phase appeared" in (tr7["stop_reason"] or "")
        and not tr7.get("n_recenter")
    )
    ok &= good7
    print(
        f"[7] walker 'melts' above 1100 K: highest traced T {top7:.1f} K, stop='{tr7['stop_reason']}'  -> {'PASS' if good7 else 'FAIL'}"
    )

    report(
        [str(out / "down.json"), str(out / "up.json")],
        str(out / "report"),
        "Synthetic sub-regular solution: traced vs exact",
    )
    # overlay exact binodal on the report figure data for the record
    exact = [dict(T=T, **(truth(L0, L1, EB, T) or {})) for T in np.arange(600, tc, 25)]
    (out / "exact_binodal.json").write_text(json.dumps(exact, indent=2, default=float))
    print(f"engine calls: down {len(tr['points'])} pts; report -> {out / 'report'}")
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
