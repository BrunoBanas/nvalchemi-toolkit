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
"""Composition-temperature phase boundaries from semi-grand-canonical (SGC) MC isotherms.

Implements van de Walle & Asta, Modelling Simul. Mater. Sci. Eng. 10, 521 (2002):
  * eq. (2)-(3)  at fixed T:  d phi = -x d(dmu)            (thermodynamic integration)
  * eq. (6)-(7)  boundary where phi_alpha = phi_gamma, x = -d phi / d dmu of each phase
  * eq. (17)-(26) singularity (first-order jump) detection by polynomial extrapolation
  * eq. (29)     cross-temperature check  d dmu/d beta = (E_g - E_a)/(beta (x_g - x_a)) - dmu/beta

Conventions (binary A-B): x = mole fraction of B, dmu = mu_B - mu_A (so dG/dx = dmu),
phi = F - dmu*x per atom (mu_A taken as the energy zero), G(x) = phi + dmu*x.

Input: a tidy CSV (see benchmark/phase_boundary/README.md) or a directory of nvalchemi-style
*.equilibration.json files (--format nvalchemi-json).  Output: figure, summary JSON,
markdown report and per-temperature phi/G tables in --out.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

KB_EV = 8.617333262e-5  # eV/K
Z_99 = 2.5758293035489  # sqrt(2) erfinv(0.99), eq. (15) with alpha = 0.01

CATEGORICAL = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
]


# ----------------------------------------------------------------------------- data loading
def _f(v, default=None):
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return default
    return float(v)


def _b(v, default=True):
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return default
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "y", "t")
    return bool(v)


def load_csv(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            r = {k.strip(): v for k, v in r.items()}
            rows.append(
                dict(
                    T=float(r["T"]),
                    branch=str(r.get("branch", "all")).strip(),
                    mu=float(r["mu"]),
                    x=float(r["x"]),
                    x_se=_f(r.get("x_se"), None),
                    x_drift=_f(r.get("x_drift"), 0.0),
                    E=_f(r.get("E"), None),
                    E_se=_f(r.get("E_se"), None),
                    resolved=_b(r.get("resolved"), True),
                    order=_f(r.get("order"), None),
                    run_id=r.get("run_id", ""),
                )
            )
    return rows


def load_nvalchemi_json(directory, species=None):
    """*.equilibration.json from nvalchemi-toolkit benchmark/hybrid_sgc_npt/run_campaign.py.

    x / E are the last-window means; x_se is the single-window SE (combined SE of the two
    compared windows / sqrt 2); x_drift is last-minus-penultimate window mean.
    """
    rows = []
    files = sorted(
        glob.glob(os.path.join(directory, "**", "*.equilibration.json"), recursive=True)
    )
    for f in files:
        d = json.load(open(f))
        if "composition_gate" not in d or "chemical_potentials_ev" not in d:
            continue
        mus = d["chemical_potentials_ev"]
        names = species or list(mus)
        a, b = names[0], names[1]
        c, e = d["composition_gate"], d.get("energy_gate", {})
        rid = d.get("run_id", Path(f).name)
        m = re.search(r"\.([A-Za-z]+rich)\b", rid) or re.search(
            r"(A_rich|B_rich|Arich|Brich)", rid
        )
        step = re.search(r"\.dmu(\d+)", rid)
        rows.append(
            dict(
                T=float(d["temperature_K"]),
                branch=m.group(1) if m else "all",
                mu=float(mus[b]) - float(mus.get(a, 0.0)),
                x=float(c["mean_last_window"]),
                x_se=float(c["combined_standard_error"]) / math.sqrt(2),
                x_drift=float(c.get("difference", 0.0)),
                E=_f(e.get("mean_last_window")),
                E_se=(_f(e.get("combined_standard_error"), 0.0) or 0.0) / math.sqrt(2)
                or None,
                resolved=bool(d.get("resolved", True)),
                order=float(step.group(1)) if step else 0.0,
                run_id=rid,
            )
        )
    if not rows:
        sys.exit(
            f"no *.equilibration.json with composition_gate found under {directory}"
        )
    return rows


# ----------------------------------------------------------------------------- containers
@dataclass
class Series:
    """Ordered SGC points along one sweep (or one phase), sorted by increasing dmu."""

    mu: np.ndarray
    x: np.ndarray
    se: np.ndarray
    E: np.ndarray | None
    resolved: np.ndarray
    label: str = ""

    @classmethod
    def from_rows(cls, rows, label=""):
        rows = sorted(rows, key=lambda r: r["mu"])
        E = np.array([r["E"] if r["E"] is not None else np.nan for r in rows])
        return cls(
            np.array([r["mu"] for r in rows]),
            np.array([r["x"] for r in rows]),
            np.array([r["se_eff"] for r in rows]),
            None if np.all(np.isnan(E)) else E,
            np.array([r["resolved"] for r in rows]),
            label,
        )


@dataclass
class Phase:
    name: str  # "low" (A-rich side) or "high" (B-rich side)
    s: Series
    phi: np.ndarray | None = None  # phi at s.mu
    anchor: str = "unanchored"
    notes: list = field(default_factory=list)


# ----------------------------------------------------------------------------- eq. (17)-(26)
def _polyfit_predict(c, q, c_new, deg):
    X = np.vander(c, deg + 1, increasing=True)
    a, *_ = np.linalg.lstsq(X, q, rcond=None)
    xn = np.vander(np.atleast_1d(c_new), deg + 1, increasing=True)[0]
    return float(xn @ a), X, a, xn


def jump_test(c, q, se, c_new, q_new, se_new, z=Z_99, window=6):
    """Is q_new a statistically significant departure from the extrapolated sequence?"""
    c, q, se = c[-window:], q[-window:], se[-window:]
    n = len(c)
    if n < 3:
        return False, np.nan, np.nan
    best, best_cv = 0, np.inf
    for deg in range(0, min(2, n - 2) + 1):  # eq. (17) leave-one-out CV
        errs = [
            (_polyfit_predict(np.delete(c, i), np.delete(q, i), c[i], deg)[0] - q[i])
            ** 2
            for i in range(n)
        ]
        if np.mean(errs) < best_cv:
            best, best_cv = deg, np.mean(errs)
    pred, X, a, xn = _polyfit_predict(c, q, c_new, best)
    s2_22 = float(np.mean(se**2))  # eq. (22)
    dof = n - best - 1
    s2_23 = float(np.sum((q - X @ a) ** 2) / dof) if dof > 0 else 0.0  # eq. (23)
    s2 = max(s2_22, s2_23)
    v = float(xn @ (s2 * np.linalg.pinv(X.T @ X)) @ xn)  # eq. (21), (25)
    thresh = z * math.sqrt(v + max(se_new**2, s2))  # eq. (26)
    return abs(q_new - pred) >= thresh, pred, thresh


def segment_branch(rows, ascending, min_jump, z, other_phase=None, allow_fallback=True):
    """Walk one branch in sweep order; split it at VERIFIED first-order jumps.

    A candidate jump is a point that fails the eq. (26) extrapolation test and moves x in the
    sweep direction by more than min_jump. A single sweep cannot tell a first-order jump from
    a steep continuous rise, so a candidate is confirmed with the paper's Fig. 5 criterion,
    applied against the OTHER PHASE's sequence (the opposite branch's own pre-jump segment,
    ``other_phase``), not against the opposite walker's raw data -- once that walker has itself
    transformed, its raw data sit in the same phase as this branch and would wrongly veto the jump:

      1. the two sequences are distinct phases: where their dmu ranges overlap, this branch's
         pre-jump points differ significantly from the other phase (hysteresis), and
      2. the new point is better predicted by the other phase's sequence than by its own
         extrapolation (the walker has fallen into the other phase).

    Rejected candidates immediately preceding a confirmed jump whose run was still drifting
    toward the other phase are points caught mid-transformation (neither phase's equilibrium
    state); they are returned in ``transforming`` and excluded from both phases.
    """
    seq = sorted(
        rows,
        key=lambda r: (
            r["order"] if r["order"] is not None else 0.0,
            r["mu"] if ascending else -r["mu"],
        ),
    )
    toward = 1.0 if ascending else -1.0  # direction of the other phase in x

    def _moving(p):
        """Run still drifting TOWARD the other phase between its last two windows."""
        return toward * (p.get("x_drift") or 0.0) > z * (p.get("x_se") or p["se_eff"])

    segments, cur, jumps, candidates, transforming = [], [seq[0]], [], [], []
    pending = []  # consecutive rejected candidates (possible onset)
    held = []  # pending points that are drifting: kept OUT of the trend
    for r in seq[1:]:
        c = np.array([p["mu"] for p in cur])
        q = np.array([p["x"] for p in cur])
        se = np.array([p["se_eff"] for p in cur])
        # A run drifting toward the other phase is tested with its RAW error bar: that drift is the
        # transformation signal, and folding it into the error bar (as for ordinary unresolved runs)
        # would hide exactly the point where the walker starts to change phase.
        se_new = (r.get("x_se") or r["se_eff"]) if _moving(r) else r["se_eff"]
        flag, pred, thr = jump_test(c, q, se, r["mu"], r["x"], se_new, z=z)
        dx = r["x"] - cur[-1]["x"]
        sign_ok = (
            (dx > 0) if ascending else (dx < 0)
        )  # a real transition moves x WITH the sweep
        if not (flag and sign_ok and abs(r["x"] - pred) >= min_jump):
            cur.extend(held)
            held = []  # the suspected onset was a fluctuation after all
            cur.sort(key=lambda p: p["mu"] if ascending else -p["mu"])
            cur.append(r)
            pending = []
            continue
        rec = dict(
            mu_before=cur[-1]["mu"],
            mu_after=r["mu"],
            x_before=cur[-1]["x"],
            x_after=r["x"],
            predicted=pred,
            threshold=thr,
        )
        verdict = "unverifiable (no other-phase sequence to compare with)"
        if other_phase is not None and len(other_phase[0]) >= 2:
            omu, ox, ose = other_phase
            steps = np.abs(np.diff(np.sort(omu)))
            margin = float(np.median(steps[steps > 0])) if np.any(steps > 0) else 0.0
            order = np.argsort(omu)
            om, oxs, oses = omu[order], ox[order], ose[order]

            def other_at(mu, reach):
                """Other phase's x at mu: interpolated, or linearly extrapolated from its nearest
                end if mu lies within ``reach`` of it; None if too far to say."""
                if om[0] <= mu <= om[-1]:
                    return float(np.interp(mu, om, oxs))
                if mu > om[-1] and mu - om[-1] <= reach:
                    return float(
                        oxs[-1]
                        + (oxs[-1] - oxs[-2]) / (om[-1] - om[-2]) * (mu - om[-1])
                    )
                if mu < om[0] and om[0] - mu <= reach:
                    return float(
                        oxs[0] + (oxs[1] - oxs[0]) / (om[1] - om[0]) * (mu - om[0])
                    )
                return None

            x_other = other_at(r["mu"], margin)
            if x_other is None:
                verdict = "unverifiable (other phase not sampled at this dmu)"
            else:
                se_other = float(np.interp(r["mu"], om, oses))
                miss_own = abs(r["x"] - pred)
                miss_other = abs(r["x"] - x_other)
                # Fig. 5: (1) the two sequences are distinct phases -- they predict clearly different x
                # at the new dmu (a steep single-phase region has both sequences on the same curve);
                # (2) the new point is better predicted by the other phase's sequence.
                own_in = [p for p in cur if om[0] <= p["mu"] <= om[-1]]
                basis = "overlap" if len(own_in) >= 2 else "no-overlap"
                if len(own_in) >= 2:
                    # where both sequences were sampled, compare them directly: identical curves mean
                    # one continuous phase (e.g. above T_c), however badly a steep trend extrapolates
                    distinct = any(
                        abs(p["x"] - np.interp(p["mu"], om, oxs))
                        > max(
                            z * math.hypot(p["se_eff"], np.interp(p["mu"], om, oses)),
                            min_jump,
                        )
                        for p in own_in
                    )
                else:
                    # No shared dmu (narrow hysteresis loop). A first-order jump crosses a gap: just
                    # BEFORE it the walker was far from the other phase. On one continuous (steep)
                    # curve the two walkers meet smoothly, so the other phase extrapolated to the
                    # pre-jump point sits right on top of it.
                    x_other_before = other_at(rec["mu_before"], 2 * margin)
                    se_b = cur[-1]["se_eff"]
                    distinct = x_other_before is not None and abs(
                        rec["x_before"] - x_other_before
                    ) > max(z * math.hypot(se_b, se_other), min_jump)
                tol = max(z * math.hypot(se_new, se_other), 0.5 * miss_own)
                rec["basis"] = basis
                if distinct and miss_other < miss_own and miss_other < tol:
                    verdict = (
                        "confirmed"
                        if (basis == "overlap" or allow_fallback)
                        else (
                            "unverifiable (no shared dmu and the other walker never crossed: a steep "
                            "single-phase rise looks the same)"
                        )
                    )
                elif not distinct:
                    verdict = "rejected (branch and other phase coincide: continuous steep region, not a transition)"
                else:
                    verdict = "rejected (departs from its own sequence but is not yet in the other phase)"
        rec["verdict"] = verdict
        if verdict == "confirmed":
            # A rejected candidate right before the jump is only "mid-transformation" if the run itself
            # was still drifting TOWARD the other phase (late-minus-early window mean beyond z*SE).
            # Without that evidence it is a metastable point on a steepening branch and stays in its phase.
            moving = [p for p in pending if _moving(p)]
            for p in moving:
                if p in cur:
                    cur.remove(p)
                transforming.append(p)
            held = []
            rec["onset_mu"] = moving[0]["mu"] if moving else r["mu"]
            jumps.append(rec)
            segments.append(cur)
            cur = [r]
            pending = []
            continue
        candidates.append(rec)
        pending.append(r)
        if _moving(r):
            held.append(r)  # keep it out of the trend that judges the next point
        else:
            cur.append(r)
    cur.extend(held)  # ladder ended mid-suspicion: nothing confirmed, keep them
    cur.sort(key=lambda p: p["mu"] if ascending else -p["mu"])
    segments.append(cur)
    return segments, jumps, candidates, transforming


def phase_x(s, mu):
    """x of one phase's series at dmu, extrapolated linearly (non-decreasing) past its ends."""
    k = min(3, len(s.mu))
    s0 = (
        max(np.polyfit(s.mu[:k], s.x[:k], 1)[0], 0.0)
        if k >= 2 and np.ptp(s.mu[:k]) > 0
        else 0.0
    )
    s1 = (
        max(np.polyfit(s.mu[-k:], s.x[-k:], 1)[0], 0.0)
        if k >= 2 and np.ptp(s.mu[-k:]) > 0
        else 0.0
    )
    if mu > s.mu[-1]:
        return float(np.clip(s.x[-1] + s1 * (mu - s.mu[-1]), 0, 1))
    if mu < s.mu[0]:
        return float(np.clip(s.x[0] + s0 * (mu - s.mu[0]), 0, 1))
    return float(np.interp(mu, s.mu, s.x))


# ----------------------------------------------------------------------------- phi integration
def integrate(mu, x, phi0=0.0, i0=0):
    """phi(mu) with phi[i0] = phi0, trapezoid on d phi = -x d mu (eq. 3 at fixed beta)."""
    cum = np.concatenate([[0.0], np.cumsum(-0.5 * np.diff(mu) * (x[1:] + x[:-1]))])
    return phi0 + cum - cum[i0]


def dilute_tail_low(x1, T):
    """phi_low(mu_1) - F_A = -int_{-inf}^{mu_1} x dmu  for a Langmuir/Henry tail = kT ln(1-x1)."""
    return KB_EV * T * math.log(max(1.0 - x1, 1e-300))


def dilute_tail_high(xN, T):
    """phi_high(mu_N) + mu_N - F_B = kT ln(x_N)  (mirror of the low tail)."""
    return KB_EV * T * math.log(max(xN, 1e-300))


def estimate_pure_energy(s: Series, end):
    """Rigid-lattice estimate F_pure ~ E_pure: quadratic E(x) extrapolated to x=0 or 1."""
    if s.E is None:
        return None
    ok = ~np.isnan(s.E)
    x, E = s.x[ok], s.E[ok]
    if len(x) < 3:
        return None
    idx = np.argsort(x)[:4] if end == "A" else np.argsort(x)[-4:]
    deg = min(2, len(idx) - 1)
    if np.ptp(x[idx]) < 1e-3:  # points already at the pure end
        return float(np.mean(E[idx]))
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = np.polyfit(x[idx], E[idx], deg)
    return float(np.polyval(p, 0.0 if end == "A" else 1.0))


# ----------------------------------------------------------------------------- one isotherm
def analyse_isotherm(T, rows, opts, pure=None, rng=None, perturb=False):
    """Returns dict with phases, boundary (or None) and diagnostics for one temperature."""
    rows = [dict(r) for r in rows]
    if perturb:
        for r in rows:
            r["x"] = float(np.clip(r["x"] + rng.normal(0, r["se_eff"]), 1e-6, 1 - 1e-6))
    branches = {}
    for r in rows:
        branches.setdefault(r["branch"], []).append(r)
    means = {b: np.mean([r["x"] for r in rs]) for b, rs in branches.items()}
    labels = sorted(branches, key=lambda b: means[b])
    diag = dict(T=T, branches={}, jumps=[], jump_candidates=[], warnings=[])

    # --- segment each branch and assign segments to phases (Fig. 5 logic, binary case)
    low_rows, high_rows = [], []
    home_rows = None
    if len(labels) == 1:
        segs, jumps, cands, _trans = segment_branch(
            branches[labels[0]], ascending=True, min_jump=opts.min_jump, z=opts.z
        )
        diag["jumps"] += [dict(branch=labels[0], **j) for j in jumps]
        diag["jump_candidates"] += [dict(branch=labels[0], **j) for j in cands]
        low_rows += segs[0]
        for s in segs[1:]:
            high_rows += s
    else:
        if len(labels) > 2:
            diag["warnings"].append(
                f"{len(labels)} branches; using lowest-x {labels[0]!r} and highest-x {labels[-1]!r}, "
                f"others pooled by nearest mean composition"
            )

        # Each branch is verified against the OTHER branch's home-phase segment. That segment is
        # only known after the other branch is itself segmented, so iterate to self-consistency,
        # starting from the raw opposite branch.
        def _arr(rs):
            rs = sorted(rs, key=lambda r: r["mu"])
            return (
                np.array([r["mu"] for r in rs]),
                np.array([r["x"] for r in rs]),
                np.array([r["se_eff"] for r in rs]),
            )

        def _segment_all(allow):
            home_seg = {b: branches[b] for b in labels}
            result = {}
            for _ in range(6):
                new_result = {}
                for i, b in enumerate(labels):
                    ascending = (
                        means[b] <= np.median(list(means.values()))
                        if len(labels) > 2
                        else (i == 0)
                    )
                    ob = labels[-1] if i == 0 else labels[0]
                    new_result[b] = (ascending,) + segment_branch(
                        branches[b],
                        ascending=ascending,
                        min_jump=opts.min_jump,
                        z=opts.z,
                        other_phase=_arr(home_seg[ob]),
                        allow_fallback=allow[b],
                    )
                key = {
                    b: [(j["mu_before"], j["mu_after"]) for j in new_result[b][2]]
                    for b in labels
                }
                if result and key == {
                    b: [(j["mu_before"], j["mu_after"]) for j in result[b][2]]
                    for b in labels
                }:
                    return new_result
                result = new_result
                home_seg = {b: result[b][1][0] for b in labels}
            return result

        # A jump confirmed WITHOUT shared dmu (narrow loop) is only trusted if the opposite walker also
        # crossed: two opposite crossings are what a first-order loop looks like, while a lone
        # "crossing" can be one walker climbing a steep continuous curve (e.g. just above T_c).
        allow = {b: True for b in labels}
        result = _segment_all(allow)
        for _ in range(2):
            lone = [
                b
                for b in (labels[0], labels[-1])
                if any(j.get("basis") == "no-overlap" for j in result[b][2])
                and not result[labels[-1] if b == labels[0] else labels[0]][2]
            ]
            if not lone:
                break
            for b in lone:
                allow[b] = False
            result = _segment_all(allow)
        diag["transforming"] = []
        home_rows = {"low": result[labels[0]][1][0], "high": result[labels[-1]][1][0]}
        for b in labels:
            ascending, segs, jumps, cands, trans = result[b]
            diag["jumps"] += [dict(branch=b, **j) for j in jumps]
            diag["jump_candidates"] += [dict(branch=b, **j) for j in cands]
            diag["transforming"] += [
                dict(branch=b, mu=p["mu"], x=p["x"]) for p in trans
            ]
            home, other = (low_rows, high_rows) if ascending else (high_rows, low_rows)
            home += segs[0]
            for s_ in segs[1:]:  # after a jump the walker sits in the other phase
                other += s_
    for b in labels:
        rs = branches[b]
        diag["branches"][b] = dict(
            n=len(rs),
            mu_range=[min(r["mu"] for r in rs), max(r["mu"] for r in rs)],
            x_range=[min(r["x"] for r in rs), max(r["x"] for r in rs)],
            n_unresolved=sum(not r["resolved"] for r in rs),
        )

    # raw-branch hysteresis where the two outermost branches share dmu values
    hyst = []
    if len(labels) >= 2:
        A, B = (
            sorted(branches[labels[0]], key=lambda r: r["mu"]),
            sorted(branches[labels[-1]], key=lambda r: r["mu"]),
        )
        lo, hi = max(A[0]["mu"], B[0]["mu"]), min(A[-1]["mu"], B[-1]["mu"])
        if hi >= lo - 1e-12:
            muA = np.array([r["mu"] for r in A])
            xA = np.array([r["x"] for r in A])
            sA = np.array([r["se_eff"] for r in A])
            muB = np.array([r["mu"] for r in B])
            xB = np.array([r["x"] for r in B])
            sB = np.array([r["se_eff"] for r in B])
            grid = np.unique(np.concatenate([muA, muB]))
            grid = grid[(grid >= lo - 1e-12) & (grid <= hi + 1e-12)]
            for m in grid:
                dx = np.interp(m, muB, xB) - np.interp(m, muA, xA)
                hyst.append(
                    dict(
                        mu=float(m),
                        dx=float(dx),
                        dx_se=float(
                            math.hypot(np.interp(m, muA, sA), np.interp(m, muB, sB))
                        ),
                    )
                )
    diag["hysteresis"] = hyst
    if len(labels) >= 2 and len(hyst) <= 1:
        diag["warnings"].append(
            "the two branches share at most one dmu value, so hysteresis (the paper's evidence for "
            "a first-order transition) cannot be tested; extend each ladder past the other's start"
        )
    diag["hysteresis_significant"] = any(
        abs(h["dx"]) > opts.z * h["dx_se"] and abs(h["dx"]) > opts.min_jump
        for h in hyst
    )

    # --- junction continuity when the two outermost branches share no dmu: extrapolate each
    # branch (eq. 26 test) to the nearest point of the other; both failing => discontinuous
    diag["junction"] = None
    if len(labels) >= 2 and not hyst:
        A = sorted(branches[labels[0]], key=lambda r: r["mu"])
        B = sorted(branches[labels[-1]], key=lambda r: r["mu"])
        if A[-1]["mu"] < B[0]["mu"]:

            def arr(rs):
                return (
                    np.array([r["mu"] for r in rs]),
                    np.array([r["x"] for r in rs]),
                    np.array([r["se_eff"] for r in rs]),
                )

            ma, xa_, sa = arr(A)
            mb, xb_, sb = arr(B)
            fa, pa, ta = jump_test(
                ma, xa_, sa, B[0]["mu"], B[0]["x"], B[0]["se_eff"], z=opts.z
            )
            fb, pb, tb = jump_test(
                mb[::-1],
                xb_[::-1],
                sb[::-1],
                A[-1]["mu"],
                A[-1]["x"],
                A[-1]["se_eff"],
                z=opts.z,
            )
            disc = bool(fa and fb and abs(B[0]["x"] - A[-1]["x"]) > opts.min_jump)
            diag["junction"] = dict(
                mu_low_end=A[-1]["mu"],
                x_low_end=A[-1]["x"],
                mu_high_start=B[0]["mu"],
                x_high_start=B[0]["x"],
                discontinuous=disc,
            )
            diag["warnings"].append(
                f"branches do not overlap (gap {A[-1]['mu']:.4f} → {B[0]['mu']:.4f} eV); junction judged "
                f"{'DIScontinuous' if disc else 'continuous'} by extrapolation only — any boundary is tentative"
            )

    # --- anchor-free hysteresis bracket (van de Walle & Asta Fig. 6): a low-x walker that jumps UP
    # has passed coexistence, so dmu_coex <= where it left; a high-x walker that jumps DOWN gives
    # dmu_coex >= where it left. "onset" uses the first point that departed from its own sequence
    # (tighter); "wide" uses the point where the walker had fully arrived in the other phase.
    diag["bracket"] = None
    if len(labels) >= 2:
        up = [j for j in diag["jumps"] if j["branch"] == labels[0]]
        down = [j for j in diag["jumps"] if j["branch"] == labels[-1]]
        if up or down:
            br = dict(
                upper_wide=min(j["mu_after"] for j in up) if up else None,
                upper_onset=min(j["onset_mu"] for j in up) if up else None,
                lower_wide=max(j["mu_after"] for j in down) if down else None,
                lower_onset=max(j["onset_mu"] for j in down) if down else None,
            )
            if up and down:
                if br["lower_onset"] <= br["upper_onset"]:
                    br["estimate"] = 0.5 * (br["lower_onset"] + br["upper_onset"])
                    br["half_width"] = 0.5 * (br["upper_onset"] - br["lower_onset"])
                    br["basis"] = "onset"
                else:
                    br["estimate"] = 0.5 * (br["lower_wide"] + br["upper_wide"])
                    br["half_width"] = 0.5 * abs(br["upper_wide"] - br["lower_wide"])
                    br["basis"] = "wide (onset bounds crossed)"
            diag["bracket"] = br

    # --- decide whether low/high are one connected phase (continuity anchor)
    junction_break = bool(diag["junction"] and diag["junction"]["discontinuous"])
    connected = (
        not diag["jumps"] and not diag["hysteresis_significant"] and not junction_break
    )
    if not high_rows or not low_rows:
        connected = True

    phases = []
    if connected:
        s = Series.from_rows(low_rows + high_rows, "single")
        # average duplicate dmu (e.g. both walkers at the junction)
        mu_u = np.unique(s.mu)
        if len(mu_u) < len(s.mu):
            x_u = np.array([s.x[s.mu == m].mean() for m in mu_u])
            se_u = np.array(
                [
                    np.sqrt(np.sum(s.se[s.mu == m] ** 2)) / np.sum(s.mu == m)
                    for m in mu_u
                ]
            )
            E_u = (
                None
                if s.E is None
                else np.array([np.nanmean(s.E[s.mu == m]) for m in mu_u])
            )
            r_u = np.array([bool(np.all(s.resolved[s.mu == m])) for m in mu_u])
            s = Series(mu_u, x_u, se_u, E_u, r_u, "single")
        ph = Phase("single", s)
        FA = pure.get("A") if pure else None
        if FA is None and opts.anchor == "energy":
            FA = estimate_pure_energy(s, "A")
        ph.phi = integrate(
            s.mu, s.x, (FA if FA is not None else 0.0) + dilute_tail_low(s.x[0], T), 0
        )
        ph.anchor = "continuity" + (
            " + pure-A tail" if FA is not None else " (phi zero arbitrary)"
        )
        phases.append(ph)
    else:
        for name, rs in (("low", low_rows), ("high", high_rows)):
            s = Series.from_rows(rs, name)
            ph = Phase(name, s)
            end = "A" if name == "low" else "B"
            F = pure.get(end) if pure else None
            src = "supplied"
            if F is None and opts.anchor in ("energy", "auto"):
                F, src = (
                    estimate_pure_energy(s, end),
                    "E-extrapolated (rigid-lattice F=E assumption)",
                )
            if F is None:
                ph.phi = integrate(s.mu, s.x, 0.0, 0)
                ph.anchor = "unanchored"
            elif name == "low":
                ph.phi = integrate(s.mu, s.x, F + dilute_tail_low(s.x[0], T), 0)
                ph.anchor = f"pure-A tail, F_A {src}"
                if s.x[0] > opts.max_tail_x:
                    ph.notes.append(
                        f"lowest x = {s.x[0]:.3f} > {opts.max_tail_x}: dilute-tail anchor is approximate"
                    )
            else:
                n = len(s.mu) - 1
                ph.phi = integrate(
                    s.mu, s.x, F - s.mu[n] + dilute_tail_high(s.x[n], T), n
                )
                ph.anchor = f"pure-B tail, F_B {src}"
                if 1 - s.x[n] > opts.max_tail_x:
                    ph.notes.append(
                        f"highest x = {s.x[n]:.3f} < {1 - opts.max_tail_x}: dilute-tail anchor is approximate"
                    )
            phases.append(ph)

    out = dict(
        T=T,
        phases=phases,
        diag=diag,
        connected=connected,
        boundary=None,
        home={k: Series.from_rows(v, k) for k, v in home_rows.items()}
        if home_rows
        else None,
    )
    if connected:
        unverif = [
            j
            for j in diag["jump_candidates"]
            if j["verdict"].startswith("unverifiable")
        ]
        if unverif:
            out["status"] = "inconclusive"
            diag["warnings"].append(
                f"{len(unverif)} candidate first-order jump(s) could not be checked against an opposite sweep; a single "
                "sweep cannot tell a jump from a steep continuous rise. Run the reverse sweep over the same dmu window"
            )
        else:
            out["status"] = "single-phase"
        return out
    if any(p.anchor == "unanchored" for p in phases):
        if boundary_from_bracket(out, phases, diag):
            diag["warnings"].append(
                "phi unanchored (no pure-end free energies); boundary taken from the "
                "anchor-free hysteresis bracket instead"
            )
            return out
        out["status"] = "unanchored"
        diag["warnings"].append(
            "two phases detected but no pure-end free energies available to anchor phi; "
            "pass --pure-free-energies or --anchor energy"
        )
        return out

    # --- phi_low = phi_high (eqs. 6-7)
    lo_p, hi_p = phases
    m_lo, m_hi = (
        max(lo_p.s.mu.min(), hi_p.s.mu.min()),
        min(lo_p.s.mu.max(), hi_p.s.mu.max()),
    )
    # search a padded window: outside a phase's sampled range phi is continued linearly with its
    # end slope -x (first-order extrapolation); a crossing found there is flagged "extrapolated"
    steps = np.concatenate([np.diff(lo_p.s.mu), np.diff(hi_p.s.mu)])
    pad = (
        2 * float(np.median(np.abs(steps[steps != 0]))) if np.any(steps != 0) else 0.01
    )
    if m_hi <= m_lo:
        m_lo, m_hi = (
            min(lo_p.s.mu.max(), hi_p.s.mu.min()),
            max(lo_p.s.mu.max(), hi_p.s.mu.min()),
        )
    grid = np.linspace(m_lo - pad, m_hi + pad, 4001)

    def end_slopes(p):
        mu, x = p.s.mu, p.s.x
        k = min(3, len(mu))
        s0 = np.polyfit(mu[:k], x[:k], 1)[0] if k >= 2 and np.ptp(mu[:k]) > 0 else 0.0
        s1 = (
            np.polyfit(mu[-k:], x[-k:], 1)[0] if k >= 2 and np.ptp(mu[-k:]) > 0 else 0.0
        )
        return max(s0, 0.0), max(s1, 0.0)  # x(dmu) is non-decreasing in a stable phase

    def x_at(p, g):
        mu, x = p.s.mu, p.s.x
        s0, s1 = end_slopes(p)
        val = np.interp(g, mu, x)
        val = np.where(g > mu[-1], x[-1] + s1 * (g - mu[-1]), val)
        return np.clip(np.where(g < mu[0], x[0] + s0 * (g - mu[0]), val), 0.0, 1.0)

    def phi_at(p, g):
        # beyond the data: x continued linearly, so phi = phi_end - x_end d - s d^2 / 2
        mu, x, phi = p.s.mu, p.s.x, p.phi
        s0, s1 = end_slopes(p)
        val = np.interp(g, mu, phi)
        d1, d0 = g - mu[-1], g - mu[0]
        val = np.where(g > mu[-1], phi[-1] - x[-1] * d1 - 0.5 * s1 * d1**2, val)
        return np.where(g < mu[0], phi[0] - x[0] * d0 - 0.5 * s0 * d0**2, val)

    d = phi_at(hi_p, grid) - phi_at(lo_p, grid)
    sgn = np.sign(d)
    cross = np.where(np.diff(sgn) != 0)[0]
    if len(cross) == 0:
        out["status"] = "no-crossing"
        diag["warnings"].append(
            f"phi_low - phi_high keeps one sign over dmu in [{m_lo:.4f}, {m_hi:.4f}]: "
            f"{'high' if d.mean() < 0 else 'low'}-x phase stable throughout; coexistence lies outside"
        )
        if boundary_from_bracket(out, phases, diag):
            diag["warnings"].append(
                "the walkers' own transitions contradict that (hysteresis bracket exists), so the "
                "phi anchors are the suspect part; boundary taken from the anchor-free bracket"
            )
        return out
    i = cross[0]
    mu_c = grid[i] - d[i] * (grid[i + 1] - grid[i]) / (d[i + 1] - d[i])
    # x = -d phi/d dmu of each phase at mu_c (eqs. 6-7); np.interp clamps to the end value outside
    xa, xg = (
        float(x_at(lo_p, np.array([mu_c]))[0]),
        float(x_at(hi_p, np.array([mu_c]))[0]),
    )
    Ea = float(np.interp(mu_c, lo_p.s.mu, lo_p.s.E)) if lo_p.s.E is not None else None
    Eg = float(np.interp(mu_c, hi_p.s.mu, hi_p.s.E)) if hi_p.s.E is not None else None
    tol = 0.25 * pad
    inside = (lo_p.s.mu.min() - tol <= mu_c <= lo_p.s.mu.max() + tol) and (
        hi_p.s.mu.min() - tol <= mu_c <= hi_p.s.mu.max() + tol
    )
    out["boundary"] = dict(
        mu_coex=float(mu_c),
        x_alpha=xa,
        x_gamma=xg,
        E_alpha=Ea,
        E_gamma=Eg,
        extrapolated=bool(not inside),
    )
    out["status"] = (
        "boundary-confirmed"
        if (inside and diag["hysteresis_significant"])
        else "boundary-tentative"
    )
    br = diag.get("bracket") or {}
    lo_b = br.get("lower_wide")
    hi_b = br.get("upper_wide")
    if (lo_b is not None and mu_c < lo_b) or (hi_b is not None and mu_c > hi_b):
        diag["warnings"].append(
            f"phi crossing at {mu_c:.4f} eV lies outside the hysteresis bracket "
            f"[{lo_b}, {hi_b}]: the anchors disagree with the simulated transition"
        )
        out["status"] = "boundary-tentative"
    return out


def boundary_from_bracket(out, phases, diag):
    """Anchor-free boundary: dmu_coex from the hysteresis bracket, x from each phase there."""
    br = diag.get("bracket") or {}
    if "estimate" not in br or len(phases) != 2:
        return False
    lo_p, hi_p = phases
    mu_c, hw = br["estimate"], br["half_width"]
    # Each coexisting composition is read from the walker that approached dmu_coex from INSIDE that
    # phase (its home segment). The other walker's points just after its jump sit in the same phase
    # but are still relaxing after arrival, and mixing them in biases the extrapolation.
    home = out.get("home") or {}
    s_lo = home.get("low", lo_p.s)
    s_hi = home.get("high", hi_p.s)
    out["boundary"] = dict(
        mu_coex=float(mu_c),
        x_alpha=phase_x(s_lo, mu_c),
        x_gamma=phase_x(s_hi, mu_c),
        E_alpha=None,
        E_gamma=None,
        extrapolated=False,
        from_hysteresis_bracket=True,
        mu_coex_half_width=float(hw),
        x_alpha_range=[phase_x(s_lo, mu_c - hw), phase_x(s_lo, mu_c + hw)],
        x_gamma_range=[phase_x(s_hi, mu_c - hw), phase_x(s_hi, mu_c + hw)],
        x_alpha_all_walkers=phase_x(lo_p.s, mu_c),
        x_gamma_all_walkers=phase_x(hi_p.s, mu_c),
    )
    out["status"] = "boundary-bracketed"
    return True


# ----------------------------------------------------------------------------- G(x), hull, stability
def g_of_x(ph: Phase):
    return ph.phi + ph.s.mu * ph.s.x


def hull_excess(xs, Gs):
    order = np.argsort(xs)
    xs, Gs = xs[order], Gs[order]
    hull = []
    for p in zip(xs, Gs):
        while (
            len(hull) >= 2
            and (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1])
            - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0])
            <= 0
        ):
            hull.pop()
        hull.append(p)
    hx, hG = map(np.array, zip(*hull))
    return float(np.max(Gs - np.interp(xs, hx, hG))), hx, hG


# ----------------------------------------------------------------------------- eq. (29)
def clausius_clapeyron_check(results):
    """Predict dmu_coex at the next temperature from eq. (29) (Heun average of both ends)."""
    bs = [
        (r["T"], r["boundary"])
        for r in results
        if r["boundary"] and r["boundary"]["E_alpha"] is not None
    ]
    bs.sort()
    checks = []
    for (T1, b1), (T2, b2) in zip(bs, bs[1:]):

        def slope(T, b):
            beta = 1.0 / (KB_EV * T)
            return (b["E_gamma"] - b["E_alpha"]) / (
                beta * (b["x_gamma"] - b["x_alpha"])
            ) - b["mu_coex"] / beta

        dbeta = 1 / (KB_EV * T2) - 1 / (KB_EV * T1)
        pred = b1["mu_coex"] + 0.5 * (slope(T1, b1) + slope(T2, b2)) * dbeta
        checks.append(
            dict(
                T_from=T1,
                T_to=T2,
                mu_coex_predicted=pred,
                mu_coex_measured=b2["mu_coex"],
                residual=b2["mu_coex"] - pred,
            )
        )
    return checks


# ----------------------------------------------------------------------------- plotting
def make_figure(results, boots, opts, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ts = [r["T"] for r in results]
    if len(Ts) <= len(CATEGORICAL):
        col = {T: CATEGORICAL[i] for i, T in enumerate(Ts)}
    else:
        cmap = plt.get_cmap("cividis")
        col = {T: cmap(i / (len(Ts) - 1)) for i, T in enumerate(Ts)}
    fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    ax_x, ax_phi, ax_pd, ax_G, ax_chi, ax_h = axs.flat
    A, B = opts.species
    for r in results:
        T, c = r["T"], col[r["T"]]
        rows = r["rows"]
        for b in sorted({q["branch"] for q in rows}):
            rs = sorted([q for q in rows if q["branch"] == b], key=lambda q: q["mu"])
            mu = np.array([q["mu"] for q in rs])
            x = np.array([q["x"] for q in rs])
            se = np.array([q["se_eff"] for q in rs])
            ok = np.array([q["resolved"] for q in rs])
            ax_x.errorbar(mu, x, 2 * se, color=c, lw=1.3, elinewidth=0.7)
            ax_x.scatter(
                mu[ok], x[ok], s=26, color=c, edgecolor="white", lw=0.8, zorder=3
            )
            ax_x.scatter(
                mu[~ok], x[~ok], s=26, facecolor="white", edgecolor=c, lw=1.1, zorder=3
            )
        # phi relative to a common straight line so curvature/crossings are visible
        for p, ls in zip(r["phases"], ("-", "--")):
            ref = r["phi_ref"](p.s.mu)
            ax_phi.plot(
                p.s.mu,
                1e3 * (p.phi - ref),
                color=c,
                ls=ls,
                lw=1.8,
                marker="o",
                ms=3,
                mec="white",
                label=f"{T:g} K {p.name}",
            )
            G = g_of_x(p)
            ax_G.plot(
                p.s.x,
                1e3 * (G - r["G_ref"](p.s.x)),
                color=c,
                ls=ls,
                lw=1.8,
                marker="o",
                ms=3,
                mec="white",
                label=f"{T:g} K {p.name}",
            )
            xm = 0.5 * (p.s.x[1:] + p.s.x[:-1])
            dx = np.diff(p.s.x)
            good = (np.abs(dx) > 2 * np.hypot(p.s.se[1:], p.s.se[:-1])) & (
                np.diff(p.s.mu) != 0
            )
            if good.any():
                with np.errstate(all="ignore"):
                    ax_chi.plot(
                        xm[good],
                        (np.diff(p.s.mu) / dx)[good],
                        ls="none",
                        marker="o",
                        ms=4,
                        color=c,
                        mec="white",
                    )
        xs = np.linspace(0.02, 0.98, 200)
        ax_chi.plot(
            xs,
            KB_EV * T / (xs * (1 - xs)),
            color=c,
            lw=1,
            ls=":",
            label=f"{T:g} K ideal",
        )
        if r["boundary"]:
            b = r["boundary"]
            ax_phi.axvline(b["mu_coex"], color=c, lw=0.8, ls=":")
            ax_x.axvline(b["mu_coex"], color=c, lw=0.8, ls=":")
        h = r["diag"]["hysteresis"]
        if h:
            ax_h.errorbar(
                [q["mu"] for q in h],
                [q["dx"] for q in h],
                [2 * q["dx_se"] for q in h],
                color=c,
                marker="o",
                ms=4,
                lw=1,
                label=f"{T:g} K",
            )
        # T-x panel: sampled single-phase ranges + boundaries
        for p in r["phases"]:
            ax_pd.plot(
                [p.s.x.min(), p.s.x.max()],
                [T, T],
                color=c,
                lw=6,
                alpha=0.25,
                solid_capstyle="butt",
            )
        if r["boundary"]:
            bt = boots.get(T, {})
            for key in ("x_alpha", "x_gamma"):
                lo_w, hi_w = (
                    (bt[key][1] - bt[key][0], bt[key][2] - bt[key][1])
                    if bt and key in bt
                    else (0.0, 0.0)
                )
                ax_pd.errorbar(
                    r["boundary"][key],
                    T,
                    xerr=[[lo_w], [hi_w]],
                    color=c,
                    marker="o" if not r["boundary"]["extrapolated"] else "D",
                    ms=7,
                    mfc=c if r["status"] == "boundary-confirmed" else "white",
                    mec=c,
                    capsize=3,
                )
    xa = [
        (r["T"], r["boundary"]["x_alpha"], r["boundary"]["x_gamma"])
        for r in results
        if r["boundary"]
    ]
    if len(xa) >= 2:
        xa.sort()
        ax_pd.plot([q[1] for q in xa], [q[0] for q in xa], color="0.4", lw=1)
        ax_pd.plot([q[2] for q in xa], [q[0] for q in xa], color="0.4", lw=1)
    ax_pd.set_xlim(0, 1)
    ax_x.set(
        xlabel=rf"$\Delta\mu=\mu_{{\rm {B}}}-\mu_{{\rm {A}}}$ (eV)",
        ylabel=rf"$x_{{\rm {B}}}$",
        title="(a) SGC isotherms (open = equilibration gate failed; bars 2σ)",
    )
    ax_phi.set(
        xlabel=r"$\Delta\mu$ (eV)",
        ylabel=r"$\phi-\phi_{\rm ref}$ (meV/atom)",
        title=r"(b) $\phi(\Delta\mu)$, eq. (3); boundary where phases cross",
    )
    ax_pd.set(
        xlabel=rf"$x_{{\rm {B}}}$",
        ylabel="T (K)",
        title="(c) T–x (bars: sampled x; ● confirmed, ○ tentative, ◇ extrapolated)",
    )
    ax_G.set(
        xlabel=rf"$x_{{\rm {B}}}$",
        ylabel=r"$G-G_{\rm chord}$ (meV/atom)",
        title=r"(d) $G=\phi+\Delta\mu\,x$ (non-convex ⇒ gap)",
    )
    ax_chi.set(
        xlabel=rf"$x_{{\rm {B}}}$",
        ylabel=r"$\partial\Delta\mu/\partial x$ (eV)",
        title="(e) Stability (spinodal at 0)",
    )
    ax_chi.set_ylim(bottom=0)
    ax_h.axhline(0, color="0.5", lw=0.8)
    ax_h.set(
        xlabel=r"$\Delta\mu$ (eV)",
        ylabel=r"$x_{\rm high\,branch}-x_{\rm low\,branch}$",
        title="(f) Branch hysteresis over shared Δμ (≠0 ⇒ first order)",
    )
    for ax in axs.flat:
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.title.set_fontsize(9)
    for ax in (ax_phi, ax_G, ax_chi, ax_h):
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False, fontsize=7)
    ax_x.legend(
        handles=[plt.Line2D([], [], color=col[T], lw=2, label=f"{T:g} K") for T in Ts],
        frameon=False,
        fontsize=8,
    )
    if opts.title:
        fig.suptitle(opts.title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "input",
        help="tidy CSV, or directory of *.equilibration.json (with --format nvalchemi-json)",
    )
    ap.add_argument(
        "--format", choices=["csv", "nvalchemi-json", "auto"], default="auto"
    )
    ap.add_argument(
        "--species",
        nargs=2,
        default=None,
        metavar=("A", "B"),
        help="species names; x is the fraction of B and dmu = mu_B - mu_A (default: A B, or JSON key order)",
    )
    ap.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=None,
        help="analyse only these T (K)",
    )
    ap.add_argument("--out", default="sgc_phase_boundary_out")
    ap.add_argument(
        "--anchor",
        choices=["auto", "energy", "none"],
        default="auto",
        help="how to fix phi's constant when two distinct phases are found and no --pure-free-energies: "
        "'auto'/'energy' extrapolate E to the pure ends (valid only for rigid-lattice SGC, F_pure=E_pure); "
        "'none' leaves phases unanchored",
    )
    ap.add_argument(
        "--pure-free-energies",
        default=None,
        help='JSON {"<T>": {"A": F_A, "B": F_B}} per-atom free energies of the pure end members (eV)',
    )
    ap.add_argument(
        "--drop-unresolved",
        action="store_true",
        help="discard runs whose equilibration gate failed",
    )
    ap.add_argument(
        "--no-drift-inflation",
        action="store_true",
        help="do not inflate x_se by the window-to-window drift of unresolved runs",
    )
    ap.add_argument(
        "--min-jump",
        type=float,
        default=0.03,
        help="smallest x discontinuity treated as first order",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="false-positive rate of the jump test, eq. (15)",
    )
    ap.add_argument(
        "--max-tail-x",
        type=float,
        default=0.1,
        help="warn if the dilute-tail anchor starts beyond this",
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=300,
        help="resamples for boundary uncertainties (0 = off)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default=None)
    opts = ap.parse_args(argv)
    opts.z = math.sqrt(2) * _erfinv(1 - opts.alpha)

    fmt = opts.format
    if fmt == "auto":
        fmt = "nvalchemi-json" if os.path.isdir(opts.input) else "csv"
    rows = (
        load_nvalchemi_json(opts.input, opts.species)
        if fmt == "nvalchemi-json"
        else load_csv(opts.input)
    )
    if opts.species is None:
        opts.species = ["A", "B"]
    for r in rows:
        se = r["x_se"] if r["x_se"] is not None else 0.005
        if not opts.no_drift_inflation and not r["resolved"]:
            se = math.hypot(se, r["x_drift"] or 0.0)
        r["se_eff"] = max(se, 1e-4)
    if opts.drop_unresolved:
        rows = [r for r in rows if r["resolved"]]
    if not rows:
        sys.exit(
            "no SGC rows to analyse (empty input, or everything removed by --drop-unresolved)"
        )
    Ts = sorted({r["T"] for r in rows})
    if opts.temperatures:
        Ts = [T for T in Ts if any(abs(T - t) < 1e-6 for t in opts.temperatures)]
    if not Ts:
        sys.exit(
            f"none of --temperatures {opts.temperatures} found; available: {sorted({r['T'] for r in rows})}"
        )
    pure_all = {}
    if opts.pure_free_energies:
        pure_all = {
            float(k): v for k, v in json.load(open(opts.pure_free_energies)).items()
        }

    os.makedirs(opts.out, exist_ok=True)
    rng = np.random.default_rng(opts.seed)
    results, boots = [], {}
    for T in Ts:
        rT = [r for r in rows if r["T"] == T]
        pure = pure_all.get(T)
        res = analyse_isotherm(T, rT, opts, pure)
        res["rows"] = rT
        # common reference line/chord for plotting
        allmu = np.concatenate([p.s.mu for p in res["phases"]])
        allphi = np.concatenate([p.phi for p in res["phases"]])
        allx = np.concatenate([p.s.x for p in res["phases"]])
        allG = np.concatenate([g_of_x(p) for p in res["phases"]])
        i0, i1 = np.argmin(allmu), np.argmax(allmu)
        res["phi_ref"] = (
            lambda m, a=allmu[i0], b=allmu[i1], pa=allphi[i0], pb=allphi[i1]: (
                pa + (pb - pa) * (m - a) / (b - a)
            )
        )
        j0, j1 = np.argmin(allx), np.argmax(allx)
        res["G_ref"] = lambda x, a=allx[j0], b=allx[j1], ga=allG[j0], gb=allG[j1]: (
            ga + (gb - ga) * (x - a) / (b - a)
        )
        res["hull_excess_meV"] = (
            1e3 * hull_excess(allx, allG)[0] if res["status"] != "unanchored" else None
        )
        if res["boundary"] and opts.bootstrap:
            samp = []
            for _ in range(opts.bootstrap):
                rb = analyse_isotherm(T, rT, opts, pure, rng=rng, perturb=True)
                if rb["boundary"]:
                    samp.append(
                        [rb["boundary"][k] for k in ("mu_coex", "x_alpha", "x_gamma")]
                    )
            if samp:
                samp = np.array(samp)
                boots[T] = {
                    k: tuple(np.percentile(samp[:, i], [16, 50, 84]))
                    for i, k in enumerate(("mu_coex", "x_alpha", "x_gamma"))
                }
                boots[T]["fraction_with_boundary"] = len(samp) / opts.bootstrap
        # per-T table
        with open(os.path.join(opts.out, f"phi_G_T{T:g}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["phase", "mu", "x", "x_se_eff", "phi", "G", "E", "resolved"])
            for p in res["phases"]:
                G = g_of_x(p)
                for k in range(len(p.s.mu)):
                    w.writerow(
                        [
                            p.name,
                            p.s.mu[k],
                            p.s.x[k],
                            p.s.se[k],
                            p.phi[k],
                            G[k],
                            "" if p.s.E is None else p.s.E[k],
                            bool(p.s.resolved[k]),
                        ]
                    )
        results.append(res)

    checks = clausius_clapeyron_check(results)
    make_figure(results, boots, opts, os.path.join(opts.out, "phase_boundary.png"))

    summary = dict(
        species=dict(
            A=opts.species[0],
            B=opts.species[1],
            x="fraction of B",
            dmu="mu_B - mu_A (eV)",
        ),
        input=str(opts.input),
        options={k: v for k, v in vars(opts).items() if k not in ("input",)},
        temperatures={},
        eq29_checks=checks,
    )
    for r in results:
        summary["temperatures"][f"{r['T']:g}"] = dict(
            status=r["status"],
            boundary=r["boundary"],
            bootstrap_16_50_84=boots.get(r["T"]),
            phases=[
                dict(
                    name=p.name,
                    anchor=p.anchor,
                    notes=p.notes,
                    n=len(p.s.mu),
                    mu_range=[float(p.s.mu.min()), float(p.s.mu.max())],
                    x_range=[float(p.s.x.min()), float(p.s.x.max())],
                )
                for p in r["phases"]
            ],
            G_above_convex_hull_meV=r["hull_excess_meV"],
            **{k: v for k, v in r["diag"].items() if k != "T"},
        )
    json.dump(
        summary,
        open(os.path.join(opts.out, "summary.json"), "w"),
        indent=2,
        default=float,
    )
    write_report(summary, os.path.join(opts.out, "report.md"))
    print(open(os.path.join(opts.out, "report.md")).read())


def _erfinv(y):
    # Giles (2010) single-precision approximation refined by two Newton steps
    w = -math.log((1.0 - y) * (1.0 + y))
    if w < 5:
        w -= 2.5
        p = 2.81022636e-08
        for c in (
            3.43273939e-07,
            -3.5233877e-06,
            -4.39150654e-06,
            0.00021858087,
            -0.00125372503,
            -0.00417768164,
            0.246640727,
            1.50140941,
        ):
            p = c + p * w
    else:
        w = math.sqrt(w) - 3
        p = -0.000200214257
        for c in (
            0.000100950558,
            0.00134934322,
            -0.00367342844,
            0.00573950773,
            -0.0076224613,
            0.00943887047,
            1.00167406,
            2.83297682,
        ):
            p = c + p * w
    x = p * y
    for _ in range(2):
        x -= (math.erf(x) - y) / (2 / math.sqrt(math.pi) * math.exp(-x * x))
    return x


def write_report(s, path):
    A, B = s["species"]["A"], s["species"]["B"]
    L = [
        f"# SGC phase-boundary analysis ({A}–{B}, x = x_{B}, Δμ = μ_{B} − μ_{A})",
        "",
        "| T (K) | status | Δμ_coex (eV) | x_α | x_γ | G above hull (meV) | jumps | hysteresis | anchors |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for T, t in s["temperatures"].items():
        b, bs = t["boundary"], t["bootstrap_16_50_84"]

        def fmt(k):
            if not b:
                return "–"
            if bs and k in bs:
                lo, mid, hi = bs[k]
                return f"{b[k]:.4f} (+{hi - mid:.4f}/−{mid - lo:.4f})"
            return f"{b[k]:.4f}"

        hx = t["G_above_convex_hull_meV"]
        L.append(
            f"| {T} | {t['status']} | {fmt('mu_coex')} | {fmt('x_alpha')} | {fmt('x_gamma')} | "
            f"{'–' if hx is None else f'{hx:.3f}'} | {len(t['jumps'])} | "
            f"{'yes' if t['hysteresis_significant'] else 'no'} ({len(t['hysteresis'])} shared Δμ) | "
            f"{'; '.join(p['anchor'] for p in t['phases'])} |"
        )
    L += ["", "## Diagnostics", ""]
    for T, t in s["temperatures"].items():
        L.append(f"**{T} K**")
        for bname, bd in t["branches"].items():
            L.append(
                f"- branch `{bname}`: {bd['n']} runs, Δμ {bd['mu_range'][0]:.4f}…{bd['mu_range'][1]:.4f}, "
                f"x {bd['x_range'][0]:.3f}…{bd['x_range'][1]:.3f}, unresolved {bd['n_unresolved']}"
            )
        for j in t["jumps"]:
            L.append(
                f"- jump on `{j['branch']}` between Δμ {j['mu_before']:.4f} → {j['mu_after']:.4f}: "
                f"x {j['x_before']:.3f} → {j['x_after']:.3f} (predicted {j['predicted']:.3f} ± {j['threshold']:.3f})"
            )
        for p in t.get("transforming", []):
            L.append(
                f"- `{p['branch']}` point at Δμ {p['mu']:.4f} (x {p['x']:.3f}) caught mid-transformation; "
                f"excluded from both phases"
            )
        b_ = t.get("bracket")
        if b_:
            if "estimate" in b_:
                L.append(
                    f"- hysteresis bracket (anchor-free): Δμ_coex = {b_['estimate']:.4f} ± {b_['half_width']:.4f} eV "
                    f"[{b_['basis']}; onset {b_['lower_onset']:.4f}…{b_['upper_onset']:.4f}, wide {b_['lower_wide']:.4f}…{b_['upper_wide']:.4f}]"
                )
            else:
                L.append(f"- one-sided hysteresis bound only: {b_}")
        for j in t.get("jump_candidates", []):
            L.append(
                f"- candidate jump on `{j['branch']}` Δμ {j['mu_before']:.4f} → {j['mu_after']:.4f} "
                f"(x {j['x_before']:.3f} → {j['x_after']:.3f}): {j['verdict']}"
            )
        for p in t["phases"]:
            for n in p["notes"]:
                L.append(f"- {p['name']}: {n}")
        for w in t["warnings"]:
            L.append(f"- ⚠ {w}")
        L.append("")
    if s["eq29_checks"]:
        L += [
            "## Eq. (29) consistency between temperatures",
            "",
            "| from | to | Δμ_coex predicted | measured | residual |",
            "|---|---|---|---|---|",
        ]
        for c in s["eq29_checks"]:
            L.append(
                f"| {c['T_from']:g} | {c['T_to']:g} | {c['mu_coex_predicted']:.4f} | {c['mu_coex_measured']:.4f} | {c['residual']:+.4f} |"
            )
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
