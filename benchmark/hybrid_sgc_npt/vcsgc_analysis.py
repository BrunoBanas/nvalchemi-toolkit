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
"""Coexistence from VC-SGC(-NPT) walkers: dmu(c), g(c), common tangent (numpy/matplotlib only).

Per walker (``run_vcsgc_scan.py`` ``*.series.json``), over the production blocks:
  c_bar, its SE (batch means), dmu = dmu_ref + 2 kappa (c0 - c_bar) and SE = 2 kappa SE(c_bar),
  std(c) against the unimodal estimate sqrt(kT / (2 kappa N)), the bimodality coefficient, and
  the last-two-window drift gate.

Then g(c) per atom (Gibbs energy of the fixed-composition NPT system, configurational) from
dg/dc = dmu(c). dmu is split into its ideal part dmu_ref + kT ln[c/(1-c)], integrated
analytically, and the excess w(c), interpolated linearly in c between walkers and held constant
beyond the outermost ones (Henry's law). With g(0) = 0:

  * dF_pure = g(1) = g_Pt - g_Au: directly comparable with pure_free_energy_analysis.py's dG;
  * the lower convex hull of g(c) gives the common tangent: dmu_coex and both compositions.
    (Equivalent to the equal-area rule on dmu(c), and correct however the finite cell
    behaves inside the gap, provided each walker is equilibrated.)

Errors by parametric bootstrap over the walkers' dmu SEs. Also written:
``pure_free_energies_vcsgc.json`` ({"T": {"A": 0, "B": dF_pure}}) for
sgc_phase_boundary.py --pure-free-energies (only F_B - F_A matters there).

    python vcsgc_analysis.py <vcsgc root> --out <dir> [--sgc-dir <scan atoms500 dir>] [--discard 0.33]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

KB_EV = 8.617333262e-5
WINDOW = 25


def batch_means_se(x: np.ndarray, n_batches: int = 5) -> float:
    n = len(x) // n_batches
    if n < 2:
        return float("nan")
    means = x[: n * n_batches].reshape(n_batches, n).mean(axis=1)
    return float(means.std(ddof=1) / math.sqrt(n_batches))


def bimodality(x: np.ndarray) -> float:
    """Sarle's coefficient; > 5/9 ~ bimodal or strongly skewed."""
    n = len(x)
    if n < 4 or x.std() == 0:
        return float("nan")
    z = (x - x.mean()) / x.std()
    g, k = (z**3).mean(), (z**4).mean() - 3
    return float((g**2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def passes_run_start(gate: list[dict]) -> int:
    """Block count at the first check of the final unbroken run of passing gate checks."""
    k = len(gate)
    while k > 0 and gate[k - 1]["passed"]:
        k -= 1
    return gate[k]["blocks"] if k < len(gate) else gate[-1]["blocks"]


def walker_table(paths: list[Path], discard: float) -> list[dict]:
    rows = []
    for p in paths:
        s = json.loads(p.read_text())
        c = np.array(s["c"])
        if len(c) < 2 * WINDOW:
            continue
        T, kappa, n = s["temperature_K"], s["kappa"], s["n_atoms"]
        # Production: from the start of the windows the live gate first found stationary (the
        # driver stopped the walker after enough consecutive passes); otherwise drop --discard.
        passes = [g for g in s.get("gate", []) if g["passed"]]
        if s.get("stopped") == "equilibrated" and passes:
            start = max(passes_run_start(s["gate"]) - 2 * WINDOW, 0)
        else:
            start = int(discard * len(c))
        prod = c[start:]
        u = np.array(s["u"])[start:]
        v = np.array(s["v"])[start:]
        cbar, se = float(prod.mean()), batch_means_se(prod)
        last, prev = c[-WINDOW:], c[-2 * WINDOW : -WINDOW]
        drift = float(last.mean() - prev.mean())
        comb = math.hypot(last.std() / math.sqrt(WINDOW), prev.std() / math.sqrt(WINDOW))
        stopped = s.get("stopped")
        rows.append(
            dict(
                run_id=s["run_id"], T=T, c0=s["c0"], kappa=kappa, init=s["init"], n_atoms=n,
                blocks=len(c), production_blocks=len(prod), stopped=stopped,
                dmu_ref=s["delta_mu_ref_eV"], c_bar=cbar, c_se=se,
                dmu=s["delta_mu_ref_eV"] + 2 * kappa * (s["c0"] - cbar),
                dmu_excess=2 * kappa * (s["c0"] - cbar), dmu_se=2 * kappa * se,
                c_std=float(prod.std()), c_std_unimodal=math.sqrt(KB_EV * T / (2 * kappa * n)),
                bimodality=bimodality(prod), drift=drift,
                resolved=(stopped == "equilibrated") if stopped else (bool(abs(drift) < 2 * comb) if comb > 0 else True),
                u_mean=float(u.mean()), v_mean=float(v.mean()),
                acceptance=s["acceptance"][-1]["acceptance"] if s["acceptance"] else None,
                series_c=c.tolist(),
            )
        )
    return sorted(rows, key=lambda r: r["c_bar"])


def combine_same_c0(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One (c, dmu_excess, se) per c0: inverse-variance mean over starting states."""
    by = {}
    for r in rows:
        by.setdefault(round(r["c0"], 4), []).append(r)
    out = []
    for group in by.values():
        w = np.array([1 / max(r["dmu_se"], 1e-5) ** 2 for r in group])
        c = float(np.average([r["c_bar"] for r in group], weights=w))
        m = float(np.average([r["dmu_excess"] for r in group], weights=w))
        spread = np.ptp([r["dmu_excess"] for r in group]) / 2 if len(group) > 1 else 0.0
        out.append((c, m, math.hypot(1 / math.sqrt(w.sum()), spread)))
    out.sort()
    return tuple(np.array(v) for v in zip(*out))


def g_curve(c_pts, w_pts, ref, kt, grid):
    """g(c) - g(0) per atom on ``grid`` (which spans [0, 1])."""
    w = np.interp(grid, c_pts, w_pts)  # constant beyond the ends
    with np.errstate(divide="ignore", invalid="ignore"):
        ideal = ref * grid + kt * np.where(grid > 0, grid * np.log(grid), 0.0) + kt * np.where(
            grid < 1, (1 - grid) * np.log(1 - grid), 0.0
        )
    excess = np.concatenate([[0.0], np.cumsum(0.5 * np.diff(grid) * (w[1:] + w[:-1]))])
    return ideal + excess


def common_tangent(grid, g):
    """Widest edge of the lower convex hull: (slope, c_left, c_right) or None."""
    hull = []
    for i in range(len(grid)):
        while len(hull) >= 2:
            a, b = hull[-2], hull[-1]
            if (g[b] - g[a]) * (grid[i] - grid[a]) >= (g[i] - g[a]) * (grid[b] - grid[a]):
                hull.pop()
            else:
                break
        hull.append(i)
    widths = [(grid[hull[k + 1]] - grid[hull[k]], k) for k in range(len(hull) - 1)]
    width, k = max(widths)
    if width < 5 * (grid[1] - grid[0]):
        return None
    i, j = hull[k], hull[k + 1]
    return float((g[j] - g[i]) / (grid[j] - grid[i])), float(grid[i]), float(grid[j])


def analyse(c, w, se, ref, kt, rng, n_boot):
    grid = np.concatenate([np.linspace(0, 0.002, 41)[:-1], np.linspace(0.002, 0.998, 4000), 1 - np.linspace(0, 0.002, 41)[::-1][1:]])
    grid = np.unique(np.concatenate([grid, [1.0]]))

    def one(wv):
        g = g_curve(c, wv, ref, kt, grid)
        ct = common_tangent(grid, g)
        return g, float(g[-1]), ct

    g, dF, ct = one(w)
    boots = [one(w + rng.normal(0, se)) for _ in range(n_boot)]
    dFs = np.array([b[1] for b in boots])
    cts = np.array([b[2] for b in boots if b[2] is not None])

    def pct(a):
        return [float(v) for v in np.percentile(a, [16, 50, 84])] if len(a) else None

    return dict(
        grid=grid, g=g, dF_pure=dF, dF_pure_boot=pct(dFs),
        tangent=ct, tangent_boot=dict(mu=pct(cts[:, 0]) if len(cts) else None,
                                      c_alpha=pct(cts[:, 1]) if len(cts) else None,
                                      c_gamma=pct(cts[:, 2]) if len(cts) else None,
                                      fraction_with_gap=len(cts) / n_boot),
    )


def load_sgc(directory: Path) -> list[tuple[float, float, bool]]:
    pts = []
    for f in glob.glob(str(directory / "*.equilibration.json")):
        d = json.loads(Path(f).read_text())
        pts.append((d["chemical_potentials_ev"]["Pt"], d["composition_gate"]["mean_last_window"], bool(d["resolved"])))
    return pts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--discard", type=float, default=1 / 3, help="fraction of each walker's blocks discarded")
    ap.add_argument("--sgc-dir", type=Path, help="SGC scan *.equilibration.json directory, overlaid for comparison")
    ap.add_argument("--exclude-init", nargs="*", default=[], help="starting states to leave out (e.g. random)")
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = walker_table(sorted(args.root.glob("*.series.json")), args.discard)
    if not rows:
        raise SystemExit(f"no *.series.json with >= {2 * WINDOW} blocks under {args.root}")
    Ts, refs, kappas = {r["T"] for r in rows}, {r["dmu_ref"] for r in rows}, {r["kappa"] for r in rows}
    if len(Ts) != 1 or len(refs) != 1:
        raise SystemExit("walkers mix temperatures or dmu_ref; analyse one (T, dmu_ref) set at a time")
    T, ref = Ts.pop(), refs.pop()
    kt = KB_EV * T
    used = [r for r in rows if r["init"] not in args.exclude_init]
    c, m, se = combine_same_c0(used)
    w = m - kt * np.log(c / (1 - c))  # excess beyond the ideal-solution slope
    res = analyse(c, w, se, ref, kt, np.random.default_rng(args.seed), args.bootstrap)

    flags = []
    for r in rows:
        if r["stopped"] == "cap":
            flags.append(f"{r['run_id']}: hit its block cap without passing the gate -- extend it (raise n_blocks)")
        elif not r["resolved"]:
            flags.append(f"{r['run_id']}: drifting (last-window change {r['drift']:+.4f})")
        if r["c_std"] > 2 * r["c_std_unimodal"] or (r["bimodality"] == r["bimodality"] and r["bimodality"] > 0.555):
            flags.append(f"{r['run_id']}: c distribution broad/bimodal (std {r['c_std']:.4f} vs {r['c_std_unimodal']:.4f}, "
                         f"BC {r['bimodality']:.2f}): kappa may be too small here")
    inits = {}
    for r in rows:
        inits.setdefault(round(r["c0"], 4), {})[r["init"]] = r["dmu_excess"]
    hysteresis = {k: 1e3 * (v["slab"] - v["random"]) for k, v in inits.items() if {"slab", "random"} <= set(v)}

    summary = dict(
        T=T, delta_mu_ref=ref, kappas=sorted(kappas), n_walkers=len(rows),
        dF_pure=res["dF_pure"], dF_pure_boot_16_50_84=res["dF_pure_boot"],
        dF_pure_excess=res["dF_pure"] - ref,
        common_tangent=None if res["tangent"] is None else dict(
            dmu_coex=res["tangent"][0], dmu_excess_coex=res["tangent"][0] - ref,
            c_alpha=res["tangent"][1], c_gamma=res["tangent"][2]),
        common_tangent_boot_16_50_84=res["tangent_boot"],
        slab_minus_random_dmu_meV=hysteresis, flags=flags,
        walkers=[{k: v for k, v in r.items() if k != "series_c"} for r in rows],
    )
    (args.out / "vcsgc_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out / "pure_free_energies_vcsgc.json").write_text(json.dumps({f"{T:g}": {"A": 0.0, "B": res["dF_pure"]}}, indent=2) + "\n")

    print(f"T={T:g} K  dmu_ref={ref:.5f} eV  {len(rows)} walkers (kappa {sorted(kappas)})")
    print(" c0     init    blocks(prod)  c_bar    +-      dmu_excess(meV) +-    std(c)/unimodal  BC    resolved  acc")
    for r in rows:
        print(f" {r['c0']:.3f}  {r['init']:10s}  {r['blocks']:4d}({r['production_blocks']:3d})  {r['c_bar']:.4f} {r['c_se']:.4f}  {1e3 * r['dmu_excess']:+8.2f} {1e3 * r['dmu_se']:5.2f}"
              f"   {r['c_std'] / r['c_std_unimodal']:5.2f}          {r['bimodality']:.2f}  {str(r['resolved']):5s}  {r['acceptance']}")
    b = res["dF_pure_boot"]
    print(f"\ng_Pt - g_Au = {res['dF_pure']:.5f} eV  (excess over dmu_ref {1e3 * (res['dF_pure'] - ref):+.2f} meV; "
          f"bootstrap 16/50/84 {[round(1e3 * (v - ref), 2) for v in b]} meV)")
    if res["tangent"]:
        mu, ca, cg = res["tangent"]
        tb = res["tangent_boot"]
        print(f"common tangent: dmu_coex = {mu:.5f} eV (excess {1e3 * (mu - ref):+.2f} meV; boot {[round(1e3 * (v - ref), 2) for v in tb['mu']]}), "
              f"x_alpha = {ca:.4f} {tb['c_alpha']}, x_gamma = {cg:.4f} {tb['c_gamma']}; gap in {100 * tb['fraction_with_gap']:.0f}% of resamples")
    else:
        print("no common tangent: g(c) is convex (no miscibility gap resolved)")
    if hysteresis:
        print("slab - random dmu at the same c0 (meV):", {k: round(v, 2) for k, v in hysteresis.items()})
    for f in flags:
        print("FLAG:", f)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    cols = {"slab": "#2a78d6", "random": "#eb6834"}
    for r in rows:
        ax[0].errorbar(r["c_bar"], 1e3 * r["dmu_excess"], 2e3 * r["dmu_se"], fmt="o" if r["resolved"] else "o",
                       mfc=cols.get(r["init"], "k") if r["resolved"] else "white", color=cols.get(r["init"], "k"))
    cc = np.linspace(0.005, 0.995, 400)
    ax[0].plot(cc, 1e3 * (np.interp(cc, c, w) + kt * np.log(cc / (1 - cc))), color="0.4", lw=1, label="interpolation")
    if args.sgc_dir:
        sg = load_sgc(args.sgc_dir)
        ax[0].scatter([x for _, x, _ in sg], [1e3 * (mu - ref) for mu, _, _ in sg], marker="s", s=14, color="0.6",
                      label="SGC scan", zorder=1)
    if res["tangent"]:
        mu, ca, cg = res["tangent"]
        ax[0].hlines(1e3 * (mu - ref), ca, cg, color="#c0392b", lw=1.5, label="common tangent")
    ax[0].set(xlabel=r"$x_{\rm Pt}$", ylabel=r"$\Delta\mu-\Delta\mu_{\rm ref}$ (meV)", ylim=(-250, 250),
              title="(a) VC-SGC dmu(c) (blue slab, orange random; open = drifting)")
    ax[0].legend(frameon=False, fontsize=8)
    grid, g = res["grid"], res["g"]
    chord = g[0] + (g[-1] - g[0]) * grid
    ax[1].plot(grid, 1e3 * (g - chord), color="#2a78d6")
    if res["tangent"]:
        mu, ca, cg = res["tangent"]
        ga = np.interp(ca, grid, g)
        ax[1].plot([ca, cg], [1e3 * (ga - np.interp(ca, grid, chord)), 1e3 * (ga + mu * (cg - ca) - np.interp(cg, grid, chord))],
                   color="#c0392b")
    ax[1].set(xlabel=r"$x_{\rm Pt}$", ylabel="g - chord (meV/atom)", title="(b) g(c) and common tangent")
    for r in rows:
        ax[2].plot(r["series_c"], lw=0.7, color=cols.get(r["init"], "k"))
    ax[2].set(xlabel="block", ylabel=r"$x_{\rm Pt}$", title="(c) composition per block")
    for a in ax:
        a.grid(alpha=0.25, lw=0.5)
        a.spines[["top", "right"]].set_visible(False)
        a.title.set_fontsize(9)
    fig.tight_layout()
    fig.savefig(args.out / "vcsgc_analysis.png", dpi=150)


if __name__ == "__main__":
    main()
