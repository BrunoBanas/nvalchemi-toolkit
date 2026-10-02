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
"""Pure-element Gibbs free energies from pure_free_energy.py output (numpy/matplotlib only).

Per element, classical and configurational (what the SGC transmutation acceptance samples):

    g_harm(T) = e0 + P v0 + (kT / 2N) [sum ln lambda_i - (3N-3) ln(2 pi kT)]
    h_harm(T) = e0 + P v0 + (3N-3)/(2N) kT
    dh(T)     = <U> + P<V> - h_harm(T)              (anharmonic enthalpy, -> 0 as T -> 0)
    g(T)      = g_harm(T) - T * integral_0^T dh(T')/T'^2 dT'        (Gibbs-Helmholtz)

dh(T) is fitted as a T^2 + b T^3 (+ c T^4 with --order 4), weighted by the ladder SEs, so the
integral is analytic: a T + b T^2/2 (+ c T^3/3). The ln(2 pi kT) and length-unit constants are
identical for both elements and cancel in every difference.

Outputs in --out:
  pure_free_energies.json   {"<T>": {"A": g_Au, "B": g_Pt}}: feed to
                            sgc_phase_boundary.py --pure-free-energies (absolute UMA zero, eV/atom)
  pure_free_energy_summary.json, pure_free_energy.png

The number the phase boundary depends on is

    dG_excess(T) = [g_Pt - g_Au] - [<U>_Pt - <U>_Au] = -T dS(Pt - Au) + P dV

i.e. how far the free-energy difference sits from the energy-only delta_mu_ref. In the dilute
limit dmu_excess,coex ~ dG_excess + kT [x_Pt(alpha) - x_Au(gamma)].

    python pure_free_energy_analysis.py <dir with Au.json, Pt.json> --out <dir> \
        [--temperature 700] [--calibration auto_reference_energies.json]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

SPECIES = ("Au", "Pt")  # A, B in the sgc-phase-boundary convention (x = x_Pt)


# ----------------------------------------------------------------------------- force constants
def translation_map(frac: np.ndarray) -> np.ndarray:
    """``m[i, j]``: the atom at ``r_j - (r_i - r_0)`` (mod the cell), for fractional coords."""
    n = len(frac)
    m = np.empty((n, n), dtype=np.int64)
    for i in range(n):
        shifted = frac - (frac[i] - frac[0])
        diff = shifted[:, None, :] - frac[None, :, :]
        diff -= np.round(diff)
        dist = np.abs(diff).max(axis=-1)
        m[i] = dist.argmin(axis=1)
        if dist[np.arange(n), m[i]].max() > 1e-4:
            raise ValueError(
                f"structure is not translation-invariant under the shift taking atom 0 to atom {i}"
            )
    return m


def assemble_force_constants(column: np.ndarray, frac: np.ndarray) -> np.ndarray:
    """Full (3N, 3N) force-constant matrix from atom 0's column.

    ``column[j, b, a] = d^2 U / (d u_{j b} d u_{0 a}) = -dF_{j b} / d u_{0 a}``. Translation
    invariance gives ``Phi[(j, b), (i, a)] = column[m[i, j], b, a]``. The result is symmetrized
    and the acoustic sum rule (each row sums to zero) is imposed on the diagonal blocks.
    """
    n = len(frac)
    m = translation_map(frac)
    phi = np.zeros((n, 3, n, 3))
    for i in range(n):
        phi[:, :, i, :] = column[m[i]]
    phi = phi.reshape(3 * n, 3 * n)
    phi = 0.5 * (phi + phi.T)
    blocks = phi.reshape(n, 3, n, 3)
    for i in range(n):
        off = blocks[i].sum(axis=1) - blocks[i, :, i, :]
        blocks[i, :, i, :] = -off
    phi = blocks.reshape(3 * n, 3 * n)
    return 0.5 * (phi + phi.T)


def harmonic_summary(phi: np.ndarray) -> dict:
    """Eigen-decomposition summary: sum of ln(lambda) over the 3N-3 nonzero modes, and checks."""
    lam = np.linalg.eigvalsh(phi)
    order = np.argsort(np.abs(lam))
    acoustic, rest = lam[order[:3]], np.sort(lam[order[3:]])
    return dict(
        n_modes=int(len(rest)),
        sum_ln_lambda=float(np.sum(np.log(rest))) if np.all(rest > 0) else None,
        mean_ln_lambda=float(np.mean(np.log(rest))) if np.all(rest > 0) else None,
        lambda_min=float(rest[0]),
        lambda_max=float(rest[-1]),
        n_negative=int(np.sum(rest <= 0)),
        acoustic_eigenvalues=[float(v) for v in acoustic],
    )


def fit_lattice(a: np.ndarray, e: np.ndarray) -> tuple[float, float, float]:
    """Cubic fit of e(a): returns (a0, e0, curvature d2e/da2 at a0)."""
    p = np.polyfit(a, e, 3)
    roots = np.roots(np.polyder(p))
    roots = roots[np.isreal(roots)].real
    curv = np.polyval(np.polyder(p, 2), roots)
    a0 = float(roots[curv > 0][np.argmin(np.abs(roots[curv > 0] - a[len(a) // 2]))])
    return a0, float(np.polyval(p, a0)), float(np.polyval(np.polyder(p, 2), a0))


# ----------------------------------------------------------------------------- free energy
def harmonic(state: dict, delta: str | None = None) -> tuple[float, float, str]:
    """(sum ln lambda, its spread over the two deltas, delta used)."""
    deltas = state["hessian"]["deltas"]
    keys = sorted(deltas, key=float)
    use = delta or keys[0]
    vals = [deltas[k]["sum_ln_lambda"] for k in keys]
    if any(v is None for v in vals):
        raise SystemExit(f"{state['meta']['element']}: unstable force constants (negative eigenvalues)")
    return deltas[use]["sum_ln_lambda"], float(np.ptp(vals)), use


def g_harm(T, e0, pv0, sum_ln, n, kb):
    kt = kb * np.asarray(T, float)
    return e0 + pv0 + kt / (2 * n) * (sum_ln - (3 * n - 3) * np.log(2 * np.pi * kt))


def h_harm(T, e0, pv0, n, kb):
    return e0 + pv0 + (3 * n - 3) / (2 * n) * kb * np.asarray(T, float)


def fit_anharmonic(T, dh, se, order):
    """Weighted least squares dh = sum_k c_k T^k, k = 2..order. Returns coeffs, covariance."""
    powers = np.arange(2, order + 1)
    A = np.asarray(T, float)[:, None] ** powers[None, :]
    w = 1.0 / np.asarray(se, float)
    Aw, yw = A * w[:, None], np.asarray(dh, float) * w
    coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    resid = yw - Aw @ coef
    dof = max(len(T) - len(powers), 1)
    scale = max(float(resid @ resid) / dof, 1.0)  # inflate by reduced chi^2 when > 1
    cov = np.linalg.inv(Aw.T @ Aw) * scale
    return powers, coef, cov, float(resid @ resid) / dof


def anharmonic_integral(T, powers, coef, cov):
    """T * integral_0^T dh/T'^2 dT' = sum_k c_k T^k / (k-1), and its SE."""
    T = float(T)
    grad = np.array([T**k / (k - 1) for k in powers])
    return float(grad @ coef), float(math.sqrt(grad @ cov @ grad))


def analyse_element(state: dict, order: int, delta: str | None) -> dict:
    meta, lat = state["meta"], state["lattice"]
    n, kb, p = meta["n_atoms"], meta["kb_ev"], meta["pressure_ev_per_a3"]
    e0, pv0 = lat["e0"], p * lat["v0"]
    sum_ln, sum_ln_spread, used = harmonic(state, delta)
    rows = sorted(state.get("ladder", {}).values(), key=lambda r: r["T"])
    if len(rows) < order:
        raise SystemExit(f"{meta['element']}: only {len(rows)} ladder temperatures")
    T = np.array([r["T"] for r in rows])
    h = np.array([r["u_mean"] + p * r["v_mean"] for r in rows])
    se = np.array([max(math.hypot(r["u_se"], p * r["v_se"]), 1e-5) for r in rows])
    dh = h - h_harm(T, e0, pv0, n, kb)
    powers, coef, cov, chi2 = fit_anharmonic(T, dh, se, order)
    # constant-term check: a nonzero T -> 0 limit means e0 and the ladder disagree (settings?)
    A0 = np.c_[np.ones_like(T), T[:, None] ** powers[None, :]] / se[:, None]
    c_free = np.linalg.lstsq(A0, dh / se, rcond=None)[0]
    return dict(
        element=meta["element"],
        n_atoms=n,
        a0=lat["a0"],
        e0=e0,
        sum_ln_lambda=sum_ln,
        delta_used=used,
        harmonic_delta_spread_per_atom_at_1000K=kb * 1000 / (2 * n) * sum_ln_spread,
        ladder_T=T.tolist(),
        ladder_u=[r["u_mean"] for r in rows],
        ladder_v=[r["v_mean"] for r in rows],
        ladder_resolved=[bool(r["u_gate"].get("resolved")) for r in rows],
        dh=dh.tolist(),
        dh_se=se.tolist(),
        fit_powers=powers.tolist(),
        fit_coef=coef.tolist(),
        fit_cov=cov.tolist(),
        fit_reduced_chi2=chi2,
        dh_T0_limit_if_free=float(c_free[0]),
        _consts=(e0, pv0, sum_ln, n, kb),
    )


def harmonic_ts(el: dict, T: float) -> float:
    """g_harm - h_harm + P v0 = -T s_harm (+ the common constant): the harmonic-only excess."""
    e0, pv0, sum_ln, n, kb = el["_consts"]
    return float(g_harm(T, e0, pv0, sum_ln, n, kb) - h_harm(T, e0, pv0, n, kb))


def g_of_T(el: dict, T: float) -> tuple[float, float]:
    e0, pv0, sum_ln, n, kb = el["_consts"]
    anh, anh_se = anharmonic_integral(T, np.array(el["fit_powers"]), np.array(el["fit_coef"]), np.array(el["fit_cov"]))
    return float(g_harm(T, e0, pv0, sum_ln, n, kb)) - anh, anh_se


def u_at(el: dict, T: float) -> tuple[float, float]:
    """<U>(T) from the ladder (exact point if sampled, else the fitted model)."""
    if T in el["ladder_T"]:
        i = el["ladder_T"].index(T)
        return el["ladder_u"][i], el["dh_se"][i]
    e0, pv0, _, n, kb = el["_consts"]
    dh = sum(c * T**k for k, c in zip(el["fit_powers"], el["fit_coef"]))
    return float(h_harm(T, e0, pv0, n, kb) + dh - pv0), float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--temperature", type=float, nargs="+", default=[700.0])
    ap.add_argument("--order", type=int, default=4, choices=[3, 4, 5], help="highest power in the dh(T) fit")
    ap.add_argument("--delta", default=None, help="finite-difference displacement to use (default: smallest)")
    ap.add_argument("--calibration", type=Path, help="auto_reference_energies.json from the SGC scan, for comparison")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    els = {s: analyse_element(json.loads((args.root / f"{s}.json").read_text()), args.order, args.delta) for s in SPECIES}
    cal = json.loads(args.calibration.read_text())["reference"] if args.calibration else {}

    pure, table = {}, []
    for T in args.temperature:
        (ga, ga_se), (gb, gb_se) = g_of_T(els["Au"], T), g_of_T(els["Pt"], T)
        (ua, ua_se), (ub, ub_se) = u_at(els["Au"], T), u_at(els["Pt"], T)
        d_g, d_u = gb - ga, ub - ua
        row = dict(
            T=T,
            g_Au=ga,
            g_Pt=gb,
            dG=d_g,
            dG_anharmonic_se=math.hypot(ga_se, gb_se),
            dU_ladder=d_u,
            dU_ladder_se=math.hypot(ua_se, ub_se),
            dG_excess=d_g - d_u,
            dG_excess_se=math.sqrt(ga_se**2 + gb_se**2 + ua_se**2 + ub_se**2),
            harmonic_only_dG_excess=harmonic_ts(els["Pt"], T) - harmonic_ts(els["Au"], T),
        )
        row["dS_k_per_atom"] = -(row["dG_excess"]) / (els["Au"]["_consts"][4] * T)
        c = cal.get(f"{T:g}")
        if c:
            row["calibration_delta_mu_ref"] = c["delta_mu_ref_eV"]
            row["dG_minus_calibration_delta_mu_ref"] = d_g - c["delta_mu_ref_eV"]
        pure[f"{T:g}"] = {"A": ga, "B": gb}
        table.append(row)

    (args.out / "pure_free_energies.json").write_text(json.dumps(pure, indent=2) + "\n")
    summary = dict(
        elements={s: {k: v for k, v in e.items() if not k.startswith("_")} for s, e in els.items()},
        temperatures=table,
        convention="g per atom, classical configurational Gibbs energy at the run pressure, eV; "
        "dG_excess = (g_Pt - g_Au) - (<U>_Pt - <U>_Au)",
    )
    (args.out / "pure_free_energy_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    for s, e in els.items():
        print(
            f"{s}: a0={e['a0']:.4f} A  e0={e['e0']:.6f}  mean ln lambda={e['sum_ln_lambda'] / (3 * e['n_atoms'] - 3):.4f}  "
            f"harmonic delta spread {1e3 * e['harmonic_delta_spread_per_atom_at_1000K']:.3f} meV@1000K  "
            f"anharmonic fit chi2/dof={e['fit_reduced_chi2']:.2f}  free T->0 limit {1e3 * e['dh_T0_limit_if_free']:+.2f} meV"
        )
    for r in table:
        print(
            f"T={r['T']:g} K: g_Pt-g_Au = {r['dG']:.5f} eV, <U>_Pt-<U>_Au = {r['dU_ladder']:.5f} eV  ->  "
            f"dG_excess = {1e3 * r['dG_excess']:+.2f} +/- {1e3 * r['dG_excess_se']:.2f} meV "
            f"(harmonic only {1e3 * r['harmonic_only_dG_excess']:+.2f}), dS(Pt-Au) = {r['dS_k_per_atom']:+.3f} k"
            + (
                f"; vs SGC calibration delta_mu_ref {r['calibration_delta_mu_ref']:.5f}: "
                f"g_Pt-g_Au - delta_mu_ref = {1e3 * r['dG_minus_calibration_delta_mu_ref']:+.2f} meV"
                if "calibration_delta_mu_ref" in r
                else ""
            )
        )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    tt = np.linspace(1, max(max(e["ladder_T"]) for e in els.values()), 200)
    for s, col in zip(SPECIES, ("#d4a017", "#5b6b7a")):
        e = els[s]
        ax[0].errorbar(e["ladder_T"], 1e3 * np.array(e["dh"]), 2e3 * np.array(e["dh_se"]), fmt="o", color=col, label=s)
        fit = sum(c * tt**k for k, c in zip(e["fit_powers"], e["fit_coef"]))
        ax[0].plot(tt, 1e3 * fit, color=col, lw=1)
        e0 = e["e0"]
        ax[1].plot(tt, [1e3 * (g_of_T(e, t)[0] - e0) for t in tt], color=col, label=f"{s}: g - e0")
        ax[1].plot(tt, 1e3 * (g_harm(tt, *e["_consts"]) - e0), color=col, ls=":", lw=1)
    d_ex = [1e3 * ((g_of_T(els["Pt"], t)[0] - g_of_T(els["Au"], t)[0]) - (u_at(els["Pt"], t)[0] - u_at(els["Au"], t)[0])) for t in tt]
    ax[2].plot(tt, d_ex, color="#2a78d6")
    for r in table:
        ax[2].errorbar(r["T"], 1e3 * r["dG_excess"], 2e3 * r["dG_excess_se"], fmt="o", color="#eb6834")
    ax[0].set(xlabel="T (K)", ylabel="h - h_harm (meV/atom)", title="(a) anharmonic enthalpy, fit (2σ bars)")
    ax[1].set(xlabel="T (K)", ylabel="meV/atom", title="(b) configurational g(T) (dotted: harmonic)")
    ax[2].set(xlabel="T (K)", ylabel="meV/atom", title=r"(c) $\Delta G_{\rm excess}=\Delta g-\Delta\langle U\rangle$ (Pt$-$Au)")
    for a in ax:
        a.grid(alpha=0.25, lw=0.5)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(frameon=False)
    ax[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.out / "pure_free_energy.png", dpi=150)


if __name__ == "__main__":
    main()
