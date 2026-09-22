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
"""Bracket UMA melting points from uma_melting_coexistence.py runs.

For each element, the solid-fraction growth rate d f_s/dt is positive below
T_m (crystal grows) and negative above it. It is fitted over the part of the
production run where an interface still exists (0.05 < f_s < 0.95), capped at
``--fit-ps``. T_m is bracketed by the highest temperature that grows and the
lowest that shrinks, and estimated by linear interpolation of the rate to zero.

Runs whose pure crystal already melted during the solid-equilibration stage
(solid-like < 0.5, i.e. above the superheating limit) or that start production
without a crystalline slab (f_s < 0.15) carry no interface information: they are
excluded from the bracket and reported as the superheating limit instead. A run
that keeps its interface for >= 80 % of the requested production with
|delta f_s| < 0.15 is flagged "near T_m".
Needs only numpy/matplotlib, so it runs on a laptop after copying the output back.

    python uma_melting_analysis.py <out-dir> [--fit-ps 10]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load(root: Path) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = {}
    for summary in sorted(root.glob("*/T*/summary.json")):
        s = json.loads(summary.read_text())
        with (summary.parent / "series.csv").open() as fh:
            rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(fh)]
        s["series"] = rows
        runs.setdefault(s["element"], []).append(s)
    for v in runs.values():
        v.sort(key=lambda s: s["temperature_k"])
    return runs


def rate(series: list[dict], fit_ps: float) -> float:
    t = np.array([r["time_ps"] for r in series]); f = np.array([r["solid_fraction"] for r in series])
    outside = np.nonzero((f <= 0.05) | (f >= 0.95))[0]
    t_end = t[outside[0]] if len(outside) else t.max()
    keep = t <= min(fit_ps, t_end)
    return float(np.polyfit(t[keep], f[keep], 1)[0]) if keep.sum() >= 3 else float("nan")


def has_interface(s: dict) -> bool:
    return s["solid_eq_solid_fraction"] >= 0.5 and s["solid_fraction_initial"] >= 0.15


def near_tm(s: dict, production_ps: float) -> bool:
    return (s["production_ps_run"] >= 0.8 * production_ps
            and abs(s["solid_fraction_final"] - s["solid_fraction_initial"]) < 0.15)


def bracket(Ts: np.ndarray, rates: np.ndarray) -> dict:
    grow, shrink = Ts[rates > 0], Ts[rates < 0]
    out = dict(highest_growing_k=float(grow.max()) if len(grow) else None,
               lowest_shrinking_k=float(shrink.min()) if len(shrink) else None, tm_estimate_k=None)
    for i in range(len(Ts) - 1):
        if rates[i] > 0 >= rates[i + 1]:
            out["tm_estimate_k"] = float(Ts[i] + rates[i] * (Ts[i + 1] - Ts[i]) / (rates[i] - rates[i + 1]))
            break
    if out["highest_growing_k"] and out["lowest_shrinking_k"] and out["highest_growing_k"] > out["lowest_shrinking_k"]:
        out["warning"] = "non-monotonic rates (noise or unequilibrated runs); extend production or add replicas"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("--fit-ps", type=float, default=50.0)
    ap.add_argument("--production-ps", type=float, default=50.0, help="requested production length of the runs")
    args = ap.parse_args()
    runs = load(args.root)
    if not runs:
        raise SystemExit(f"no */T*/summary.json under {args.root}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = {}
    fig, axs = plt.subplots(len(runs), 2, figsize=(11, 3.8 * len(runs)), squeeze=False)
    for row, (el, rs) in enumerate(sorted(runs.items())):
        all_rates = np.array([rate(s["series"], args.fit_ps) for s in rs])
        valid = np.array([has_interface(s) for s in rs])
        Ts = np.array([s["temperature_k"] for s in rs])[valid]; rates = all_rates[valid]
        b = bracket(Ts, rates)
        b["experimental_tm_k"] = rs[0]["experimental_tm_k"]
        b["near_tm_k"] = [s["temperature_k"] for s, v in zip(rs, valid) if v and near_tm(s, args.production_ps)]
        survived = [s["temperature_k"] for s in rs if s["solid_eq_solid_fraction"] >= 0.5]
        melted_alone = [s["temperature_k"] for s in rs if s["solid_eq_solid_fraction"] < 0.5]
        b["superheating_limit_bracket_k"] = [max(survived) if survived else None, min(melted_alone) if melted_alone else None]
        b["runs"] = [dict(T=s["temperature_k"], rate_per_ps=r, has_interface=bool(v), near_tm=bool(v and near_tm(s, args.production_ps)),
                          f_initial=s["solid_fraction_initial"], f_final=s["solid_fraction_final"], verdict=s["verdict"],
                          a_solid=s["solid_lattice_a_ang"], solid_eq_fraction=s["solid_eq_solid_fraction"])
                     for s, r, v in zip(rs, all_rates, valid)]
        result[el] = b
        cmap = plt.get_cmap("coolwarm")
        for i, s in enumerate(rs):
            t = [r["time_ps"] for r in s["series"]]; f = [r["solid_fraction"] for r in s["series"]]
            ok = has_interface(s)
            axs[row, 0].plot(t, f, color=cmap(i / max(1, len(rs) - 1)), lw=1.6 if ok else 0.8, ls="-" if ok else ":",
                             label=f"{s['temperature_k']:g} K" + ("" if ok else " (no interface)"))
        axs[row, 0].set(xlabel="production time (ps)", ylabel="solid-like fraction", ylim=(0, 1), title=f"{el}: coexistence runs")
        axs[row, 0].legend(frameon=False, fontsize=7, ncol=2)
        axs[row, 1].axhline(0, color="0.5", lw=0.8)
        axs[row, 1].plot(Ts, rates, "o-", color="#2a78d6", label="runs with an interface")
        if b["superheating_limit_bracket_k"][1] is not None:
            axs[row, 1].axvspan(b["superheating_limit_bracket_k"][1], max(s["temperature_k"] for s in rs),
                                color="0.85", label="pure crystal melts alone (superheating)")
        axs[row, 1].axvline(b["experimental_tm_k"], color="0.3", ls=":", lw=1, label=f"experiment {b['experimental_tm_k']:g} K")
        if b["tm_estimate_k"]:
            axs[row, 1].axvline(b["tm_estimate_k"], color="#eb6834", lw=1.2, label=f"UMA estimate {b['tm_estimate_k']:.0f} K")
        axs[row, 1].set(xlabel="T (K)", ylabel="d f_s/dt (1/ps)", title=f"{el}: interface velocity sign")
        axs[row, 1].legend(frameon=False, fontsize=8)
        for ax in axs[row]:
            ax.grid(alpha=0.25, lw=0.5); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.root / "melting_summary.png", dpi=150)
    (args.root / "melting_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for el, b in result.items():
        tm = f"{b['tm_estimate_k']:.0f}" if b["tm_estimate_k"] else "n/a"
        print(f"{el}: T_m(UMA) ~ {tm} K  bracket (grows <= T < shrinks) [{b['highest_growing_k']}, {b['lowest_shrinking_k']}] K  "
              f"near-T_m runs {b['near_tm_k']}  superheating limit {b['superheating_limit_bracket_k']} K  "
              f"(experiment {b['experimental_tm_k']:g} K){'  WARNING: ' + b['warning'] if 'warning' in b else ''}")
        for r in b["runs"]:
            tag = "near T_m" if r["near_tm"] else ("" if r["has_interface"] else "no interface (excluded)")
            print(f"   {r['T']:6g} K  rate={r['rate_per_ps']:+.4f}/ps  f {r['f_initial']:.2f}->{r['f_final']:.2f}  "
                  f"a={r['a_solid']:.3f} A  {r['verdict']}  {tag}")


if __name__ == "__main__":
    main()
