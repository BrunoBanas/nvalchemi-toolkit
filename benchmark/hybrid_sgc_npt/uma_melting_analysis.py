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

For each element, the solid-fraction growth rate d f_s/dt over the first
``--fit-ps`` of production is positive below T_m (crystal grows) and negative
above it. T_m is bracketed by the highest temperature that grows and the lowest
that shrinks, and estimated by linear interpolation of the rate to zero.
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
    keep = t <= min(fit_ps, t.max())
    return float(np.polyfit(t[keep], f[keep], 1)[0]) if keep.sum() >= 3 else float("nan")


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
    ap.add_argument("--fit-ps", type=float, default=10.0)
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
        Ts = np.array([s["temperature_k"] for s in rs]); rates = np.array([rate(s["series"], args.fit_ps) for s in rs])
        b = bracket(Ts, rates)
        b["experimental_tm_k"] = rs[0]["experimental_tm_k"]
        b["runs"] = [dict(T=s["temperature_k"], rate_per_ps=r, f_initial=s["solid_fraction_initial"],
                          f_final=s["solid_fraction_final"], verdict=s["verdict"], a_solid=s["solid_lattice_a_ang"],
                          solid_eq_fraction=s["solid_eq_solid_fraction"]) for s, r in zip(rs, rates)]
        result[el] = b
        cmap = plt.get_cmap("coolwarm")
        for i, s in enumerate(rs):
            t = [r["time_ps"] for r in s["series"]]; f = [r["solid_fraction"] for r in s["series"]]
            axs[row, 0].plot(t, f, color=cmap(i / max(1, len(rs) - 1)), lw=1.6, label=f"{s['temperature_k']:g} K")
        axs[row, 0].set(xlabel="production time (ps)", ylabel="solid-like fraction", ylim=(0, 1), title=f"{el}: coexistence runs")
        axs[row, 0].legend(frameon=False, fontsize=7, ncol=2)
        axs[row, 1].axhline(0, color="0.5", lw=0.8)
        axs[row, 1].plot(Ts, rates, "o-", color="#2a78d6")
        axs[row, 1].axvline(b["experimental_tm_k"], color="0.3", ls=":", lw=1, label=f"experiment {b['experimental_tm_k']:g} K")
        if b["tm_estimate_k"]:
            axs[row, 1].axvline(b["tm_estimate_k"], color="#eb6834", lw=1.2, label=f"UMA estimate {b['tm_estimate_k']:.0f} K")
        axs[row, 1].set(xlabel="T (K)", ylabel=f"d f_s/dt, first {args.fit_ps:g} ps (1/ps)", title=f"{el}: interface velocity sign")
        axs[row, 1].legend(frameon=False, fontsize=8)
        for ax in axs[row]:
            ax.grid(alpha=0.25, lw=0.5); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.root / "melting_summary.png", dpi=150)
    (args.root / "melting_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for el, b in result.items():
        print(f"{el}: T_m(UMA) ~ {b['tm_estimate_k']} K  bracket [{b['highest_growing_k']}, {b['lowest_shrinking_k']}] K  "
              f"(experiment {b['experimental_tm_k']:g} K){'  WARNING: ' + b['warning'] if 'warning' in b else ''}")
        for r in b["runs"]:
            print(f"   {r['T']:6g} K  rate={r['rate_per_ps']:+.4f}/ps  f {r['f_initial']:.2f}->{r['f_final']:.2f}  "
                  f"a={r['a_solid']:.3f} A  {r['verdict']}")


if __name__ == "__main__":
    main()
