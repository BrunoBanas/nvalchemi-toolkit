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
"""Bisect the UMA melting point of a pure metal by solid-liquid coexistence, for several cell sizes.

Each coexistence run (uma_melting_coexistence.py, one subprocess per temperature) says on which side
of T_m it is: the solid slab grows or freezes completely (T < T_m), or shrinks or melts completely
(T > T_m). Instead of a fixed temperature grid, this driver bisects:

1. check the bracket ends (a "below" end that is actually above T_m, or vice versa, is moved out by
   half the bracket width, up to 3 times);
2. halve the bracket until it is narrower than ``--resolution`` (25 K);
3. move to the next cell size with a bracket of +-``--next-half-width`` K around the previous
   size's T_m, so a size series costs ~6 runs per size.

A run left undecided at the end of production (solid fraction changed by < 0.02) is taken as
T ~ T_m and ends the bisection for that size. Runs far from T_m stop early (fully melted/frozen),
so most of the cost is the 2-3 runs near T_m.

Sizes are ``NXxNZ``: an NX x NX x NZ conventional-cell slab PER PHASE, i.e. 8 NX^2 NZ atoms in
total (5x5 = 1000, 8x8 = 4096, 10x10 = 8000). Larger cross sections reduce interface-fluctuation
and slab-melting artefacts; the size series shows whether the 1000-atom T_m has converged.

Solid-liquid coexistence line of the alloy (``--delta-mu-ev MU1 MU2 ...``): the same bisection on
T, one delta_mu at a time (hybrid SGC-MD, see uma_melting_coexistence.py), in the first size only.
At fixed delta_mu the solid is stable below T* and the liquid above, so the bisection logic is
unchanged; each point gives T*(delta_mu) and the solidus/liquidus Pt fractions of the runs that
bracket it. List the delta_mu values from the pure-element end inward (Au branch: increasing,
Pt branch: decreasing): each point's bracket is +-``--next-half-width`` around the previous T*.
Output: ``<out>/n<atoms>/dmu<MU>/<El>/T<T>/`` and ``<out>/liquidus_<El>.json``.

Resumable: a temperature whose ``summary.json`` exists is read, not rerun, so a resubmitted job
replays the bisection to where it stopped. Output per size: ``<out>/n<atoms>/<El>/T<T>/`` (the
coexistence script's files) and ``<out>/n<atoms>/bisect_<El>.json``; ``<out>/size_series_<El>.json``
collects all sizes.

    python benchmark/hybrid_sgc_npt/uma_melting_bisect.py --element Au --t-lo 700 --t-hi 1100 \\
        --sizes 5x5 8x8 --out-dir "$RUN_ROOT/uma_melting_size"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MERGED = "compile=false,merge_mole=true,tf32=true,activation_checkpointing=false"
MERGED_CKPT = "compile=false,merge_mole=true,tf32=true,activation_checkpointing=true"


def gpu_memory_gib() -> float:
    """Total memory of the first visible GPU (0 if nvidia-smi is unavailable)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout.split()
        return float(out[0]) / 1024
    except (OSError, subprocess.CalledProcessError, IndexError, ValueError):
        return 0.0


def side(summary: dict) -> int:
    """+1 if the run says T > T_m, -1 if T < T_m, 0 if undecided."""
    v = summary["verdict"]
    if "melted" in v or "shrinking" in v:
        return 1
    if "frozen" in v or "growing" in v:
        return -1
    d = summary["solid_fraction_final"] - summary["solid_fraction_initial"]
    return 0 if abs(d) < 0.02 else (1 if d < 0 else -1)


def main() -> None:
    """Bisect T_m for each requested size in turn."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--element", required=True, choices=["Au", "Pt"])
    ap.add_argument("--t-lo", type=float, required=True, help="initial bracket, expected below T_m (K)")
    ap.add_argument("--t-hi", type=float, required=True, help="initial bracket, expected above T_m (K)")
    ap.add_argument("--sizes", nargs="+", default=["5x5", "8x8"], help="NXxNZ per phase, in run order")
    ap.add_argument("--resolution", type=float, default=25.0, help="stop when the bracket is this narrow (K)")
    ap.add_argument("--next-half-width", type=float, default=150.0,
                    help="bracket half-width around the previous size's T_m (K)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--inference-settings", default=MERGED)
    ap.add_argument("--large-atoms", type=int, default=3000,
                    help="cells with more atoms use activation checkpointing on cards < 60 GiB")
    ap.add_argument("--production-ps", type=float, default=50.0)
    ap.add_argument("--delta-mu-ev", type=float, nargs="*", default=None,
                    help="alloy coexistence points (absolute Pt chemical potential, Au = 0), in trace order")
    ap.add_argument("--mc-step-fraction", type=float, default=0.1)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the first command only")
    args = ap.parse_args()

    el, gib = args.element, gpu_memory_gib()
    lo, hi = args.t_lo, args.t_hi
    series = {}
    alloy = bool(args.delta_mu_ev)
    if alloy and "merge_mole=true" in args.inference_settings:
        args.inference_settings = args.inference_settings.replace("merge_mole=true", "merge_mole=false")
    points = [(args.sizes[0], mu) for mu in args.delta_mu_ev] if alloy else [(s, None) for s in args.sizes]
    for size, mu in points:
        nx, nz = (int(v) for v in size.lower().split("x"))
        n_atoms = 8 * nx * nx * nz
        settings = MERGED_CKPT if (n_atoms > args.large_atoms and gib < 60) else args.inference_settings
        if alloy:
            settings = settings.replace("merge_mole=true", "merge_mole=false")
        root = args.out_dir / f"n{n_atoms}" / ("" if mu is None else f"dmu{mu:+.4f}")
        key = size if mu is None else f"{mu:+.4f}"
        lo, hi = round(lo / 5.0) * 5.0, round(hi / 5.0) * 5.0
        runs: dict[float, dict] = {}

        def run(T: float) -> int:
            T = float(round(T / 5.0) * 5)
            summ = root / el / f"T{T:g}" / "summary.json"
            if not summ.is_file():
                cmd = [sys.executable, str(HERE / "uma_melting_coexistence.py"), "--element", el,
                       "--temperature-k", f"{T:g}", "--out-dir", str(root), "--nx", str(nx), "--nz", str(nz),
                       "--production-ps", f"{args.production_ps:g}", "--inference-settings", settings]
                if mu is not None:
                    cmd += ["--delta-mu-ev", f"{mu:.6f}", "--mc-step-fraction", f"{args.mc_step_fraction:g}"]
                print(f"[bisect] {el} n={n_atoms} T={T:g} K: {' '.join(cmd)}", flush=True)
                if args.dry_run:
                    raise SystemExit(0)
                subprocess.run(cmd, check=True)
            s = json.loads(summ.read_text())
            runs[T] = dict(verdict=s["verdict"], side=side(s), f_initial=s["solid_fraction_initial"],
                           f_final=s["solid_fraction_final"], rate=s["solid_fraction_rate_per_ps_first_10ps"],
                           solid_eq_fraction=s["solid_eq_solid_fraction"], wall_minutes=s["wall_minutes"],
                           x_pt_solid=s.get("x_pt_solid"), x_pt_liquid=s.get("x_pt_liquid"))
            print(f"[bisect] {el} n={n_atoms} T={T:g} K: {s['verdict']} (side {runs[T]['side']:+d}, "
                  f"solid {s['solid_fraction_initial']:.2f} -> {s['solid_fraction_final']:.2f})", flush=True)
            return runs[T]["side"]

        print(f"[bisect] {el} {key} ({n_atoms} atoms), bracket {lo:g}-{hi:g} K, settings {settings!r}, "
              f"GPU {gib:.0f} GiB", flush=True)
        tm = None
        for _ in range(3):  # make sure lo is below and hi above T_m
            s = run(lo)
            if s == 0:
                tm = lo
                break
            if s < 0:
                break
            lo, hi = lo - (hi - lo) / 2, lo
        for _ in range(3):
            if tm is not None:
                break
            s = run(hi)
            if s == 0:
                tm = hi
                break
            if s > 0:
                break
            lo, hi = hi, hi + (hi - lo) / 2
        while tm is None and hi - lo > args.resolution:
            mid = round((lo + hi) / 2 / 5.0) * 5
            if mid in (lo, hi):
                break
            s = run(mid)
            if s == 0:
                tm = mid
            elif s > 0:
                hi = mid
            else:
                lo = mid
        result = dict(element=el, size=size, n_atoms=n_atoms, settings=settings,
                      t_m_k=tm if tm is not None else 0.5 * (lo + hi),
                      t_m_halfwidth_k=0.0 if tm is not None else 0.5 * (hi - lo),
                      bracket_k=[lo, hi], undecided_run_k=tm,
                      superheating_seen_k=sorted(T for T, r in runs.items() if r["solid_eq_fraction"] < 0.5),
                      runs={f"{T:g}": r for T, r in sorted(runs.items())})
        if mu is not None:
            # Solidus/liquidus compositions: mean over the runs at the bracket ends (both still
            # had an interface for part of the run) or the undecided run.
            near = [runs[T] for T in ([tm] if tm is not None else [lo, hi]) if T in runs]
            def _mean(k: str) -> float | None:
                v = [r[k] for r in near if r.get(k) is not None and r[k] == r[k]]
                return sum(v) / len(v) if v else None
            result.update(delta_mu_ev=mu, x_pt_solid=_mean("x_pt_solid"), x_pt_liquid=_mean("x_pt_liquid"))
        (root / f"bisect_{el}.json").write_text(json.dumps(result, indent=2) + "\n")
        series[key] = result
        print(f"[bisect] {el} {key}: T_m = {result['t_m_k']:.0f} +- {result['t_m_halfwidth_k']:.0f} K "
              f"({len(runs)} runs)", flush=True)
        lo, hi = result["t_m_k"] - args.next_half_width, result["t_m_k"] + args.next_half_width

        if alloy:  # rewrite after every point, so a partial trace is readable
            (args.out_dir / f"liquidus_{el}.json").write_text(json.dumps(series, indent=2) + "\n")

    (args.out_dir / (f"liquidus_{el}.json" if alloy else f"size_series_{el}.json")).write_text(json.dumps(series, indent=2) + "\n")
    print("[bisect] " + "; ".join(f"{k}: {v['t_m_k']:.0f}+-{v['t_m_halfwidth_k']:.0f} K ({v['n_atoms']} atoms)"
                                  for k, v in series.items()), flush=True)


if __name__ == "__main__":
    main()
