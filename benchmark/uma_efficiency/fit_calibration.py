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
"""Fit plan_run.py's time and memory coefficients to efficiency-matrix results.

Reads every ``<root>/<run_tag>/<kernel>/<config>_w<width>/run/metrics.json`` written by
run_efficiency_matrix.sh, maps each (kernel, config) to a settings class and memory
family, and fits, per class, with least squares on relative error:

  time per step   t = c1 * (N0 + w * (N * sf + Nw))     [ms; sf only for compiled]
  memory          M = base + a * A + b * A * (w - 1)    [GiB; A = w * N / 500;
                                                         b only for SGC-NPT]

MC time per step = MC block time / round(0.2 N) trials; MD time per step = MD block
time / 50 (hybrid phases use the median block when recorded). Prints the fits with
their worst error, and with ``--write`` updates the matching calibration.json entries
(sources are rewritten; hand-set entries such as checkpointed merged MC are kept).

    python benchmark/uma_efficiency/fit_calibration.py <efficiency_matrix_root> [--write]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# (kernel, config) -> (settings class, memory family); kernels of run_efficiency_matrix.sh.
CLASSES = {
    ("kawasaki", "best"): ("compiled_merged", "kawasaki_mc"),
    ("kawasaki_wide", "best"): ("compiled_merged", "kawasaki_mc"),
    ("kawasaki_2048", "best"): ("compiled_merged", "kawasaki_mc"),
    ("kawasaki", "merge_nocompile"): ("eager_merged", "kawasaki_mc"),
    ("kawasaki_2048", "merge_nocompile"): ("eager_merged", "kawasaki_mc"),
    ("sgc", "best"): ("eager_unmerged", "sgc_mc"),
    ("sgc_wide", "best"): ("eager_unmerged", "sgc_mc"),
    ("sgc_compile", "best"): ("eager_unmerged", "sgc_mc"),
    ("hybrid", "best"): ("eager_unmerged", "sgc_npt"),
    ("sgc_npt_108", "nomerge_nocompile"): ("eager_unmerged", "sgc_npt"),
    ("sgc_npt_256", "nomerge_nocompile"): ("eager_unmerged", "sgc_npt"),
    ("sgc_npt_2048", "nomerge_nocompile"): ("eager_unmerged", "sgc_npt"),
    ("sgc_npt_ckpt", "nomerge_nocompile_ckpt"): ("checkpointed_unmerged", "sgc_npt"),
    ("sgc_npt_2048", "nomerge_nocompile_ckpt"): ("checkpointed_unmerged", "sgc_npt"),
    ("kawasaki_npt", "merge_nocompile"): ("eager_merged", "kawasaki_npt"),
    ("kawasaki_npt_wide", "merge_nocompile"): ("eager_merged", "kawasaki_npt"),
    ("kawasaki_npt_2048", "merge_nocompile"): ("eager_merged", "kawasaki_npt"),
    ("kawasaki_npt", "nomerge_nocompile"): ("eager_unmerged", "kawasaki_npt"),
    ("kawasaki_npt_2048", "merge_nocompile_ckpt"): (
        "checkpointed_merged",
        "kawasaki_npt",
    ),
    ("npt", "merge_nocompile"): ("eager_merged", "md"),
    ("npt_wide", "merge_nocompile"): ("eager_merged", "md"),
    ("npt_2048", "merge_nocompile"): ("eager_merged", "md"),
    ("npt", "nomerge_nocompile"): ("eager_unmerged", "md"),
    ("npt_2048", "nomerge_nocompile"): ("eager_unmerged", "md"),
    ("npt", "merge_nocompile_ckpt"): ("checkpointed_merged", "md"),
    ("npt_2048", "merge_nocompile_ckpt"): ("checkpointed_merged", "md"),
}
# (time table, class) -> classes whose cells feed the fit; checkpointing does not slow
# unmerged energy-only MC, so those cells join the unmerged MC fit.
TIME_GROUPS = {
    ("mc_energy_only", "compiled_merged"): ("compiled_merged",),
    ("mc_energy_only", "eager_merged"): ("eager_merged",),
    ("mc_energy_only", "eager_unmerged"): ("eager_unmerged", "checkpointed_unmerged"),
    ("md_full_outputs", "eager_merged"): ("eager_merged",),
    ("md_full_outputs", "eager_unmerged"): ("eager_unmerged",),
    ("md_full_outputs", "checkpointed_merged"): ("checkpointed_merged",),
    ("md_full_outputs", "checkpointed_unmerged"): ("checkpointed_unmerged",),
}


def size_factor(n: int, cal: dict) -> float:
    """Calibration's large-graph slowdown of compiled per-atom time."""
    s = cal["size_scaling"]
    if n <= s["ramp_from_atoms"]:
        return 1.0
    frac = min(
        1.0, (n - s["ramp_from_atoms"]) / (s["ramp_to_atoms"] - s["ramp_from_atoms"])
    )
    return 1.0 + (s["large_system_factor"] - 1.0) * frac


def collect(root: Path) -> list[dict]:
    """One point per measured (kernel, config, width) cell with a known class."""
    points = []
    for metrics_path in sorted(root.glob("*/*/*_w*/run/metrics.json")):
        kernel = metrics_path.parents[2].name
        config, _, width = metrics_path.parents[1].name.rpartition("_w")
        if (kernel, config) not in CLASSES:
            continue
        cls, family = CLASSES[(kernel, config)]
        m = json.loads(metrics_path.read_text())
        n = m.get("n_atoms") or m.get("n_atoms_per_walker")
        point = {
            "cell": f"{kernel}/{config}_w{width}",
            "cls": cls,
            "family": family,
            "n": n,
            "w": int(width),
            "gib": m["peak_gpu_memory_reserved_GB"],
        }
        phases = m.get("phase_timing")
        if phases:

            def median(key: str) -> float:
                return phases[key].get("median_seconds", phases[key]["mean_seconds"])

            if median("mc") > 0:
                point["mc_ms"] = (
                    1000 * median("mc") / round(m.get("mc_step_fraction", 0.2) * n)
                )
            if median("md") > 0:
                point["md_ms"] = 1000 * median("md") / m["md_steps_per_block"]
        else:
            steps = (
                m.get("total_mc_steps")
                or m["n_blocks_completed"] * m["mc_steps_per_block"]
            )
            point["mc_ms"] = 1000 * m["mc_run_wall_seconds"] / steps
        points.append(point)
    return points


def weighted_lstsq(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares minimizing relative error."""
    w = 1.0 / y
    sol, *_ = np.linalg.lstsq(x * w[:, None], y * w, rcond=None)
    return sol


def fit_time(points: list[dict], cal: dict) -> dict:
    """Fit c1, N0, Nw per (table, class)."""
    fits = {}
    for (table, cls), members in TIME_GROUPS.items():
        key = "mc_ms" if table == "mc_energy_only" else "md_ms"
        sel = [p for p in points if p["cls"] in members and key in p]
        if len(sel) < 2:
            continue
        compiled = cls == "compiled_merged"
        y = np.array([p[key] for p in sel])
        x = np.array(
            [
                [
                    1.0,
                    p["w"] * p["n"] * (size_factor(p["n"], cal) if compiled else 1.0),
                    p["w"],
                ]
                for p in sel
            ]
        )
        if len(sel) < 3:
            x = x[:, :2]
        sol = weighted_lstsq(x, y)
        if len(sol) == 3 and sol[2] < 0:  # a negative per-walker term is not physical
            sol = np.append(weighted_lstsq(x[:, :2], y), 0.0)
        a, b, g = (list(sol) + [0.0])[:3]
        err = 100 * (x[:, : len(sol)] @ sol[: x.shape[1]] / y - 1)
        fits[(table, cls)] = {
            "c1_ms_per_atom": round(float(b), 5),
            "n0_atoms": round(float(a / b)),
            "nw_atoms": round(float(g / b)),
            "cells": len(sel),
            "max_abs_err_pct": round(float(np.max(np.abs(err))), 1),
        }
    return fits


def fit_memory(points: list[dict]) -> dict:
    """Fit base, per-walker and (SGC-NPT) cross terms per (family, class)."""
    fits = {}
    for family, cls in sorted({(p["family"], p["cls"]) for p in points}):
        sel = [p for p in points if p["family"] == family and p["cls"] == cls]
        y = np.array([p["gib"] for p in sel])
        atoms = np.array([p["w"] * p["n"] / 500 for p in sel])
        width = np.array([p["w"] for p in sel])
        cols = [np.ones_like(atoms), atoms]
        if family == "sgc_npt" and len(sel) >= 3:
            cols.append(atoms * (width - 1))
        x = np.stack(cols, 1)
        if len(sel) < x.shape[1]:
            continue
        sol = weighted_lstsq(x, y)
        err = 100 * (x @ sol / y - 1)
        rec = {
            "base_gib": round(float(sol[0]), 2),
            "per_walker_gib": round(float(sol[1]), 3),
        }
        if len(sol) == 3:
            rec["per_walker_cross_gib"] = round(float(sol[2]), 3)
        rec.update(cells=len(sel), max_abs_err_pct=round(float(np.max(np.abs(err))), 1))
        fits[(family, cls)] = rec
    return fits


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "root",
        type=Path,
        help="efficiency-matrix output root (contains <run_tag>/<kernel>/...)",
    )
    ap.add_argument("--calibration", type=Path, default=HERE / "calibration.json")
    ap.add_argument(
        "--write", action="store_true", help="update the calibration file in place"
    )
    args = ap.parse_args()
    cal = json.loads(args.calibration.read_text())
    points = collect(args.root)
    print(f"{len(points)} cells")
    time_fits, mem_fits = fit_time(points, cal), fit_memory(points)
    for (table, cls), f in time_fits.items():
        print(f"time   {table:16s} {cls:22s} {f}")
    for (family, cls), f in mem_fits.items():
        print(f"memory {family:16s} {cls:22s} {f}")
    if not args.write:
        return
    for (table, cls), f in time_fits.items():
        entry = cal["time_model"].setdefault(table, {}).setdefault(cls, {})
        entry.update({k: f[k] for k in ("c1_ms_per_atom", "n0_atoms", "nw_atoms")})
        entry["source"] = (
            f"fit_calibration.py: {f['cells']} cells, max error {f['max_abs_err_pct']}%"
        )
    for (family, cls), f in mem_fits.items():
        entry = cal["memory_model"].setdefault(family, {}).setdefault(cls, {})
        entry.update(
            {
                k: f[k]
                for k in ("base_gib", "per_walker_gib", "per_walker_cross_gib")
                if k in f
            }
        )
        entry["source"] = (
            f"fit_calibration.py: {f['cells']} cells, max error {f['max_abs_err_pct']}%"
        )
    args.calibration.write_text(json.dumps(cal, indent=2) + "\n")
    print(f"updated {args.calibration}")


if __name__ == "__main__":
    main()
