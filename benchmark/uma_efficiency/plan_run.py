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
"""Plan an nvalchemi-toolkit + FairChem UMA run: settings, memory, batch width, wall time.

Uses the measured coefficients in calibration.json (A100/H100 runs, see
benchmark/uma_efficiency/README.md). Two models, both fitted to real runs:

  time   one batched step = c1 * (N0 + width * N_eff)      [ms, reference GPU]
         N0 = fixed per-call overhead in atom-equivalents (what batching amortizes)
  memory peak = base + width * per_walker * (N_eff / 500)  [GiB, reserved]

N_eff = n_atoms * (neighbours per atom / reference neighbours), so a sparser or
denser system than the calibration (fcc Au-Pt, a = 4.00 A) is scaled by its
edge count, which is what UMA's cost tracks.

Examples
--------
  plan_run.py --kind sgc-npt --n-atoms 2048 --walkers 22 --n-blocks 200
  plan_run.py --kind kawasaki-npt --structure np_3nm.xyz --walkers 8 --gpu a100-pcie-40gb
  plan_run.py --kind npt --n-atoms 864 --density 0.085 --json
  plan_run.py --kind kawasaki-npt --structure particle.xyz --walkers 1      # single system

Two regimes rank the settings differently. Throughput (many small, independent
walkers): batch them and pick the settings with the most walker-blocks per second
per GPU at their best width. Single (one system, or too large to batch): width 1,
pick the shortest time per block.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

CAL = json.loads((Path(__file__).resolve().parent / "calibration.json").read_text())
REF_N = CAL["reference"]["reference_n_atoms"]
REF_RHO = CAL["reference"]["reference_density_atoms_per_A3"]
MEASURED_MAX_WIDTH = 4

SPECS = {
    "compiled_merged": "turbo",
    "eager_merged": "compile=false,merge_mole=true,tf32=true,activation_checkpointing=false",
    "eager_unmerged": "compile=false,merge_mole=false,tf32=true,activation_checkpointing=false",
    "checkpointed_merged": "compile=false,merge_mole=true,tf32=true,activation_checkpointing=true",
    "checkpointed_unmerged": "compile=false,merge_mole=false,tf32=true,activation_checkpointing=true",
}

# kind -> (has_mc, has_md, composition_fixed, geometry_fixed, memory family, ensemble)
KINDS = {
    "kawasaki": (True, False, True, True, "kawasaki_mc", None),
    "sgc": (True, False, False, True, "sgc_mc", None),
    "vcsgc": (True, False, False, True, "sgc_mc", None),
    "npt": (False, True, True, False, "md", "npt"),
    "nvt": (False, True, True, False, "md", "nvt"),
    "kawasaki-npt": (True, True, True, False, "kawasaki_npt", "npt"),
    "kawasaki-nvt": (True, True, True, False, "kawasaki_npt", "nvt"),
    "sgc-npt": (True, True, False, False, "sgc_npt", "npt"),
    "sgc-nvt": (True, True, False, False, "sgc_npt", "nvt"),
    "vcsgc-npt": (True, True, False, False, "sgc_npt", "npt"),
    "vcsgc-nvt": (True, True, False, False, "sgc_npt", "nvt"),
}


def reference_neighbours(cutoff: float) -> int:
    """Neighbours within `cutoff` in the calibration lattice (fcc, a = 4.00 A)."""
    a = 4.0
    basis = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    n = math.ceil(cutoff / a) + 1
    count = 0
    for i, j, k in itertools.product(range(-n, n + 1), repeat=3):
        for bx, by, bz in basis:
            d = a * math.sqrt((i + bx) ** 2 + (j + by) ** 2 + (k + bz) ** 2)
            if 0 < d < cutoff:
                count += 1
    return count


def neighbour_ratio(args) -> tuple[float, str, int | None]:
    """Edge-count ratio of the target system to the calibration system."""
    ref = reference_neighbours(args.cutoff)
    if args.structure:
        try:
            from ase.io import read
            from ase.neighborlist import neighbor_list
        except ImportError:
            sys.exit(
                "--structure needs ASE (pip install ase), or pass --mean-neighbors / --density instead"
            )
        atoms = read(args.structure)
        i = neighbor_list("i", atoms, args.cutoff)
        mean = len(i) / len(atoms)
        return (
            min(mean, args.max_neighbors) / ref,
            f"{mean:.1f} neighbours/atom from {args.structure} (reference {ref})",
            len(atoms),
        )
    if args.mean_neighbors:
        return (
            min(args.mean_neighbors, args.max_neighbors) / ref,
            f"{args.mean_neighbors:.1f} neighbours/atom given (reference {ref})",
            None,
        )
    if args.density:
        return (
            args.density / REF_RHO,
            f"density {args.density:.4f} vs reference {REF_RHO} atoms/A^3 (dense-solid approximation)",
            None,
        )
    return (
        1.0,
        "no structure/density given: assuming a dense metal like the fcc Au-Pt calibration",
        None,
    )


def size_factor(n_atoms: int) -> float:
    s = CAL["size_scaling"]
    if n_atoms <= s["ramp_from_atoms"]:
        return 1.0
    frac = min(
        1.0,
        (n_atoms - s["ramp_from_atoms"]) / (s["ramp_to_atoms"] - s["ramp_from_atoms"]),
    )
    return 1.0 + (s["large_system_factor"] - 1.0) * frac


def valid_classes(kind: str, merge_ok: bool) -> list[str]:
    """Settings classes that are correct for this kind (the planner ranks these)."""
    has_mc, has_md, _, geom_fixed, _, _ = KINDS[kind]
    classes = []
    if merge_ok and geom_fixed and not has_md:
        classes.append(
            "compiled_merged"
        )  # compile only when neither graph shape nor composition changes
    if merge_ok:
        classes.append("eager_merged")
    classes.append("eager_unmerged")
    if has_md:  # checkpointing only saves memory in a backward pass: MC-only energy calls have none
        if merge_ok:
            classes.append("checkpointed_merged")
        classes.append("checkpointed_unmerged")
    return classes


def forced_class_notes(kind: str, cls: str, merge_ok: bool) -> list[str]:
    _, _, _, geom_fixed, _, _ = KINDS[kind]
    notes = []
    if cls == "compiled_merged" and not geom_fixed:
        notes.append(
            "WARNING: compile with changing geometry (MD/NPT) recompiles every step: "
            "measured ~20 min lost per process, then falls back to uncompiled speed."
        )
    if "merged" in cls and "unmerged" not in cls and not merge_ok:
        notes.append(
            "ERROR: merge_mole holds one composition; here composition changes (SGC/VC-SGC) or "
            "differs between batched walkers (fairchem 2.21 asserts, 2.22 silently falls back)."
        )
    return notes


def _lookup(table: dict, cls: str) -> tuple[dict, str] | None:
    """Exact class first (e.g. checkpointed_merged), then its generic 'checkpointed' entry."""
    for key in (cls, "checkpointed" if cls.startswith("checkpointed") else None):
        if key and key in table:
            return table[key], key
    return None


def phase_coeffs(table: str, cls: str) -> dict:
    return _lookup(CAL["time_model"][table], cls)[0]


def memory_coeffs(family: str, cls: str) -> tuple[dict, str]:
    fam = CAL["memory_model"][family]
    hit = _lookup(fam, cls)
    if hit:
        return hit
    key = cls
    fallback = "eager_unmerged" if "eager_unmerged" in fam else next(iter(fam))
    return fam[fallback], f"{fallback} (no {key} measurement for {family})"


def step_ms(
    coef: dict, width: int, n_eff: float, compiled: bool, n_atoms: int
) -> float:
    per_atom = coef["c1_ms_per_atom"] * (size_factor(n_atoms) if compiled else 1.0)
    return coef["c1_ms_per_atom"] * coef["n0_atoms"] + per_atom * width * n_eff


def gpu_spec(name: str) -> dict:
    if name == "any":
        return {
            "capacity_gib": min(g["capacity_gib"] for g in CAL["gpus"].values()),
            "time_factor": max(g["time_factor"] for g in CAL["gpus"].values()),
            "note": "card not pinned: sized for the smallest (A100-PCIE-40GB) and slowest card",
        }
    return CAL["gpus"][name]


def evaluate(ctx: dict, cls: str, max_width: int) -> dict:
    """Width table (memory, s/block, throughput) for one settings class."""
    has_mc, has_md = ctx["has_mc"], ctx["has_md"]
    n_eff, n_atoms, gpu = ctx["n_eff"], ctx["n_atoms"], ctx["gpu"]
    compiled = cls == "compiled_merged"
    notes = []
    mem_c, mem_key = memory_coeffs(ctx["family"], cls)
    if mem_key not in (cls, "checkpointed"):
        notes.append(f"memory for {cls}: using {mem_key}")
    mem_scale = (n_eff / REF_N) * (
        1.0 if cls.startswith("checkpointed") else size_factor(n_atoms)
    )

    def memory(width: int) -> float:
        if mem_c.get("superlinear"):
            # The fit's intercept is an artifact of faster-than-linear growth: scale the
            # whole measured curve rather than only its per-walker term.
            return max(
                0.5, (mem_c["base_gib"] + width * mem_c["per_walker_gib"]) * mem_scale
            )
        return max(0.5, mem_c["base_gib"] + width * mem_c["per_walker_gib"] * mem_scale)

    def block_seconds(width: int) -> dict:
        mc = md = handoff = 0.0
        if has_mc:
            mc = ctx["mc_steps"] * step_ms(
                phase_coeffs("mc_energy_only", cls), width, n_eff, compiled, n_atoms
            )
        if has_md:
            md_coef = phase_coeffs("md_full_outputs", cls)
            md = ctx["md_steps"] * step_ms(md_coef, width, n_eff, False, n_atoms)
            if has_mc:  # force/energy recompute after each MC block
                handoff = step_ms(md_coef, width, n_eff, False, n_atoms)
        f = gpu["time_factor"] / 1000.0
        return {
            "mc": mc * f,
            "md": md * f,
            "handoff": handoff * f,
            "total": (mc + md + handoff) * f,
        }

    widths = []
    for w in range(1, max_width + 1):
        mem, b = memory(w), block_seconds(w)
        widths.append(
            {
                "width": w,
                "memory_gib": mem,
                "fits": mem <= ctx["budget"],
                "block_s": b["total"],
                "mc_s": b["mc"],
                "md_s": b["md"],
                "walker_blocks_per_s": w / b["total"],
            }
        )
    for row in widths:
        row["gain_vs_w1"] = (
            row["walker_blocks_per_s"] / widths[0]["walker_blocks_per_s"]
        )
        row["per_walker_slowdown"] = row["block_s"] / widths[0]["block_s"]
    feasible = [r for r in widths if r["fits"]]
    rec = None
    if feasible:
        best = max(r["walker_blocks_per_s"] for r in feasible)
        rec = min(
            (
                r
                for r in feasible
                if r["walker_blocks_per_s"] >= ctx["efficiency"] * best
            ),
            key=lambda r: r["width"],
        )
    return {
        "settings_class": cls,
        "inference_settings": SPECS[cls],
        "widths": widths,
        "recommended": rec,
        "best_throughput": max(
            (r["walker_blocks_per_s"] for r in feasible), default=0.0
        ),
        "notes": notes,
    }


def plan(args) -> dict:
    ratio, ratio_note, struct_n = neighbour_ratio(args)
    n_atoms = args.n_atoms or struct_n
    if not n_atoms:
        sys.exit("give --n-atoms or --structure")
    has_mc, has_md, comp_fixed, geom_fixed, family, ensemble = KINDS[args.kind]
    walkers = args.walkers or (1 if args.mode == "single" else None)
    mode = (
        args.mode
        if args.mode != "auto"
        else ("single" if walkers == 1 else "throughput")
    )
    notes = []
    # Merge holds one composition. A single system (or several run one at a time, each with
    # its own freshly merged model) can use it whenever its composition is fixed; a batch
    # can only if every walker in it has the same composition.
    merge_ok = comp_fixed and (mode == "single" or not args.mixed_compositions)
    if comp_fixed and mode == "throughput":
        notes.append(
            "--mixed-compositions: merge off -- batched walkers differ, a merged model holds one "
            "composition."
            if args.mixed_compositions
            else "merge assumes every batched walker has the same composition (replicas of one structure); "
            "pass --mixed-compositions if they differ in composition, structure or size."
        )
    if comp_fixed and mode == "single":
        notes.append(
            "single system: merge is valid (fixed composition). Running several systems one after "
            "another needs a freshly merged model per system (load a new UMAWrapper per system)."
        )

    ctx = {
        "has_mc": has_mc,
        "has_md": has_md,
        "family": family,
        "n_atoms": n_atoms,
        "n_eff": n_atoms * ratio,
        "mc_steps": args.mc_steps
        if args.mc_steps is not None
        else (round(0.2 * n_atoms) if has_mc else 0),
        "md_steps": args.md_steps if has_md else 0,
        "gpu": gpu_spec(args.gpu),
        "efficiency": args.efficiency,
    }
    ctx["budget"] = ctx["gpu"]["capacity_gib"] * args.memory_fraction
    max_width = (
        1 if mode == "single" else min(args.max_width, walkers or args.max_width)
    )

    if args.settings != "auto":
        cls = args.settings
        if cls == "checkpointed":
            cls = "checkpointed_merged" if merge_ok else "checkpointed_unmerged"
        candidates = [cls]
        notes += forced_class_notes(args.kind, cls, merge_ok)
    else:
        candidates = valid_classes(args.kind, merge_ok)
    evaluated = [evaluate(ctx, cls, max_width) for cls in candidates]
    feasible = [e for e in evaluated if e["recommended"]]
    # Throughput regime: most walker-blocks/s per GPU at each class's own best width.
    # Single regime (width 1): shortest time per block. Same number at width 1, so one key.
    # Ties within 3% go to the class listed first (faster to set up / less exotic).
    chosen = None
    if feasible:
        top = max(e["best_throughput"] for e in feasible)
        chosen = next(e for e in feasible if e["best_throughput"] >= 0.97 * top)
    regime = mode
    if (
        chosen
        and mode == "throughput"
        and chosen["recommended"]["width"] == 1
        and (walkers or 2) > 1
    ):
        wider = [r for r in chosen["widths"] if r["width"] == 2]
        if wider and not wider[0]["fits"]:
            regime = "single (too large to batch on this card)"
        elif not wider:
            regime = "single (one walker)"
        else:
            regime = "throughput, but batching does not pay here"

    if chosen:
        rec, cls = chosen["recommended"], chosen["settings_class"]
        notes += chosen["notes"]
        if rec["width"] > MEASURED_MAX_WIDTH:
            notes.append(
                f"recommended width {rec['width']} is beyond the measured range (widths 1-{MEASURED_MAX_WIDTH} "
                "for these settings; older full-force Kawasaki data saturated by ~8): confirm with a short "
                f"profile at {MEASURED_MAX_WIDTH}/{rec['width']}/{2 * rec['width']} before production."
            )
        if family == "sgc_npt" and cls == "eager_unmerged" and mode == "throughput":
            notes.append(
                "SGC-NPT memory grew faster than linear up to width 4 (9.0 / 17.8 / 48.7 GiB at 500 atoms); "
                "widths above 4 are extrapolated -- profile before relying on them."
            )
        if cls.startswith("checkpointed"):
            notes.append(
                "activation checkpointing chosen because the faster settings do not fit: ~2x slower per MD "
                "step, ~2-3x less memory per walker. A larger card keeps the faster settings."
            )
        if cls == "compiled_merged":
            notes.append(
                "compiled (turbo): expect a ~10-20 s compile on the first block; valid only because "
                "geometry and composition are both fixed (MC-only Kawasaki)."
            )
        if family == "sgc_npt" and regime.startswith("single"):
            notes.append(
                "single-system SGC-NPT: MD could run on a separate merged model re-merged on the current "
                "composition each block (merged NPT is 1.8x faster at width 1; one merge ~0.8 s). "
                "HybridMCMD supports separate models + before_md_block, but the re-merge is not built yet."
            )
    if n_atoms > 2.5 * REF_N or n_atoms < 0.4 * REF_N or abs(ratio - 1) > 0.5:
        notes.append(
            "far from the 500-atom fcc calibration: treat times as +-30% and confirm the width "
            "with a short 1/2/4 profile (or run_campaign's automatic width sweep) before production."
        )
    if ensemble == "nvt":
        notes.append(
            "NVT: MD needs forces but not stress; set active_outputs={'energy','forces'} for the MD "
            "model to also skip the strain derivative (supported, speed gain unmeasured). "
            "Coefficients here are the NPT ones (conservative)."
        )

    job = None
    if chosen:
        rec = chosen["recommended"]
        waves = math.ceil((walkers or rec["width"]) / rec["width"])
        per_wave = args.n_blocks * rec["block_s"] + CAL["process_overhead_s"]
        job = {
            "waves_on_one_gpu": waves,
            "wall_s_per_wave": per_wave,
            "wall_s_total_one_gpu": waves * per_wave,
            "gpus_for_one_wave": waves,
            "suggested_time_limit_h": math.ceil(1.5 * per_wave / 3600 * 4) / 4,
        }
    comparison = [
        {
            "settings_class": e["settings_class"],
            "width": e["recommended"]["width"] if e["recommended"] else None,
            "walker_blocks_per_s": e["best_throughput"],
            "w1_block_s": e["widths"][0]["block_s"],
            "w1_memory_gib": e["widths"][0]["memory_gib"],
            "w1_fits": e["widths"][0]["fits"],
        }
        for e in evaluated
    ]
    return {
        "kind": args.kind,
        "mode": mode,
        "regime": regime,
        "walkers": walkers,
        "n_atoms": n_atoms,
        "neighbour_ratio": ratio,
        "neighbour_note": ratio_note,
        "n_atoms_effective": ctx["n_eff"],
        "settings_class": chosen["settings_class"] if chosen else None,
        "inference_settings": chosen["inference_settings"] if chosen else None,
        "mc_energy_only": has_mc,
        "mc_steps_per_block": ctx["mc_steps"],
        "md_steps_per_block": ctx["md_steps"],
        "gpu": args.gpu,
        "gpu_note": ctx["gpu"].get("note", ""),
        "memory_budget_gib": ctx["budget"],
        "comparison": comparison,
        "widths": chosen["widths"] if chosen else evaluated[0]["widths"],
        "recommended": chosen["recommended"] if chosen else None,
        "job": job,
        "notes": notes,
    }


def report(p: dict) -> str:
    out = [
        f"kind={p['kind']}  n_atoms={p['n_atoms']}  effective={p['n_atoms_effective']:.0f} ({p['neighbour_note']})",
        f"regime: {p['regime']}  (walkers: {p['walkers'] or 'not given'})",
        f'settings: {p["settings_class"]}  ->  inference_settings="{p["inference_settings"]}"',
        f"MC energy-only: {'yes' if p['mc_energy_only'] else 'n/a'}   mc_steps/block={p['mc_steps_per_block']}  md_steps/block={p['md_steps_per_block']}",
        f"GPU: {p['gpu']} ({p['gpu_note']})  budget {p['memory_budget_gib']:.1f} GiB",
        "",
        "settings compared (rec. width; best walker-blk/s that fits; width-1 time and memory):",
        f"  {'class':<22} {'width':>5} {'best blk/s':>13} {'w1 s/block':>11} {'w1 GiB':>7}",
    ]
    for c in p["comparison"]:
        mark = "  <== chosen" if c["settings_class"] == p["settings_class"] else ""
        width = c["width"] if c["width"] else "-"
        fits = "" if c["w1_fits"] else " (no fit)"
        out.append(
            f"  {c['settings_class']:<22} {width:>5} {c['walker_blocks_per_s']:>13.3f} "
            f"{c['w1_block_s']:>11.2f} {c['w1_memory_gib']:>7.1f}{fits}{mark}"
        )
    out += [
        "",
        f"{'width':>5} {'GiB':>7} {'fits':>5} {'s/block':>8} {'walker-blk/s':>13} {'gain':>6} {'slowdown':>9}",
    ]
    rec_w = p["recommended"]["width"] if p["recommended"] else None
    misses = 0
    for r in p["widths"]:
        misses += not r["fits"]
        if misses > 2:  # two rows past the memory limit are enough to show it
            break
        if (
            r["width"] > 16
            and r["width"] not in (24, 32, 48, 64)
            and r["width"] != rec_w
        ):
            continue
        mark = "  <== recommended" if r["width"] == rec_w else ""
        out.append(
            f"{r['width']:>5} {r['memory_gib']:>7.1f} {'yes' if r['fits'] else 'NO':>5} {r['block_s']:>8.2f} "
            f"{r['walker_blocks_per_s']:>13.3f} {r['gain_vs_w1']:>5.2f}x {r['per_walker_slowdown']:>8.2f}x{mark}"
        )
    if p["job"]:
        j = p["job"]
        out += [
            "",
            f"one wave of {rec_w} walkers: {j['wall_s_per_wave'] / 60:.1f} min; "
            f"{j['waves_on_one_gpu']} wave(s) on one GPU = {j['wall_s_total_one_gpu'] / 3600:.2f} h "
            f"(or {j['gpus_for_one_wave']} GPU(s) in parallel); suggested --time per job: {j['suggested_time_limit_h']} h",
        ]
    else:
        out += [
            "",
            "NOTHING FITS at width 1 on this card: use a larger card, fewer atoms, or checkpointing.",
        ]
    if p["notes"]:
        out += [""] + [f"note: {n}" for n in p["notes"]]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--kind", required=True, choices=sorted(KINDS))
    ap.add_argument("--n-atoms", type=int)
    ap.add_argument(
        "--structure",
        help="ASE-readable structure: n_atoms and neighbour density are taken from it",
    )
    ap.add_argument(
        "--density", type=float, help="atoms per A^3 (dense solids/liquids)"
    )
    ap.add_argument(
        "--mean-neighbors", type=float, help="mean neighbours per atom within --cutoff"
    )
    ap.add_argument(
        "--cutoff",
        type=float,
        default=6.0,
        help="UMA graph cutoff in A (check backbone.cutoff)",
    )
    ap.add_argument("--max-neighbors", type=int, default=300)
    ap.add_argument("--gpu", default="any", choices=["any", *CAL["gpus"]])
    ap.add_argument(
        "--walkers",
        type=int,
        help="independent walkers (state points x replicas) to run in total",
    )
    ap.add_argument("--n-blocks", type=int, default=100)
    ap.add_argument(
        "--mc-steps", type=int, help="MC steps per block (default round(0.2 * n_atoms))"
    )
    ap.add_argument(
        "--md-steps",
        type=int,
        default=50,
        help="MD steps per block (MD-only: steps per block)",
    )
    ap.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "throughput", "single"],
        help="throughput: many independent walkers, batch them and rank settings by walker-blocks/s "
        "per GPU; single: one system (width 1), rank by time per block. auto: single if "
        "--walkers 1, else throughput",
    )
    ap.add_argument(
        "--mixed-compositions",
        action="store_true",
        help="walkers in one batch (or systems sharing one model) differ in composition, "
        "structure or size: disables merge_mole",
    )
    ap.add_argument(
        "--settings",
        default="auto",
        choices=[
            "auto",
            "compiled_merged",
            "eager_merged",
            "eager_unmerged",
            "checkpointed",
            "checkpointed_merged",
            "checkpointed_unmerged",
        ],
    )
    ap.add_argument("--memory-fraction", type=float, default=CAL["memory_fraction"])
    ap.add_argument(
        "--efficiency",
        type=float,
        default=0.97,
        help="recommend the smallest width reaching this fraction of the best feasible throughput "
        "(0.97: drop only widths that add < 3%%; SGC-type batching gains are 10-20%%, so a looser "
        "value would discard them)",
    )
    ap.add_argument("--max-width", type=int, default=64)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    p = plan(args)
    print(json.dumps(p, indent=2) if args.json else report(p))


if __name__ == "__main__":
    main()
