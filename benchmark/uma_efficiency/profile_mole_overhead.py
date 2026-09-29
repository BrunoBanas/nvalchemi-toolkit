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
"""Where does UMA's unmerged mixture-of-experts (MoLE) path spend its time?

Merged NPT runs at 52.9 ms per MD step at width 1 and 151.0 ms at width 4
(500-atom Au-Pt, A100-SXM4-80GB); unmerged at 94.9 and 336.4 ms -- about
+45 ms per step per walker (efficiency_matrix/efficiency_a100sxm/npt). With
a fixed composition, unmerged MoLE computes the same function as a merged
model, so the gap is implementation overhead. On every forward call the
unmerged path (fairchem/core/models/uma/nn/mole.py, escn_moe.py):

  coefficients  set_MOLE_coefficients: composition embedding + routing MLP
                -> per-walker expert mixing coefficients
  sizes         set_MOLE_sizes: per-walker edge counts, then `.cpu()` -- a
                device-to-host copy that makes the CPU wait for the GPU
  weights       each MOLE layer: einsum of the 32 expert weight sets with
                each walker's coefficients
  segments      each MOLE layer: a Python loop, one F.linear per walker
                (the pure-PyTorch MOLE class; MOLEDGL uses one segment_mm)

This script times one model call (MD: energy + forces + stress; MC:
energy only) for merged and unmerged UMA at widths 1 and 4, and for
unmerged with each piece cached across calls, which is what a per-block
MoLE cache could do (composition fixed within an MD block; edge counts fixed
within an MC block). The caching variants are measurement ablations only:
their caches are valid because every call here evaluates the same batch.

  variant                  what is recomputed per call
  merged                   nothing MoLE-related (one plain linear per layer)
  unmerged                 coefficients, sizes, weights, segments (stock)
  unmerged_cache_weights   coefficients, sizes, segments
  unmerged_cache_sizes     coefficients, weights, segments
  unmerged_cache_all       segments only

What remains in unmerged_cache_all vs merged is the per-walker segment loop;
what cache_weights / cache_sizes recover is what a block cache would buy.
Every patched variant is first checked against stock fairchem on the same
batch (max |dE|/atom, max |dF|), so a mismatch with the installed fairchem's
MOLE code shows up as a nonzero error instead of a silently wrong timing.

Compositions: "same" = every walker has the same Pt count (replicas; the only
case merge is valid for); "mixed" = walkers at Pt fractions 0.30-0.60, unmerged
only -- merge must be off whenever walkers differ in composition.

Outputs (--output-dir): results.json (one record per case, rewritten after
each case), summary.md, profiles/<case>.txt (torch.profiler tables),
mole_source.py (the installed fairchem MOLE code, for checking the replicas).
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

# run_campaign.py (workload builders, size tables) is in the sibling benchmark folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hybrid_sgc_npt"))

import run_campaign as campaign  # noqa: E402
from fairchem.core.models.uma import escn_moe  # noqa: E402
from fairchem.core.models.uma.nn import mole  # noqa: E402

from nvalchemi.data import AtomicData, Batch  # noqa: E402
from nvalchemi.models.uma import UMAWrapper  # noqa: E402

MERGED = "compile=false,merge_mole=true,tf32=true,activation_checkpointing=false"
UNMERGED = "compile=false,merge_mole=false,tf32=true,activation_checkpointing=false"
CACHE_FLAGS = {
    "unmerged": set(),
    "unmerged_cache_weights": {"weights"},
    "unmerged_cache_sizes": {"sizes"},
    "unmerged_cache_all": {"weights", "sizes", "coefficients"},
}
OUTPUTS = {"md": {"energy", "forces", "stress"}, "mc": {"energy"}}
MIXED_PT_FRACTIONS = (0.30, 0.40, 0.50, 0.60)

# --------------------------------------------------------------------------
# Instrumented / cacheable replicas of fairchem's MoLE code
# --------------------------------------------------------------------------

ACTIVE: set[str] = set()  # cache flags of the running variant
STORE: dict = {}  # cached tensors; cleared between cases
ORIG = {
    "mole_forward": mole.MOLE.forward,
    "sizes": escn_moe.eSCNMDMoeBackbone.set_MOLE_sizes,
    "coefficients": escn_moe.eSCNMDMoeBackbone.set_MOLE_coefficients,
}
if hasattr(mole, "MOLEDGL"):
    ORIG["moledgl_forward"] = mole.MOLEDGL.forward


def _mixed_weights(module, pattern: str) -> torch.Tensor:
    key = ("weights", id(module))
    if "weights" in ACTIVE and key in STORE:
        return STORE[key]
    with torch.autocast(device_type=module.weights.device.type, enabled=False):
        weights = torch.einsum(
            pattern,
            module.weights,
            module.global_mole_tensors.expert_mixing_coefficients,
        )
    if "weights" in ACTIVE:
        STORE[key] = weights.detach()
    return weights


def mole_forward(self, x):
    """fairchem MOLE.forward (pure-PyTorch loop), with labels and an optional weight cache."""
    with record_function("mole.weights"):
        weights = _mixed_weights(self, "eoi, be->boi")
    with record_function("mole.segments"):
        out = []
        ac_start_idx = self.global_mole_tensors.ac_start_idx
        start_idxs = [0] + torch.cumsum(
            self.global_mole_tensors.mole_sizes, dim=0
        ).tolist()
        input_segment = (ac_start_idx, ac_start_idx + x.shape[0])
        for n, segment in enumerate(zip(start_idxs, start_idxs[1:])):
            overlap = mole.interval_intersection(input_segment, segment)
            if overlap is not None:
                start, end = overlap[0] - ac_start_idx, overlap[1] - ac_start_idx
                out.append(F.linear(x[start:end], weights[n], bias=self.bias))
        return torch.concatenate(out, dim=0)


def moledgl_forward(self, x):
    """fairchem MOLEDGL.forward (fused segment_mm), with labels and an optional weight cache."""
    with record_function("mole.weights"):
        weights = _mixed_weights(self, "eoi, be->bio")
    with record_function("mole.segments"):
        shape = x.shape
        sizes = self.global_mole_tensors.mole_sizes
        if x.ndim == 2:
            r = mole.fairchem_cpp.ops.segment_mm(x, weights, sizes)
        else:
            r = mole.fairchem_cpp.ops.segment_mm(
                x.reshape(-1, shape[-1]), weights, sizes * shape[1]
            ).reshape(*shape[:-1], -1)
        if self.bias is not None:
            r += self.bias
        return r


def set_sizes(self, nsystems, batch_full, edge_index):
    with record_function("mole.sizes"):
        key = ("sizes", id(self), nsystems, edge_index.shape[1])
        if "sizes" in ACTIVE and key in STORE:
            self.global_mole_tensors.mole_sizes = STORE[key]
            return None
        result = ORIG["sizes"](self, nsystems, batch_full, edge_index)
        if "sizes" in ACTIVE and self.num_experts:
            STORE[key] = self.global_mole_tensors.mole_sizes
        return result


def set_coefficients(self, atomic_numbers_full, batch_full, csd_mixed_emb):
    with record_function("mole.coefficients"):
        key = (
            "coefficients",
            id(self),
            atomic_numbers_full.shape[0],
            csd_mixed_emb.shape[0],
        )
        if "coefficients" in ACTIVE and key in STORE:
            self.global_mole_tensors.expert_mixing_coefficients = STORE[key]
            return None
        result = ORIG["coefficients"](
            self, atomic_numbers_full, batch_full, csd_mixed_emb
        )
        if "coefficients" in ACTIVE and self.num_experts:
            STORE[key] = self.global_mole_tensors.expert_mixing_coefficients
        return result


@contextlib.contextmanager
def instrumented(flags: set[str]):
    """Swap in the replicas for one case, then restore stock fairchem."""
    ACTIVE.clear()
    ACTIVE.update(flags)
    STORE.clear()
    mole.MOLE.forward = mole_forward
    if "moledgl_forward" in ORIG:
        mole.MOLEDGL.forward = moledgl_forward
    escn_moe.eSCNMDMoeBackbone.set_MOLE_sizes = set_sizes
    escn_moe.eSCNMDMoeBackbone.set_MOLE_coefficients = set_coefficients
    try:
        yield
    finally:
        mole.MOLE.forward = ORIG["mole_forward"]
        if "moledgl_forward" in ORIG:
            mole.MOLEDGL.forward = ORIG["moledgl_forward"]
        escn_moe.eSCNMDMoeBackbone.set_MOLE_sizes = ORIG["sizes"]
        escn_moe.eSCNMDMoeBackbone.set_MOLE_coefficients = ORIG["coefficients"]
        ACTIVE.clear()
        STORE.clear()


# --------------------------------------------------------------------------
# Workload
# --------------------------------------------------------------------------


def make_batch(
    template, pt_fractions: list[float], seed: int, device: torch.device
) -> Batch:
    walkers = []
    for index, fraction in enumerate(pt_fractions):
        data = AtomicData.from_atoms(template, device=device)
        n = data.num_nodes
        generator = torch.Generator(device=device).manual_seed(seed + index)
        numbers = torch.full_like(data.atomic_numbers, campaign.SPECIES[0])
        numbers[
            torch.randperm(n, device=device, generator=generator)[: round(fraction * n)]
        ] = campaign.SPECIES[1]
        data.atomic_numbers = numbers
        data.atomic_masses = None
        data.use_default_masses()
        walkers.append(data)
    batch = Batch.from_data_list(walkers)
    campaign._wrap_batch_positions(batch)
    return batch


def call(model: UMAWrapper, batch: Batch, device: torch.device) -> dict:
    out = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return out


def max_error(ref: dict, out: dict, n_atoms: int) -> dict:
    errors = {
        "energy_eV_per_atom": float((ref["energy"] - out["energy"]).abs().max())
        / n_atoms
    }
    if "forces" in ref and "forces" in out:
        errors["forces_eV_per_A"] = float((ref["forces"] - out["forces"]).abs().max())
    return errors


def device_time_us(event) -> float:
    for name in ("device_time_total", "cuda_time_total"):
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return float("nan")


def run_case(
    model, label, variant, op, composition, width, template, args, device
) -> dict:
    fractions = (
        [campaign.PT_FRACTION] * width
        if composition == "same"
        else list(MIXED_PT_FRACTIONS[:width])
    )
    batch = make_batch(template, fractions, args.seed, device)
    n_atoms = batch.positions.shape[0] // width
    model.model_config.active_outputs = set(OUTPUTS[op])
    flags = CACHE_FLAGS.get(variant)
    record = {
        "case": label,
        "variant": variant,
        "op": op,
        "composition": composition,
        "width": width,
        "n_atoms_per_walker": n_atoms,
        "pt_fractions": fractions,
    }

    # Exactness: the instrumented replica (second call, caches warm) vs stock fairchem.
    reference = call(model, batch, device)
    if flags is not None:
        with instrumented(flags):
            call(model, batch, device)
            patched = call(model, batch, device)
        record["max_error_vs_stock"] = max_error(reference, patched, n_atoms)

    context = instrumented(flags) if flags is not None else contextlib.nullcontext()
    with context:
        for _ in range(args.warmup):
            call(model, batch, device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        times = []
        for _ in range(args.steps):
            t0 = time.perf_counter()
            call(model, batch, device)
            times.append(time.perf_counter() - t0)
        record["ms_per_call_median"] = 1000 * statistics.median(times)
        record["ms_per_call_min"] = 1000 * min(times)
        record["ms_per_call_per_walker"] = record["ms_per_call_median"] / width
        if device.type == "cuda":
            record["peak_reserved_GiB"] = torch.cuda.max_memory_reserved(device) / 2**30

        activities = [ProfilerActivity.CPU] + (
            [ProfilerActivity.CUDA] if device.type == "cuda" else []
        )
        with profile(activities=activities) as prof:
            for _ in range(args.profile_steps):
                call(model, batch, device)
    averages = prof.key_averages()
    per_call = 1000.0 * args.profile_steps  # us totals -> ms per call
    record["labels_ms_per_call"] = {
        event.key: {
            "cpu": event.cpu_time_total / per_call,
            "device": device_time_us(event) / per_call,
            "count_per_call": event.count / args.profile_steps,
        }
        for event in averages
        if event.key.startswith("mole.")
    }
    record["host_sync_ms_per_call"] = {
        event.key: {
            "cpu": event.cpu_time_total / per_call,
            "count_per_call": event.count / args.profile_steps,
        }
        for event in averages
        if any(
            word in event.key
            for word in (
                "Synchronize",
                "Memcpy",
                "aten::item",
                "aten::_local_scalar_dense",
            )
        )
    }
    profiles = args.output_dir / "profiles"
    profiles.mkdir(exist_ok=True)
    sort_device = (
        "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    )
    (profiles / f"{label}.txt").write_text(
        f"{label}: {args.profile_steps} calls\n\n== by self device time ==\n"
        + averages.table(sort_by=sort_device, row_limit=30)
        + "\n\n== by self CPU time ==\n"
        + averages.table(sort_by="self_cpu_time_total", row_limit=30)
    )
    del batch
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def mole_layer_counts(model: UMAWrapper) -> dict:
    counts: dict[str, int] = {}
    for module in model.modules():
        name = type(module).__name__
        if name in ("MOLE", "MOLEDGL"):
            counts[name] = counts.get(name, 0) + 1
    return counts


def load(spec: str, device: torch.device, template) -> tuple[UMAWrapper, dict]:
    t0 = time.perf_counter()
    model = UMAWrapper.from_checkpoint(
        campaign.CHECKPOINT,
        task_name=campaign.TASK,
        device=str(device),
        inference_settings=spec,
    )
    load_s = time.perf_counter() - t0
    # First call: fairchem's lazy init, which for merge_mole is where the merge happens.
    batch = make_batch(template, [campaign.PT_FRACTION], 0, device)
    model.model_config.active_outputs = set(OUTPUTS["md"])
    t0 = time.perf_counter()
    call(model, batch, device)
    first_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    call(model, batch, device)
    second_s = time.perf_counter() - t0
    info = {
        "spec": spec,
        "load_s": load_s,
        "first_call_s": first_s,
        "second_call_s": second_s,
        "mole_layers": mole_layer_counts(model),
    }
    print(
        f"loaded {spec}: load {load_s:.1f} s, first call {first_s:.2f} s, second {second_s:.3f} s, "
        f"MoLE layers {info['mole_layers']}",
        flush=True,
    )
    return model, info


def write_summary(path: Path, results: dict) -> None:
    rows = results["cases"]
    merged = {
        (r["op"], r["width"]): r["ms_per_call_median"]
        for r in rows
        if r["variant"] == "merged"
    }
    lines = [
        "# MoLE overhead profile",
        "",
        f"GPU: {results['gpu']}  fairchem {results['fairchem_version']}  torch {results['torch_version']}  "
        f"n_atoms/walker {results['n_atoms']}",
        "",
        "Model loads (first call includes fairchem's merge for merge_mole):",
        "",
        "| spec | load s | first call s | second call s | MoLE layers |",
        "|---|---|---|---|---|",
    ]
    lines.extend(
        f"| `{info['spec']}` | {info['load_s']:.1f} | {info['first_call_s']:.2f} | "
        f"{info['second_call_s']:.3f} | {info['mole_layers']} |"
        for info in results["models"].values()
    )
    lines += [
        "",
        "| op | comp | width | variant | ms/call | ms/walker | vs merged | GiB | "
        "weights ms | sizes ms | coeff ms | segments ms | max dE/atom | max dF |",
        "|" + "---|" * 14,
    ]
    for r in rows:
        ref = merged.get((r["op"], r["width"]))
        lab = r.get("labels_ms_per_call", {})

        def cpu(name: str) -> str:
            return f"{lab[name]['cpu']:.2f}" if name in lab else "-"

        err = r.get("max_error_vs_stock", {})
        lines.append(
            f"| {r['op']} | {r['composition']} | {r['width']} | {r['variant']} | {r['ms_per_call_median']:.1f} | "
            f"{r['ms_per_call_per_walker']:.1f} | {r['ms_per_call_median'] / ref:.2f}x | "
            f"{r.get('peak_reserved_GiB', float('nan')):.2f} | {cpu('mole.weights')} | {cpu('mole.sizes')} | "
            f"{cpu('mole.coefficients')} | {cpu('mole.segments')} | "
            f"{err.get('energy_eV_per_atom', float('nan')):.1e} | {err.get('forces_eV_per_A', float('nan')):.1e} |"
            if ref
            else f"| {r['op']} | {r['composition']} | {r['width']} | {r['variant']} | {r['ms_per_call_median']:.1f} | "
            f"{r['ms_per_call_per_walker']:.1f} | - | {r.get('peak_reserved_GiB', float('nan')):.2f} | "
            f"{cpu('mole.weights')} | {cpu('mole.sizes')} | {cpu('mole.coefficients')} | {cpu('mole.segments')} | "
            f"{err.get('energy_eV_per_atom', float('nan')):.1e} | {err.get('forces_eV_per_A', float('nan')):.1e} |"
        )
    lines += [
        "",
        "Label columns are host (CPU) time inside each labelled region per call, which includes "
        "waiting on the GPU at a sync; device time per label is in results.json. "
        "Per-op tables are in profiles/.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n-atoms", type=int, default=500, choices=sorted(campaign.SIZE_REPEATS)
    )
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 4])
    parser.add_argument(
        "--ops", nargs="+", default=["md", "mc"], choices=sorted(OUTPUTS)
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["merged", *CACHE_FLAGS],
        choices=["merged", *CACHE_FLAGS],
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20, help="timed calls per case")
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026092801)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if max(args.widths) > len(MIXED_PT_FRACTIONS):
        raise SystemExit(
            f"--widths up to {len(MIXED_PT_FRACTIONS)} (one mixed composition per walker)"
        )
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        ("MOLE.forward", ORIG["mole_forward"]),
        ("set_MOLE_sizes", ORIG["sizes"]),
        ("set_MOLE_coefficients", ORIG["coefficients"]),
    ]
    if "moledgl_forward" in ORIG:
        sources.append(("MOLEDGL.forward", ORIG["moledgl_forward"]))
    (args.output_dir / "mole_source.py").write_text(
        "\n\n".join(
            f"# ---- {name} ----\n{inspect.getsource(fn)}" for name, fn in sources
        )
    )

    template = campaign.build_ase_structure(
        campaign.TEMPLATE_SYMBOL,
        campaign.CRYSTAL_STRUCTURE,
        campaign.LATTICE_A_ANG,
        campaign.SIZE_REPEATS[args.n_atoms],
        cubic=campaign.CONVENTIONAL_CELL,
    )
    import fairchem.core

    results = {
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "fairchem_version": getattr(fairchem.core, "__version__", "?"),
        "torch_version": torch.__version__,
        "fairchem_cpp_found": bool(getattr(mole, "fairchem_cpp_found", False)),
        "n_atoms": len(template),
        "models": {},
        "cases": [],
    }
    out_json = args.output_dir / "results.json"
    groups = [
        ("merged", MERGED, ["merged"]),
        ("unmerged", UNMERGED, [v for v in CACHE_FLAGS]),
    ]
    for group, spec, group_variants in groups:
        variants = [v for v in group_variants if v in args.variants]
        if not variants:
            continue
        model, info = load(spec, device, template)
        results["models"][group] = info
        for op in args.ops:
            for width in args.widths:
                # merge is only valid when every walker has the same composition
                compositions = (
                    ["same"] if group == "merged" or width == 1 else ["same", "mixed"]
                )
                for composition in compositions:
                    for variant in variants:
                        label = f"{op}_{composition}_w{width}_{variant}"
                        print(f"[{label}] ...", flush=True)
                        record = run_case(
                            model,
                            label,
                            variant,
                            op,
                            composition,
                            width,
                            template,
                            args,
                            device,
                        )
                        results["cases"].append(record)
                        print(
                            f"[{label}] {record['ms_per_call_median']:.1f} ms/call "
                            f"({record['ms_per_call_per_walker']:.1f} ms/walker), "
                            f"error {record.get('max_error_vs_stock', '-')}",
                            flush=True,
                        )
                        out_json.write_text(json.dumps(results, indent=2) + "\n")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_summary(args.output_dir / "summary.md", results)
    print((args.output_dir / "summary.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
