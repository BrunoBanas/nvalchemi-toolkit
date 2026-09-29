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
"""Report which MOLE implementation the hybrid campaign's UMA checkpoint
actually runs: the sequential pure-PyTorch loop (``MOLE``) or the
vectorized ``fairchem_cpp.ops.segment_mm`` path (``MOLEDGL``).

Background
----------
UMA replaces ordinary ``nn.Linear`` layers with a mixture-of-linear-experts
module that gives each system (walker) in a batch its own expert-mixed
weights -- necessary because SGC/Kawasaki walkers each carry a
different, independently-evolving composition. There are two
implementations of that module (``fairchem/core/models/uma/nn/mole.py``):
``MOLEDGL`` applies all walkers' weights in one fused ``segment_mm`` call;
``MOLE`` is a pure-PyTorch fallback with an explicit Python ``for`` loop
over walkers -- a genuine serial loop, not a parallel tensor op. Which one
a given checkpoint uses is decided by a ``mole_layer_type`` string baked
into the model architecture at construction time (default ``"pytorch"``
in fairchem-core's own code); ``fairchem_cpp`` being importable only makes
the ``"dgl"`` choice legal, it does not select it. That means the choice
is effectively fixed by whatever config the checkpoint was exported with,
and there is no config attribute that records it after the fact -- the
only ground truth is which class is actually sitting at each MOLE site in
the loaded model. This script checks that directly.

Usage (on any GPU node -- MOLE substitution happens at model construction,
so no forward pass or batch is needed):

    python benchmark/uma_efficiency/check_mole_layer_type.py

Matches benchmark/hybrid_sgc_npt/run_campaign.py's production CHECKPOINT /
TASK / INFERENCE_SETTINGS constants exactly, so this reports what your
actual campaigns run, not some other checkpoint variant. Override via
--checkpoint/--task if you want to check a different one.

Once fairchem_cpp is installed, rerun with --mole-layer-type dgl to check
whether UMAWrapper.from_checkpoint's overrides mechanism actually flips
these sites to MOLEDGL -- fairchem's MLIPPredictUnit threads a user
``overrides`` dict straight into the backbone's own construction kwargs
(see fairchem/core/units/mlip_unit/predict.py's _build_overrides_from_settings:
user["backbone"] takes precedence over the inference-settings-derived
defaults), and UMAWrapper.from_checkpoint already forwards ``overrides``
unchanged, so no checkpoint file or nvalchemi code change is needed to
try this -- run_campaign.py would only need
``overrides={"backbone": {"moe_layer_type": "dgl"}}`` added to its own
UMAWrapper.from_checkpoint(...) call to use it in production.
"""

from __future__ import annotations

import argparse
import sys

import torch

# Must match benchmark/hybrid_sgc_npt/run_campaign.py's module constants.
CHECKPOINT = "uma-s-1p2"
TASK = "omat"
INFERENCE_SETTINGS = "batch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--inference-settings", default=INFERENCE_SETTINGS)
    parser.add_argument(
        "--mole-layer-type",
        default=None,
        choices=["pytorch", "dgl"],
        help=(
            "Override backbone.moe_layer_type via UMAWrapper.from_checkpoint's "
            "overrides= mechanism (e.g. 'dgl' once fairchem_cpp is installed, "
            "to check whether the checkpoint actually switches to MOLEDGL). "
            "Omit to load with whatever the checkpoint's own config requests."
        ),
    )
    return parser.parse_args()


def find_mole_modules(root: torch.nn.Module):
    """Every MOLE/MOLEDGL submodule under ``root``, keyed by qualified name."""
    from fairchem.core.models.uma.nn.mole import MOLE, MOLEDGL

    return [
        (name, type(module).__name__)
        for name, module in root.named_modules()
        if isinstance(module, (MOLE, MOLEDGL))
    ]


def main() -> int:
    args = parse_args()

    print(f"[env] python={sys.executable}")
    print(f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")

    try:
        import fairchem_cpp  # noqa: F401

        fairchem_cpp_importable = True
        print("[env] fairchem_cpp: IMPORTABLE")
    except ModuleNotFoundError:
        fairchem_cpp_importable = False
        print("[env] fairchem_cpp: NOT INSTALLED (ModuleNotFoundError)")

    from fairchem.core.models.uma.nn import mole as mole_mod

    print(
        "[env] fairchem.core.models.uma.nn.mole.fairchem_cpp_found = "
        f"{mole_mod.fairchem_cpp_found}"
    )

    from nvalchemi.models.uma import UMAWrapper

    device = "cuda" if torch.cuda.is_available() else "cpu"
    overrides = (
        {"backbone": {"moe_layer_type": args.mole_layer_type}}
        if args.mole_layer_type is not None
        else None
    )
    print(
        f"\n[load] UMAWrapper.from_checkpoint({args.checkpoint!r}, "
        f"task_name={args.task!r}, device={device!r}, "
        f"inference_settings={args.inference_settings!r}, "
        f"overrides={overrides!r}) ..."
    )
    wrapper = UMAWrapper.from_checkpoint(
        args.checkpoint,
        task_name=args.task,
        device=device,
        inference_settings=args.inference_settings,
        overrides=overrides,
    )

    # Same attribute path UMAWrapper's own methods use internally (see e.g.
    # its _extract_cutoff / forward implementations in nvalchemi/models/uma.py)
    # to reach the raw GNN backbone under fairchem's predict_unit wrapper.
    backbone = wrapper.predict_unit.model.module.backbone

    mole_sites = find_mole_modules(backbone)
    if not mole_sites:
        print(
            "\n[FAIL] No MOLE/MOLEDGL modules found anywhere under "
            "predict_unit.model.module.backbone -- either this checkpoint "
            "doesn't use MOLE (unexpected for UMA), or fairchem-core changed "
            "its internal layout since this script was written. Inspect "
            "`backbone` by hand: backbone.named_modules()."
        )
        return 1

    kinds = sorted({kind for _, kind in mole_sites})
    print(
        f"\n[result] {len(mole_sites)} MOLE-family layers found; classes present: {kinds}"
    )
    for name, kind in mole_sites[:5]:
        print(f"    {name}: {kind}")
    if len(mole_sites) > 5:
        print(f"    ... and {len(mole_sites) - 5} more")

    print()
    if kinds == ["MOLEDGL"]:
        print(
            "[VERDICT] This checkpoint runs the VECTORIZED segment_mm path "
            "(MOLEDGL) already. The per-walker sequential loop is NOT your "
            "bottleneck here -- co-batching should scale with width, "
            "modulo ordinary GPU compute saturation at large n_atoms. If "
            "you're still seeing flat wall_seconds/walker_blocks_per_second "
            "vs. width, look elsewhere (e.g. the neighbor-list hook, or "
            "genuine compute saturation at a single walker's size)."
        )
    elif kinds == ["MOLE"]:
        note = (
            "fairchem_cpp is importable in this environment, so the "
            "checkpoint's own saved architecture config is what's choosing "
            "'pytorch' here -- installing/building fairchem_cpp again will "
            "change nothing for this checkpoint."
            if fairchem_cpp_importable
            else "fairchem_cpp is not installed, which is consistent with "
            "this and is the thing to fix if you want MOLEDGL instead."
        )
        print(
            "[VERDICT] This checkpoint runs the SEQUENTIAL pure-PyTorch "
            f"loop (MOLE) at every expert-mixing layer -- confirmed serial "
            f"batching bottleneck. {note}"
        )
    else:
        print(
            f"[VERDICT] Mixed classes present ({kinds}) -- unexpected; inspect `mole_sites` by hand."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
