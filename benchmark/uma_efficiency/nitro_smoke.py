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
"""Smoke-test UMA Nitro through ``UMAWrapper`` on one GPU.

Loads ``uma-s-1p2p1`` with the ``"default"`` (Nitro) and ``"high-precision"``
presets, evaluates one periodic system, and reports whether Nitro actually ran,
the graph generator version, per-call time, and the energy/force deviation.

    python benchmark/uma_efficiency/nitro_smoke.py --element Cu --reps 5
"""

from __future__ import annotations

import argparse
import time

import torch
from ase.build import bulk

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.uma import UMAWrapper


def _timed(wrapper: UMAWrapper, batch: Batch, steps: int) -> tuple[dict, float]:
    out = wrapper(batch)  # first call: Nitro capture / compile
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        out = wrapper(batch)
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) / steps


def main() -> None:
    """Compare Nitro against high-precision on one system."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="uma-s-1p2p1")
    parser.add_argument("--task", default="omat")
    parser.add_argument("--element", default="Cu")
    parser.add_argument("--reps", type=int, default=5, help="cubic supercell repeats")
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    atoms = bulk(args.element, cubic=True) * (args.reps,) * 3
    atoms.rattle(0.05, seed=0)
    batch = Batch.from_data_list([AtomicData.from_atoms(atoms)]).to("cuda")
    print(f"{len(atoms)} atoms, GPU {torch.cuda.get_device_name()}")

    results = {}
    for preset in ("default", "high-precision"):
        wrapper = UMAWrapper.from_checkpoint(
            args.checkpoint,
            task_name=args.task,
            device="cuda",
            inference_settings=preset,
        )
        out, dt = _timed(wrapper, batch, args.steps)
        settings = wrapper.predict_unit.inference_settings
        print(
            f"{preset:>14}: nitro={settings.nitro} "
            f"graph_gen=v{settings.internal_graph_gen_version} {dt * 1e3:.1f} ms/call"
        )
        results[preset] = out
        del wrapper
        torch.cuda.empty_cache()

    ref, nitro = results["high-precision"], results["default"]
    de = (nitro["energy"] - ref["energy"]).abs().max().item() / len(atoms)
    df = (nitro["forces"] - ref["forces"]).abs().max().item()
    print(f"|dE|/atom = {de * 1e3:.3f} meV, max|dF| = {df * 1e3:.3f} meV/A")


if __name__ == "__main__":
    main()
