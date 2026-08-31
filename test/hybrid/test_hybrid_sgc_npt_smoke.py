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
"""Timeout-guarded smoke test for the hybrid SGC-NPT workflow with real UMA inference.

Runs two hybrid MC-MD blocks on a 32-atom AuPt cell, batching two walkers at
different delta_mu together -- seconds on a GPU, not the hours of
benchmark/hybrid_sgc_npt/run_campaign.py's real 88-run campaign, which should
never be used as a correctness/smoke check for the method itself. Imports
run_campaign.py's own build_ase_structure/make_workload rather than
reimplementing the workload, so this exercises the exact production code
path instead of a parallel hand-rolled one.

This is a regression guard for two failures found debugging that campaign:

1. ``nvalchemi/mc/base.py::_ensure_observables`` called the AtomicData-only
   ``Batch.add_system_property`` on a real ``Batch`` -- ``AttributeError``.
2. ``inference_settings="default"`` let fairchem's merge_mole/compile fast
   path engage against the batch's initial (single) composition, then
   silently stalled for minutes falling back once SGC's per-step atom swaps
   made the two walkers' compositions diverge from what was merged/compiled.
   ``inference_settings="batch"`` is the fix -- see run_campaign.py's
   ``INFERENCE_SETTINGS``.

``@pytest.mark.timeout`` turns a reoccurrence of #2 into a fast, loud
failure instead of a silently hung GPU reservation.

Skipped when fairchem-core isn't installed, no CUDA device is visible, or
the UMA checkpoint can't be loaded (no HF access) -- mirrors
test/models/test_uma_nve_stability.py's pattern.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import torch

pytest.importorskip(
    "fairchem.core", reason="fairchem-core not installed; skipping UMA tests"
)

# run_campaign.py lives in benchmark/, not the nvalchemi package -- import it
# as a module (not reimplement its workload) so this test tracks the real
# campaign script instead of a parallel implementation that could drift.
_BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "hybrid_sgc_npt"
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

import run_campaign as campaign  # noqa: E402

from nvalchemi.models.uma import UMAWrapper  # noqa: E402
from nvalchemi.scheduling import RunSpec  # noqa: E402

# Two blocks is enough to exercise one MC pass (composition changes) and one
# MD pass (NPT dynamics) per walker. A hang here is a regression to fail on,
# not something worth waiting out.
_TIMEOUT_S = 180


@pytest.fixture(scope="module")
def uma_model() -> Any:
    try:
        return UMAWrapper.from_checkpoint(
            campaign.CHECKPOINT,
            task_name=campaign.TASK,
            device="cuda",
            inference_settings=campaign.INFERENCE_SETTINGS,
        )
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"could not load UMA checkpoint {campaign.CHECKPOINT}: {e}")


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.timeout(_TIMEOUT_S)
def test_hybrid_sgc_npt_smoke(uma_model: UMAWrapper) -> None:
    """Two hybrid MC-MD blocks on two batched, differently-doped AuPt walkers."""
    device = torch.device("cuda")

    template = campaign.build_ase_structure(
        campaign.TEMPLATE_SYMBOL,
        campaign.CRYSTAL_STRUCTURE,
        campaign.LATTICE_A_ANG,
        (2, 2, 2),  # 32 atoms -- fast to build and evaluate
        cubic=campaign.CONVENTIONAL_CELL,
    )

    # Two walkers at different delta_mu, same batch_group: batched together
    # onto one graph, so their compositions diverge from each other under
    # SGC swaps -- the exact heterogeneous-batch scenario "default" mishandles.
    runs = tuple(
        RunSpec(
            run_id=f"smoke.mu{index}",
            temperature_k=3000.0,
            pressure_ev_per_a3=campaign.PRESSURE_EV_PER_A3,
            chemical_potentials_ev={
                campaign.SPECIES[0]: 0.0,
                campaign.SPECIES[1]: delta_mu,
            },
            species=campaign.SPECIES,
            batch_group="smoke",
        )
        for index, delta_mu in enumerate((-0.5, 0.5))
    )

    hybrid, batch = campaign.make_workload(
        uma_model, template, runs, (None,) * len(runs), device
    )
    n_nodes_before = batch.num_nodes

    result = hybrid.run(batch, n_blocks=2)
    torch.cuda.synchronize(device)

    assert result is batch
    assert torch.isfinite(result.energy).all(), "non-finite energy after hybrid run"
    # SGC swaps species in place; it never adds or removes atoms.
    assert result.num_nodes == n_nodes_before
    # Swaps stay inside the configured reservoir species (Au/Pt only).
    assert set(result.atomic_numbers.tolist()) <= set(campaign.SPECIES)
    assert hybrid.mc.step_count == hybrid.mc_steps * 2
    assert hybrid.md.step_count == hybrid.md_steps * 2

    print()
    print(
        f"hybrid SGC-NPT smoke: {len(runs)} walkers, {n_nodes_before} atoms, "
        f"2 blocks, inference_settings={campaign.INFERENCE_SETTINGS!r}"
    )
    print(f"  mc acceptance: {hybrid.mc.stats.acceptance:.3f}")
    print(f"  peak GPU memory: {torch.cuda.max_memory_reserved(device) / 1024**3:.2f} GB")
