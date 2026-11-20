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
"""Swaps judged on locally relaxed energies."""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from nvalchemi.mc import RelaxedSwap
from nvalchemi.mc.relaxed_swap import fire_relax

from .conftest import default_model, tether_batch

TEMPERATURE = 1000.0


def test_fire_relaxes_movable_atoms_and_freezes_the_rest() -> None:
    model = default_model()
    batch = tether_batch(model, [1, 2, 1], 2)
    batch.positions += 0.3
    sampler = RelaxedSwap(model, TEMPERATURE)
    start = batch.positions.clone()
    e0, _ = sampler._evaluate(batch)
    movable = torch.tensor([True, True, False, True, True, False])
    e1 = fire_relax(batch, sampler._evaluate, movable, 40)
    assert bool((e1 < e0).all())
    assert torch.equal(batch.positions[~movable], start[~movable])
    offsets = (
        batch.positions[movable] - model.anchors.float().repeat(2, 1)[movable]
    ).norm(dim=1)
    assert float(offsets.max()) < 0.05


def test_rejected_trial_gets_species_and_positions_back() -> None:
    model = default_model()
    model.fields[:, 1] = torch.tensor([-50.0, 50.0, 50.0])  # B belongs on site 0
    batch = tether_batch(model, [2, 1, 1], 4)
    sampler = RelaxedSwap(model, TEMPERATURE, relax_steps=3, relax_radius=None)
    sampler.compute(batch)
    before = (
        batch.atomic_numbers.clone(),
        batch.positions.clone(),
        batch.energy.clone(),
    )
    sampler.step(batch)
    assert sampler.stats.accepted == 0
    assert torch.equal(batch.atomic_numbers, before[0])
    assert torch.equal(batch.positions, before[1])
    assert torch.equal(batch.energy, before[2])


def test_accepted_trial_keeps_its_relaxed_positions_inside_the_region() -> None:
    model = default_model()
    model.fields[:, 1] = torch.tensor([50.0, -50.0, 500.0])  # B wants site 1
    batch = tether_batch(model, [2, 1, 1], 4)
    batch.positions += 0.2
    sampler = RelaxedSwap(model, TEMPERATURE, relax_steps=20, relax_radius=1.0)
    sampler.compute(batch)
    before = batch.positions.clone()
    for _ in range(8):
        sampler.step(batch)
    numbers = batch.atomic_numbers.reshape(4, 3)
    assert bool((numbers[:, 1] == 2).all())
    moved = (batch.positions != before).any(dim=1).reshape(4, 3)
    # radius 1 A covers only the swapped atoms themselves (sites are 5 A apart)
    assert bool(moved[:, :2].all()) and not bool(moved[:, 2].any())
    energy, _ = sampler._evaluate(batch)
    assert torch.allclose(batch.energy.reshape(-1), energy, atol=1e-5)


@pytest.mark.parametrize("relax_baseline", [True, False])
def test_composition_is_conserved(relax_baseline: bool) -> None:
    model = default_model()
    batch = tether_batch(model, [2, 1, 1], 6)
    sampler = RelaxedSwap(
        model, TEMPERATURE, relax_steps=2, relax_baseline=relax_baseline
    )
    sampler.compute(batch)
    sampler.run(batch, n_steps=10)
    for row in batch.atomic_numbers.reshape(6, 3).tolist():
        assert Counter(row) == Counter([2, 1, 1])
    assert sampler.stats.attempted == 60


def test_energy_only_model_is_refused() -> None:
    model = default_model()
    model.model_config.active_outputs = {"energy"}
    batch = tether_batch(model, [2, 1, 1], 1)
    with pytest.raises(ValueError, match="forces"):
        RelaxedSwap(model, TEMPERATURE).step(batch)
