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
"""Nonequilibrium candidate MC swaps."""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from nvalchemi.mc import Displacement, NCMCSwap

from .conftest import (
    MASSES,
    arrangement_counts,
    default_model,
    exact_arrangement_probabilities,
    tether_batch,
)

TEMPERATURE = 1000.0


def test_one_switch_is_an_instantaneous_swap() -> None:
    """With n_switch=1 the work is the plain energy difference of the swap."""
    model = default_model()
    batch = tether_batch(model, [1, 1, 2], 8, seed=2)
    sampler = NCMCSwap(model, TEMPERATURE, n_switch=1, random_seed=4)
    sampler.compute(batch)
    before_energy = batch.energy.clone().reshape(-1)
    before_numbers = batch.atomic_numbers.clone()
    sampler.step(batch)
    first, second = sampler._proposal_first, sampler._proposal_second
    trial = before_numbers.clone()
    trial[first], trial[second] = before_numbers[second], before_numbers[first]
    saved = batch.atomic_numbers.clone()
    batch.atomic_numbers.copy_(trial)
    trial_energy = sampler.compute(batch)["energy"].reshape(-1)
    batch.atomic_numbers.copy_(saved)
    assert torch.allclose(sampler.last_work, trial_energy - before_energy, atol=1e-5)


def test_rejection_restores_species_positions_and_velocities() -> None:
    model = default_model()
    model.fields[:, 1] += torch.tensor([0.0, 50.0, 50.0])  # B is only welcome on site 0
    batch = tether_batch(model, [2, 1, 1], 4)
    batch.velocities = torch.randn_like(batch.positions)
    sampler = NCMCSwap(model, TEMPERATURE, n_switch=4)
    sampler.compute(batch)
    before = (
        batch.atomic_numbers.clone(),
        batch.positions.clone(),
        batch.velocities.clone(),
        batch.energy.clone(),
    )
    sampler.step(batch)
    assert sampler.stats.accepted == 0
    assert torch.equal(batch.atomic_numbers, before[0])
    assert torch.equal(batch.positions, before[1])
    assert torch.equal(batch.velocities, before[2])
    assert torch.equal(batch.energy, before[3])


def test_accepted_swaps_move_masses_and_refresh_swapped_velocities() -> None:
    model = default_model()
    model.fields[:, 1] = torch.tensor(
        [50.0, -50.0, 500.0]
    )  # B wants site 1, never site 2
    batch = tether_batch(model, [2, 1, 1], 4)
    batch.velocities = torch.zeros_like(batch.positions)
    sampler = NCMCSwap(model, TEMPERATURE, n_switch=3)
    sampler.compute(batch)
    for _ in range(10):
        sampler.step(batch)
    numbers = batch.atomic_numbers.reshape(4, 3)
    assert bool((numbers[:, 1] == 2).all())
    expected = torch.tensor([MASSES[int(z)] for z in batch.atomic_numbers.reshape(-1)])
    assert torch.equal(
        batch.atomic_masses.reshape(-1), expected.to(batch.atomic_masses.dtype)
    )
    moved = batch.velocities.abs().sum(dim=1).reshape(4, 3) > 0
    assert bool(moved[:, :2].all())  # the swapped atoms got fresh velocities
    assert not bool(moved[:, 2].any())


def test_energy_only_model_is_refused() -> None:
    model = default_model()
    model.model_config.active_outputs = {"energy"}
    batch = tether_batch(model, [2, 1, 1], 1)
    with pytest.raises(ValueError, match="forces"):
        NCMCSwap(model, TEMPERATURE).step(batch)


@pytest.mark.parametrize("n_switch", [1, 4])
def test_ncmc_with_displacements_samples_the_exact_arrangement_distribution(
    n_switch: int,
) -> None:
    """NCMC is exact at any dt: arrangement frequencies match the closed form."""
    model = default_model()
    numbers = [1, 1, 2]
    batch = tether_batch(model, numbers, 64, seed=5)
    ncmc = NCMCSwap(model, TEMPERATURE, n_switch=n_switch, dt=4.0, random_seed=6)
    displace = Displacement(model, TEMPERATURE, step_size=0.5, random_seed=7)
    ncmc.compute(batch)
    counts: dict[tuple[int, ...], int] = {}
    works: Counter[bool] = Counter()
    for sweep in range(1700):
        ncmc.step(batch)
        works[bool((ncmc.last_work.abs() > 1e-6).any())] += 1
        displace.synchronize(batch)
        for _ in range(2):
            displace.step(batch)
        ncmc.synchronize(batch)
        if sweep >= 200:
            arrangement_counts(batch, counts)
    total = sum(counts.values())
    exact = exact_arrangement_probabilities(model, numbers, TEMPERATURE)
    for arrangement, probability in exact.items():
        assert counts.get(arrangement, 0) / total == pytest.approx(
            probability, abs=0.015
        )
    assert ncmc.stats.accepted > 0
