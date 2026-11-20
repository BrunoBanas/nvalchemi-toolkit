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
"""Off-lattice displacement MC and the mixed-move sampler."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.mc import CompositeMC, Displacement, NonlocalSwap

from .conftest import (
    arrangement_counts,
    default_model,
    exact_arrangement_probabilities,
    tether_batch,
)

TEMPERATURE = 1000.0


def test_single_mode_moves_one_atom_per_graph_and_restores_it() -> None:
    model = default_model()
    batch = tether_batch(model, [1, 1, 2], 4)
    sampler = Displacement(model, TEMPERATURE, step_size=0.3, random_seed=3)
    before = batch.positions.clone()
    sampler.pre_update(batch)
    moved = (batch.positions != before).any(dim=1).reshape(4, 3)
    assert moved.sum(dim=1).tolist() == [1, 1, 1, 1]
    assert (batch.positions - before).abs().max() <= 0.3 + 1e-6
    sampler._restore_rejected(batch, torch.ones(4, dtype=torch.bool))
    assert torch.equal(batch.positions, before)


def test_collective_mode_moves_every_atom_and_restores_only_rejected_graphs() -> None:
    model = default_model()
    batch = tether_batch(model, [1, 1, 2], 2)
    sampler = Displacement(model, TEMPERATURE, step_size=0.1, mode="collective")
    before = batch.positions.clone()
    sampler.pre_update(batch)
    assert bool((batch.positions != before).all())
    trial = batch.positions.clone()
    sampler._restore_rejected(batch, torch.tensor([True, False]))
    assert torch.equal(batch.positions[:3], before[:3])
    assert torch.equal(batch.positions[3:], trial[3:])


def test_invalid_arguments_raise() -> None:
    model = default_model()
    with pytest.raises(ValueError, match="mode"):
        Displacement(model, TEMPERATURE, mode="sideways")
    with pytest.raises(ValueError, match="step_size"):
        Displacement(model, TEMPERATURE, step_size=0.0)
    with pytest.raises(ValueError, match="weights"):
        CompositeMC(
            model, TEMPERATURE, [Displacement(model, TEMPERATURE)], weights=[0.0]
        )


def test_single_atom_moves_sample_the_exact_position_distribution() -> None:
    """Fixed species: each coordinate is Gaussian with variance kT/k."""
    torch.manual_seed(0)
    model = default_model()
    numbers = [1, 2, 1]
    batch = tether_batch(model, numbers, 64)
    sampler = Displacement(model, TEMPERATURE, step_size=0.5, random_seed=11)
    sampler.run(batch, n_steps=300)
    samples = []
    for _ in range(1500):
        sampler.step(batch)
        samples.append(batch.positions.reshape(64, 3, 3).clone())
    offsets = torch.stack(samples) - model.anchors.float()
    variance = offsets.double().var(dim=(0, 1, 3))
    kt = KB_EV * TEMPERATURE
    expected = torch.tensor([kt / 1.0, kt / 1.0, kt / 1.0], dtype=torch.float64)
    expected[1] = kt / float(model.springs[1, 1])  # species 2 on site 1
    assert torch.allclose(variance, expected, rtol=0.08)


def test_displacement_plus_swap_samples_the_exact_arrangement_distribution() -> None:
    """Off-lattice MC with swaps gets the position-entropy weighting of arrangements right."""
    model = default_model()
    numbers = [1, 1, 2]
    batch = tether_batch(model, numbers, 64, seed=1)
    moves = [
        Displacement(model, TEMPERATURE, step_size=0.5),
        NonlocalSwap(model, TEMPERATURE),
    ]
    sampler = CompositeMC(
        model, TEMPERATURE, moves, weights=[0.75, 0.25], random_seed=5
    )
    sampler.run(batch, n_steps=400)
    counts: dict[tuple[int, ...], int] = {}
    for _ in range(3000):
        sampler.step(batch)
        arrangement_counts(batch, counts)
    total = sum(counts.values())
    exact = exact_arrangement_probabilities(model, numbers, TEMPERATURE)
    for arrangement, probability in exact.items():
        assert counts.get(arrangement, 0) / total == pytest.approx(
            probability, abs=0.015
        )
    stats = sampler.move_stats
    assert stats[0].attempted + stats[1].attempted == sampler.stats.attempted
    assert 0.6 < stats[0].attempted / sampler.stats.attempted < 0.9
    assert stats[1].accepted > 0
