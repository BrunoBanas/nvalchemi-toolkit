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
"""Unit tests for canonical nearest-neighbour Kawasaki Monte Carlo."""

from __future__ import annotations

from collections import Counter
from typing import Any

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.mc import Kawasaki
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


def _pair(first: int, second: int, *, separation: float = 1.0) -> AtomicData:
    """One two-atom graph with the given species, a fixed distance apart."""
    return AtomicData(
        atomic_numbers=torch.tensor([first, second], dtype=torch.long),
        positions=torch.tensor([[0.0, 0.0, 0.0], [separation, 0.0, 0.0]]),
    )


def _sampler(cutoff: float = 2.0, **kwargs: Any) -> Kawasaki:
    return Kawasaki(
        model=DemoModelWrapper(DemoModel()),
        temperature=1000.0,
        cutoff=cutoff,
        random_seed=7,
        **kwargs,
    )


def test_composition_is_conserved_for_unlike_neighbors() -> None:
    """Whether accepted or rejected, a swap never changes the species multiset."""
    batch = Batch.from_data_list([_pair(1, 2)])
    batch.energy = torch.zeros(batch.num_graphs, 1)
    sampler = _sampler()

    sampler.run(batch, n_steps=1)

    assert Counter(batch.atomic_numbers.tolist()) == Counter([1, 2])
    assert sampler.stats.attempted == 1


def test_same_species_neighbors_are_a_no_op() -> None:
    """A like-species draw leaves atom types unchanged and is trivially accepted."""
    batch = Batch.from_data_list([_pair(1, 1)])
    batch.energy = torch.zeros(batch.num_graphs, 1)
    sampler = _sampler()

    sampler.run(batch, n_steps=1)

    assert batch.atomic_numbers.tolist() == [1, 1]
    assert bool(batch.mc_accepted.reshape(-1)[0])


def test_inactive_graph_is_not_mutated() -> None:
    """A graduated graph is excluded from proposals and acceptance statistics."""
    batch = Batch.from_data_list([_pair(1, 2), _pair(1, 2)])
    batch.energy = torch.zeros(batch.num_graphs, 1)
    batch.status = torch.tensor([[0], [1]], dtype=torch.long)
    sampler = _sampler()

    sampler.run(batch, n_steps=1)

    second_graph = batch.atomic_numbers[batch.batch_ptr[1] : batch.batch_ptr[2]]
    assert second_graph.tolist() == [1, 2]
    assert sampler.stats.attempted == 1


def test_cutoff_must_be_positive() -> None:
    """A non-positive proposal cutoff is rejected at construction time."""
    with pytest.raises(ValueError, match="cutoff must be positive"):
        _sampler(cutoff=0.0)


def test_missing_neighbor_pairs_raise() -> None:
    """A cutoff shorter than the only interatomic distance leaves no proposal edge."""
    batch = Batch.from_data_list([_pair(1, 2, separation=5.0)])
    batch.energy = torch.zeros(batch.num_graphs, 1)
    sampler = _sampler(cutoff=1.0)

    with pytest.raises(ValueError, match="no neighbour pairs"):
        sampler.run(batch, n_steps=1)


def test_synchronize_rebuilds_the_proposal_graph() -> None:
    """After positions move outside the cutoff, synchronize must refresh the graph."""
    batch = Batch.from_data_list([_pair(1, 2, separation=1.0)])
    batch.energy = torch.zeros(batch.num_graphs, 1)
    sampler = _sampler(cutoff=2.0)
    sampler.run(batch, n_steps=1)

    with torch.no_grad():
        batch.positions[1, 0] = 10.0

    with pytest.raises(ValueError, match="no neighbour pairs"):
        sampler.synchronize(batch)


def test_composition_is_conserved_over_many_steps() -> None:
    """A longer run on a larger periodic system never changes the species counts."""
    numbers = [1, 2, 1, 2, 1, 2, 1, 2]
    positions = torch.tensor(
        [[float(index), 0.0, 0.0] for index in range(len(numbers))]
    )
    data = AtomicData(
        atomic_numbers=torch.tensor(numbers, dtype=torch.long),
        positions=positions,
        cell=torch.eye(3).unsqueeze(0) * 20.0,
        pbc=torch.tensor([[True, True, True]]),
    )
    batch = Batch.from_data_list([data])
    batch.energy = torch.zeros(batch.num_graphs, 1)
    sampler = _sampler(cutoff=1.5)

    sampler.run(batch, n_steps=20)

    assert Counter(batch.atomic_numbers.tolist()) == Counter(numbers)
