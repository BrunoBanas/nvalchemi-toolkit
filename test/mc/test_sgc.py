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
"""Unit tests for semi-grand-canonical Monte Carlo."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.mc import SGC
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


def _batch(numbers: list[list[int]]) -> Batch:
    """Create one small graph per supplied sequence of atomic numbers."""
    data = [
        AtomicData(
            atomic_numbers=torch.tensor(values, dtype=torch.long),
            positions=torch.zeros(len(values), 3),
        )
        for values in numbers
    ]
    batch = Batch.from_data_list(data)
    batch.energy = torch.zeros(batch.num_graphs, 1)
    return batch


def _sampler(chemical_potentials: dict[int, float]) -> SGC:
    """Construct a deterministic test sampler around the demo model."""
    return SGC(
        model=DemoModelWrapper(DemoModel()),
        temperature=1000.0,
        species=[1, 2],
        chemical_potentials=chemical_potentials,
        random_seed=7,
    )


def test_higher_chemical_potential_favours_insertion() -> None:
    """A large positive mu for species 2 accepts the 1-to-2 transmutation."""
    batch = _batch([[1]])
    sampler = _sampler({1: 0.0, 2: 1.0e6})

    sampler.run(batch, n_steps=1)

    assert batch.atomic_numbers.tolist() == [2]
    assert batch.mc_accepted.tolist() == [[True]]
    assert sampler.stats.attempted == 1
    assert sampler.stats.accepted == 1


def test_unfavourable_transmutation_restores_the_accepted_type() -> None:
    """A large negative mu for species 2 rejects and restores the old type."""
    batch = _batch([[1]])
    sampler = _sampler({1: 0.0, 2: -1.0e6})

    sampler.run(batch, n_steps=1)

    assert batch.atomic_numbers.tolist() == [1]
    assert batch.mc_accepted.tolist() == [[False]]
    assert sampler.stats.attempted == 1
    assert sampler.stats.accepted == 0


def test_inactive_graph_is_not_mutated() -> None:
    """A graduated graph is excluded from proposals and acceptance statistics."""
    batch = _batch([[1], [1]])
    batch.status = torch.tensor([[0], [1]], dtype=torch.long)
    sampler = _sampler({1: 0.0, 2: 1.0e6})

    sampler.run(batch, n_steps=1)

    assert batch.atomic_numbers.tolist() == [2, 1]
    assert batch.mc_accepted.tolist() == [[True], [False]]
    assert sampler.stats.attempted == 1


def test_per_graph_chemical_potentials_sample_different_reservoirs() -> None:
    """One SGC batch can carry a distinct reservoir for every graph."""
    batch = _batch([[1], [1]])
    sampler = SGC(
        model=DemoModelWrapper(DemoModel()),
        temperature=torch.tensor([1000.0, 1000.0]),
        species=[1, 2],
        chemical_potentials={1: 0.0, 2: torch.tensor([1.0e6, -1.0e6])},
        random_seed=7,
    )

    sampler.run(batch, n_steps=1)

    assert batch.atomic_numbers.tolist() == [2, 1]
    assert batch.mc_accepted.tolist() == [[True], [False]]


def test_per_graph_chemical_potential_shape_is_checked() -> None:
    """A reservoir tensor must match the number of active graphs exactly."""
    batch = _batch([[1], [1]])
    sampler = SGC(
        model=DemoModelWrapper(DemoModel()),
        temperature=1000.0,
        species=[1, 2],
        chemical_potentials={1: 0.0, 2: torch.tensor([0.0])},
    )

    with pytest.raises(ValueError, match="chemical-potential tensors"):
        sampler.run(batch, n_steps=1)


def test_rejects_missing_reservoir_species() -> None:
    """The starting composition must be representable by the reservoir."""
    batch = _batch([[3]])
    sampler = _sampler({1: 0.0, 2: 0.0})

    with pytest.raises(ValueError, match="outside the configured species"):
        sampler.run(batch, n_steps=1)
