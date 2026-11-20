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
"""Parallel tempering over hybrid MC-MD blocks."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
from nvalchemi.hybrid import ParallelTemperingMCMD
from nvalchemi.mc import NonlocalSwap
from test.mc.conftest import default_model, tether_batch


def _tempering(model, temperatures, **kwargs) -> ParallelTemperingMCMD:
    temps = torch.tensor(temperatures)
    mc = NonlocalSwap(model, temps)
    md = NVTLangevin(model, dt=1.0, temperature=temps, friction=0.01)
    return ParallelTemperingMCMD(mc, md, 2, 2, temperatures, **kwargs)


def _distinct_batch(model, n_graphs: int) -> Batch:
    batch = tether_batch(model, [2, 1, 1], n_graphs)
    batch.positions += torch.arange(n_graphs).repeat_interleave(3)[:, None] * 0.01
    batch.velocities = torch.ones_like(batch.positions)
    return batch


def test_a_certain_exchange_swaps_configurations_and_rescales_velocities() -> None:
    model = default_model()
    pt = _tempering(model, [100.0, 400.0])
    batch = _distinct_batch(model, 2)
    batch.energy = torch.tensor([[5.0], [1.0]])  # cold slot hot: always accepted
    batch.atomic_numbers[3:] = torch.tensor([1, 2, 1])
    before = {
        name: getattr(batch, name).clone()
        for name in ("positions", "atomic_numbers", "energy")
    }
    perm = pt.exchange(batch)
    assert perm.tolist() == [1, 0]
    assert torch.equal(batch.positions[:3], before["positions"][3:])
    assert torch.equal(batch.atomic_numbers[:3], before["atomic_numbers"][3:])
    assert batch.energy.reshape(-1).tolist() == [1.0, 5.0]
    assert torch.allclose(
        batch.velocities[:3], torch.full((3, 3), 0.5)
    )  # sqrt(100/400)
    assert torch.allclose(batch.velocities[3:], torch.full((3, 3), 2.0))
    assert pt.replica_ids.tolist() == [1, 0]
    assert pt.acceptance_rates == [1.0]


def test_an_impossible_exchange_changes_nothing() -> None:
    model = default_model()
    pt = _tempering(model, [100.0, 400.0])
    batch = _distinct_batch(model, 2)
    batch.energy = torch.tensor([[-100.0], [100.0]])
    before = batch.positions.clone()
    pt.exchange(batch)
    assert torch.equal(batch.positions, before)
    assert pt.acceptance_rates == [0.0]


def test_pairs_alternate_between_even_and_odd_neighbours() -> None:
    model = default_model()
    pt = _tempering(model, [100.0, 200.0, 300.0])
    batch = _distinct_batch(model, 3)
    batch.energy = torch.tensor([[-100.0], [0.0], [100.0]])
    pt.exchange(batch)
    pt.exchange(batch)
    assert pt.pair_attempts == [1, 1]


def test_ladder_and_batch_are_validated() -> None:
    model = default_model()
    with pytest.raises(ValueError, match="increasing"):
        _tempering(model, [300.0, 200.0])
    pt = _tempering(model, [100.0, 200.0, 300.0])
    with pytest.raises(ValueError, match="ladder"):
        pt.exchange(_distinct_batch(model, 2))
    uneven = Batch.from_data_list(
        [
            AtomicData(
                atomic_numbers=torch.tensor([1, 2]), positions=torch.zeros(2, 3)
            ),
            AtomicData(
                atomic_numbers=torch.tensor([1, 2, 1]), positions=torch.zeros(3, 3)
            ),
        ]
    )
    uneven.energy = torch.zeros(2, 1)
    pt2 = _tempering(model, [100.0, 200.0])
    with pytest.raises(ValueError, match="same atom count"):
        pt2.exchange(uneven)


def test_hybrid_blocks_run_with_exchanges() -> None:
    model = default_model()
    pt = _tempering(model, [300.0, 600.0, 1200.0])
    batch = _distinct_batch(model, 3)
    pt.run(batch, n_blocks=4)
    assert sum(pt.pair_attempts) == 4
    assert sorted(pt.replica_ids.tolist()) == [0, 1, 2]
