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
"""Monte Carlo sub-stages inside a ``FusedStage`` (``mc + md``)."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import FusedStage
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.mc import SGC
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


def _batch() -> Batch:
    """Four one-atom graphs: statuses 0 (MC) and 1 (MD), two each."""
    data = [
        AtomicData(
            atomic_numbers=torch.tensor([1], dtype=torch.long),
            positions=torch.tensor([[float(i), 0.0, 0.0]]),
        )
        for i in range(4)
    ]
    batch = Batch.from_data_list(data)
    batch.forces = torch.zeros(batch.num_nodes, 3)
    batch.energy = torch.zeros(batch.num_graphs, 1)
    batch.velocities = torch.ones(batch.num_nodes, 3)
    batch.fmax = torch.ones(batch.num_graphs, 1)
    batch.status = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    return batch


def _fused(mu_2: float) -> tuple[FusedStage, SGC]:
    model = DemoModelWrapper(DemoModel())
    sgc = SGC(
        model=model,
        temperature=1000.0,
        species=[1, 2],
        chemical_potentials={1: 0.0, 2: mu_2},
        random_seed=7,
    )
    return sgc + DemoDynamics(model=model, n_steps=100, dt=1.0), sgc


@pytest.mark.parametrize(("mu_2", "mc_types"), [(1.0e6, [2, 2]), (-1.0e6, [1, 1])])
def test_mc_moves_only_its_graphs_and_energies_stay_current(mu_2, mc_types) -> None:
    """Accepted and rejected MC graphs, and the MD graphs, all end with their own energy."""
    fused, sgc = _fused(mu_2)
    batch = _batch()
    md_positions = batch.positions[2:].clone()

    for _ in range(3):
        fused.step(batch)

    assert batch.atomic_numbers.tolist() == [*mc_types, 1, 1]
    assert not torch.allclose(batch.positions[2:], md_positions)  # MD still moves
    assert sgc.stats.attempted == 6  # two MC graphs x three steps
    carried = batch.energy.reshape(-1).clone()
    torch.testing.assert_close(carried, fused.compute(batch)["energy"].reshape(-1))
