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
"""Tests for the alternating MC-MD block scheduler."""

from __future__ import annotations

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.hybrid import HybridMCMD
from nvalchemi.mc import SGC
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


def test_scheduler_alternates_sgc_and_md_on_the_same_batch() -> None:
    """One block advances both stages without a batch conversion."""
    model = DemoModelWrapper(DemoModel())
    data = AtomicData(
        atomic_numbers=torch.tensor([1], dtype=torch.long),
        positions=torch.zeros(1, 3),
    )
    batch = Batch.from_data_list([data])
    batch.energy = torch.zeros(1, 1)
    batch.forces = torch.zeros(1, 3)
    mc = SGC(
        model=model,
        temperature=1000.0,
        species=[1, 2],
        chemical_potentials={1: 0.0, 2: 1.0e6},
        random_seed=4,
    )
    md = DemoDynamics(model=model, n_steps=None, dt=0.01)

    result = HybridMCMD(mc=mc, md=md, mc_steps=1, md_steps=1).run(batch, n_blocks=1)

    assert result is batch
    assert mc.step_count == 1
    assert md.step_count == 1
    assert batch.atomic_numbers.tolist() == [2]
