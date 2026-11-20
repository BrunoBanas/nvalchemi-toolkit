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
"""Collective force-bias MC (tfMC and its Metropolized form)."""

from __future__ import annotations

import math

import pytest
import torch

from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.mc import ForceBiasMC
from nvalchemi.mc.force_bias import log_sinhc, sample_tilted_uniform

from .conftest import default_model, tether_batch

TEMPERATURE = 1000.0


@pytest.mark.parametrize("tilt", [-40.0, -2.0, 0.0, 1e-9, 0.7, 25.0])
def test_tilted_uniform_matches_the_analytic_mean_and_stays_in_bounds(
    tilt: float,
) -> None:
    half_width = 0.2
    generator = torch.Generator().manual_seed(4)
    uniform = torch.rand(200_000, generator=generator, dtype=torch.float64)
    a = torch.full_like(uniform, tilt)
    xi = sample_tilted_uniform(
        a, torch.tensor(half_width, dtype=torch.float64), uniform
    )
    assert bool(torch.all(xi.abs() <= half_width + 1e-12))
    x = tilt * half_width
    expected = 0.0 if abs(x) < 1e-6 else half_width / math.tanh(x) - 1.0 / tilt
    assert float(xi.mean()) == pytest.approx(expected, abs=2e-3)


def test_log_sinhc_is_accurate_and_finite() -> None:
    x = torch.tensor([0.0, 1e-6, 0.3, -2.0, 50.0, -800.0], dtype=torch.float64)
    value = log_sinhc(x)
    assert bool(torch.isfinite(value).all())
    for xi, vi in zip(x.tolist()[2:4], value.tolist()[2:4], strict=True):
        assert vi == pytest.approx(math.log(math.sinh(xi) / xi), rel=1e-12)
    assert float(value[4]) == pytest.approx(
        50.0 - math.log(2.0) - math.log(50.0), rel=1e-12
    )
    assert float(value[0]) == 0.0


def _position_variance(sampler: ForceBiasMC, n_graphs: int, steps: int) -> torch.Tensor:
    model = sampler.model
    batch = tether_batch(model, [1, 2, 1], n_graphs, seed=3)
    sampler.compute(batch)
    sampler.run(batch, n_steps=200)
    samples = []
    for _ in range(steps):
        sampler.step(batch)
        samples.append(batch.positions.reshape(n_graphs, 3, 3).clone())
    offsets = torch.stack(samples).double() - model.anchors
    return offsets.var(dim=(0, 1, 3))


def _expected_variance() -> torch.Tensor:
    kt = KB_EV * TEMPERATURE
    return torch.tensor([kt / 1.0, kt / 1.0, kt / 1.0], dtype=torch.float64)


def test_metropolized_force_bias_is_exact_at_a_large_step() -> None:
    model = default_model()
    sampler = ForceBiasMC(
        model, TEMPERATURE, max_displacement=0.4, metropolize=True, random_seed=8
    )
    variance = _position_variance(sampler, 64, 1500)
    assert torch.allclose(variance, _expected_variance(), rtol=0.08)
    assert 0.1 < sampler.stats.acceptance < 0.99


def test_tfmc_is_close_at_a_small_step() -> None:
    model = default_model()
    sampler = ForceBiasMC(model, TEMPERATURE, max_displacement=0.05, random_seed=9)
    variance = _position_variance(sampler, 64, 3000)
    assert torch.allclose(variance, _expected_variance(), rtol=0.15)
    assert sampler.stats.attempted == 0


def test_rejection_restores_positions_energy_and_forces() -> None:
    model = default_model()
    batch = tether_batch(model, [1, 2, 1], 3)
    sampler = ForceBiasMC(model, 1e-3, max_displacement=0.5, metropolize=True)
    sampler.compute(batch)
    before = (batch.positions.clone(), batch.energy.clone(), batch.forces.clone())
    sampler.step(batch)  # at 1 mK every uphill move is rejected
    assert sampler.stats.accepted == 0
    assert torch.equal(batch.positions, before[0])
    assert torch.equal(batch.energy, before[1])
    assert torch.equal(batch.forces, before[2])


def test_drift_removal_is_tfmc_only_and_zeroes_the_mass_weighted_shift() -> None:
    model = default_model()
    with pytest.raises(ValueError, match="remove_drift"):
        ForceBiasMC(model, TEMPERATURE, metropolize=True, remove_drift=True)
    batch = tether_batch(model, [1, 2, 1], 2)
    sampler = ForceBiasMC(model, TEMPERATURE, max_displacement=0.2, remove_drift=True)
    sampler.compute(batch)
    before = batch.positions.clone()
    sampler.pre_update(batch)
    shift = (batch.positions - before).double()
    masses = batch.atomic_masses.double().reshape(-1, 1)
    weighted = (masses * shift).reshape(2, 3, 3).sum(dim=1)
    assert torch.allclose(weighted, torch.zeros_like(weighted), atol=1e-6)
