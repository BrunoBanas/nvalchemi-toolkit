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
"""Base machinery shared by GPU-resident Monte Carlo samplers."""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics.base import BaseDynamics, DynamicsStage
from nvalchemi.dynamics.hooks._utils import KB_EV

if TYPE_CHECKING:
    from nvalchemi.dynamics.base import ConvergenceHook
    from nvalchemi.hooks import Hook
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["BaseMonteCarlo", "MonteCarloStats"]


@dataclass(frozen=True)
class MonteCarloStats:
    """Host-side snapshot of attempted and accepted MC moves."""

    attempted: int
    accepted: int

    @property
    def acceptance(self) -> float:
        """Return the accepted fraction, or NaN before the first attempt."""
        return self.accepted / self.attempted if self.attempted else float("nan")


class BaseMonteCarlo(BaseDynamics):
    """Base class for one independent, batched MC proposal per graph per step.

    Subclasses mutate trial atom types in :meth:`_propose` and restore rejected
    proposals in :meth:`_restore_rejected`. The model evaluates the full trial
    batch once; all acceptance arithmetic and state restoration remain on the
    batch device.
    """

    __needs_keys__: set[str] = {"energy"}
    __provides_keys__: set[str] = {"atomic_numbers", "mc_accepted"}
    _mutable_fields = (*BaseDynamics._mutable_fields, "atomic_numbers")

    def __init__(
        self,
        model: BaseModelMixin,
        temperature: float | torch.Tensor,
        random_seed: int = 420,
        n_steps: int | None = None,
        hooks: list[Hook] | None = None,
        convergence_hook: ConvergenceHook | dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize an MC sampler.

        Parameters
        ----------
        model
            Model that supplies an energy for each graph in the batch.
        temperature
            Positive scalar temperature in K, or one temperature per graph.
        random_seed
            Seed for the device-resident proposal and acceptance generator.
        n_steps
            Default number of proposals per graph for :meth:`run`.
        hooks
            Hooks dispatched at the standard dynamics stages.
        convergence_hook
            Optional Toolkit convergence hook.
        **kwargs
            Forwarded to :class:`~nvalchemi.dynamics.base.BaseDynamics`.
        """
        super().__init__(
            model=model,
            n_steps=n_steps,
            hooks=hooks,
            convergence_hook=convergence_hook,
            **kwargs,
        )
        initial_temperature = torch.as_tensor(temperature)
        if initial_temperature.ndim > 1 or torch.any(initial_temperature <= 0):
            raise ValueError("temperature must be positive and scalar or one-dimensional")
        self._temperature = temperature
        self._random_seed = random_seed
        self._generator: torch.Generator | None = None
        self._generator_device: torch.device | None = None
        self._energy: torch.Tensor | None = None
        self._energy_batch_id: int | None = None
        self._validated_batch_id: int | None = None
        self._active: torch.Tensor | None = None
        self._attempted: torch.Tensor | None = None
        self._accepted: torch.Tensor | None = None

    def _temperature_for(self, batch: Batch) -> torch.Tensor:
        """Return one positive temperature in K for every graph."""
        temperature = torch.as_tensor(
            self._temperature, dtype=batch.positions.dtype, device=batch.device
        )
        if temperature.ndim == 0:
            return temperature.expand(batch.num_graphs)
        if temperature.shape != (batch.num_graphs,):
            raise ValueError(
                "temperature tensor must have one value per graph "
                f"(expected {(batch.num_graphs,)}, got {tuple(temperature.shape)})"
            )
        return temperature

    def _active_mask(self, batch: Batch) -> torch.Tensor:
        """Return the graphs eligible to receive an MC proposal."""
        if getattr(batch, "status", None) is None:
            return torch.ones(batch.num_graphs, dtype=torch.bool, device=batch.device)
        status = batch.status.squeeze(-1) if batch.status.ndim == 2 else batch.status
        return status[: batch.num_graphs] < self.exit_status

    def _ensure_generator(self, batch: Batch) -> torch.Generator:
        """Create the persistent device generator on first use."""
        if self._generator is None or self._generator_device != batch.device:
            self._generator = torch.Generator(device=batch.device)
            self._generator.manual_seed(self._random_seed)
            self._generator_device = batch.device
        return self._generator

    def _ensure_observables(self, batch: Batch) -> None:
        """Allocate graph-level energy and acceptance storage if absent."""
        energy = getattr(batch, "energy", None)
        if energy is None:
            batch.add_system_property(
                "energy",
                torch.zeros(
                    (batch.num_graphs, 1),
                    dtype=batch.positions.dtype,
                    device=batch.device,
                ),
            )
        accepted = getattr(batch, "mc_accepted", None)
        if accepted is None:
            batch.add_system_property(
                "mc_accepted",
                torch.zeros((batch.num_graphs, 1), dtype=torch.bool, device=batch.device),
            )
        elif accepted.shape != (batch.num_graphs, 1):
            raise ValueError("batch.mc_accepted must have shape [num_graphs, 1]")

    def _model_energy(self, batch: Batch) -> torch.Tensor:
        """Evaluate the model and return its one energy value per graph."""
        outputs = self.compute(batch)
        energy = outputs["energy"]
        if energy.numel() != batch.num_graphs:
            raise ValueError("Monte Carlo requires exactly one energy per graph")
        return energy.detach().reshape(batch.num_graphs)

    def _initialize_energy(self, batch: Batch) -> None:
        """Evaluate the accepted starting configuration once."""
        if self._validated_batch_id != id(batch):
            self._validate_batch(batch)
            self._validated_batch_id = id(batch)
        self._call_hooks(DynamicsStage.BEFORE_COMPUTE, batch)
        self._energy = self._model_energy(batch)
        self._call_hooks(DynamicsStage.AFTER_COMPUTE, batch)
        self._energy_batch_id = id(batch)

    def synchronize(self, batch: Batch) -> None:
        """Adopt a trusted current energy after an external state update.

        Call this after an MD block that changes positions or the cell. The
        energy must already describe the current atom types and coordinates.
        """
        self._ensure_observables(batch)
        if self._validated_batch_id != id(batch):
            self._validate_batch(batch)
            self._validated_batch_id = id(batch)
        self._energy = batch.energy.detach().reshape(batch.num_graphs).clone()
        self._energy_batch_id = id(batch)

    def reset_statistics(self) -> None:
        """Reset cumulative attempted and accepted move counters."""
        self._attempted = None
        self._accepted = None

    @property
    def stats(self) -> MonteCarloStats:
        """Return a host-side statistics snapshot after a synchronization point."""
        attempted = 0 if self._attempted is None else int(self._attempted.item())
        accepted = 0 if self._accepted is None else int(self._accepted.item())
        return MonteCarloStats(attempted=attempted, accepted=accepted)

    @abstractmethod
    def _propose(
        self,
        batch: Batch,
        generator: torch.Generator,
        active: torch.Tensor,
    ) -> None:
        """Apply one trial proposal per active graph in-place."""

    @abstractmethod
    def _restore_rejected(self, batch: Batch, rejected: torch.Tensor) -> None:
        """Restore the trial atom types in rejected graphs."""

    def _chemical_delta(self, batch: Batch) -> torch.Tensor:
        """Return the per-graph chemical contribution to the trial potential."""
        return torch.zeros(batch.num_graphs, dtype=batch.positions.dtype, device=batch.device)

    def _validate_batch(self, batch: Batch) -> None:
        """Validate sampler-specific assumptions for a newly seen batch."""

    def pre_update(self, batch: Batch) -> None:
        """Apply one trial proposal to every active graph."""
        self._active = self._active_mask(batch)
        self._propose(batch, self._ensure_generator(batch), self._active)

    def post_update(self, batch: Batch) -> None:
        """Accept or restore proposals after the trial-energy evaluation."""
        if self._energy is None or self._active is None:
            raise RuntimeError("Monte Carlo proposal state was not initialized")
        trial_energy = batch.energy.detach().reshape(batch.num_graphs)
        delta = trial_energy - self._energy + self._chemical_delta(batch)
        log_probability = (-delta / (KB_EV * self._temperature_for(batch))).clamp(max=0.0)
        log_uniform = torch.log(
            torch.rand(batch.num_graphs, device=batch.device, generator=self._ensure_generator(batch))
        )
        accepted = self._active & (log_uniform < log_probability)
        rejected = self._active & ~accepted
        with torch.no_grad():
            self._restore_rejected(batch, rejected)
            self._energy = torch.where(accepted, trial_energy, self._energy)
            batch.energy.copy_(self._energy.reshape_as(batch.energy))
            batch.mc_accepted.copy_(accepted.reshape_as(batch.mc_accepted))
            if self._attempted is None:
                self._attempted = self._active.sum().detach()
                self._accepted = accepted.sum().detach()
            else:
                self._attempted += self._active.sum().detach()
                self._accepted += accepted.sum().detach()

    def step(self, batch: Batch) -> tuple[Batch, torch.Tensor | None]:
        """Run one complete, hook-aware MC proposal and acceptance step."""
        self._ensure_state_initialized(batch)
        self._ensure_observables(batch)
        self._call_hooks(DynamicsStage.BEFORE_STEP, batch)
        with self._stream_scope(batch.device):
            if self._energy is None or self._energy_batch_id != id(batch):
                self._initialize_energy(batch)
            self._call_hooks(DynamicsStage.BEFORE_PRE_UPDATE, batch)
            self.pre_update(batch)
            self._call_hooks(DynamicsStage.AFTER_PRE_UPDATE, batch)
            self._call_hooks(DynamicsStage.BEFORE_COMPUTE, batch)
            self._model_energy(batch)
            self._call_hooks(DynamicsStage.AFTER_COMPUTE, batch)
            self._call_hooks(DynamicsStage.BEFORE_POST_UPDATE, batch)
            self.post_update(batch)
            self._call_hooks(DynamicsStage.AFTER_POST_UPDATE, batch)
        self._call_hooks(DynamicsStage.AFTER_STEP, batch)
        converged = self._check_convergence(batch)
        self._last_converged = converged
        if converged is not None:
            self._call_hooks(DynamicsStage.ON_CONVERGE, batch)
        self.step_count += 1
        return batch, converged
