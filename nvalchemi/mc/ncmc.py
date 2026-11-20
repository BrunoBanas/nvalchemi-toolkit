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
"""Nonequilibrium candidate Monte Carlo (NCMC) for alchemical species swaps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics._units import fs_to_internal_time
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.mc.base import _is_batch
from nvalchemi.mc.swap import NonlocalSwap

if TYPE_CHECKING:
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["NCMCSwap"]


class NCMCSwap(NonlocalSwap):
    r"""Non-local swaps switched in alchemically over a short NVE trajectory (NCMC).

    A trial picks one unlike pair (as :class:`~nvalchemi.mc.NonlocalSwap`),
    draws auxiliary Maxwell--Boltzmann momenta, and drives the Hamiltonian

    .. math::

        H_\lambda = (1-\lambda)\,U(x; Z_A) + \lambda\,U(x; Z_B) + K(p)

    from the current species :math:`Z_A` to the swapped species :math:`Z_B`
    along the palindromic protocol of Nilmeier, Crooks, Minh & Chodera (PNAS
    108, E1009 (2011)): perturb :math:`\lambda` by :math:`1/n`, take one
    velocity-Verlet step at fixed :math:`\lambda`, and repeat, ending on a
    perturbation (``n_switch`` perturbations, ``n_switch - 1`` steps). The
    neighbours relax while the swapped atoms change identity. The trial is
    accepted with :math:`\min(1, e^{-\beta [H_1 - H_0]})`; because each
    velocity-Verlet step is volume-preserving and time-reversible and the
    reverse protocol is the time reverse of the forward one, this total
    energy change (protocol work plus integrator shadow work) makes the
    move exact for the canonical distribution of positions and species at
    any ``dt``. ``n_switch=1`` is a plain instantaneous swap.

    The switching trajectory must use the same mass matrix as its reverse, so
    the two swapped atoms carry the mean of their species masses while
    switching (all other atoms their own). The auxiliary momenta are
    discarded: the batch's own velocities are left
    as they were, except that atoms whose species changed get fresh
    Maxwell--Boltzmann velocities for their new mass. A rejected trial gets
    its species, positions, energy and forces back.

    Cost per trial: ``2 * n_switch`` energy-and-force calls (both species
    assignments at every configuration). The model must return forces
    (:class:`~nvalchemi.hybrid.HybridMCMD` with ``mc_energy_only=False``) and
    the batch must carry ``atomic_masses``. A MoLE-merged UMA is valid
    because a swap never changes the composition.
    """

    def __init__(
        self,
        model: BaseModelMixin,
        temperature: float | torch.Tensor,
        n_switch: int = 10,
        dt: float = 2.0,
        **kwargs: Any,
    ) -> None:
        """Initialize an NCMC swap sampler.

        Parameters
        ----------
        model
            Model that returns energies and forces.
        temperature
            Positive scalar temperature in K, or one temperature per graph.
        n_switch
            Number of :math:`\\lambda` perturbations (``>= 1``).
        dt
            Velocity-Verlet time step of the switching trajectory, in fs.
        **kwargs
            Forwarded to :class:`~nvalchemi.mc.NonlocalSwap`.
        """
        super().__init__(model=model, temperature=temperature, **kwargs)
        if n_switch < 1:
            raise ValueError("n_switch must be at least 1")
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.n_switch = int(n_switch)
        self.dt = float(dt)
        self._work: torch.Tensor | None = None

    def _validate_batch(self, batch: Batch) -> None:
        """Require forces from the model and masses on the batch."""
        if "forces" not in self.model.model_config.active_outputs:
            raise ValueError(
                "NCMCSwap needs forces: keep 'forces' in the model's "
                "active_outputs (HybridMCMD mc_energy_only=False)"
            )
        if getattr(batch, "atomic_masses", None) is None:
            raise ValueError("NCMCSwap needs batch.atomic_masses")

    def _evaluate(
        self, batch: Batch, numbers: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Energy per graph and forces with the given species, at the current positions."""
        with torch.no_grad():
            batch.atomic_numbers.reshape(-1).copy_(numbers)
        outputs = self.compute(batch)
        energy = outputs["energy"].detach().reshape(batch.num_graphs).clone()
        return energy, outputs["forces"].detach().clone()

    def _kinetic(
        self, velocity: torch.Tensor, mass: torch.Tensor, graph: torch.Tensor, n: int
    ) -> torch.Tensor:
        per_atom = 0.5 * (mass * velocity.square()).sum(dim=1)
        return torch.zeros(n, dtype=per_atom.dtype, device=per_atom.device).index_add_(
            0, graph, per_atom
        )

    @property
    def last_work(self) -> torch.Tensor | None:
        """Total work :math:`H_1 - H_0` (eV) of the last trial, one per graph."""
        return self._work

    def step(self, batch: Batch) -> tuple[Batch, torch.Tensor | None]:
        """Run one NCMC swap trial per active graph."""
        self._ensure_state_initialized(batch)
        self._ensure_observables(batch)
        self._call_hooks(DynamicsStage.BEFORE_STEP, batch)
        with self._stream_scope(batch.device):
            if self._energy is None or not _is_batch(self._energy_batch, batch):
                self._initialize_energy(batch)
            self._ensure_mass_table(batch)
            self._active = self._active_mask(batch)
            generator = self._ensure_generator(batch)
            with torch.no_grad():
                self._propose(batch, generator, self._active)
            swapped = self._proposal_swapped
            flat = batch.atomic_numbers.reshape(-1)
            numbers_b = flat.detach().clone()
            numbers_a = numbers_b.clone()
            first = self._proposal_first[swapped]
            second = self._proposal_second[swapped]
            numbers_a[first] = numbers_b[second]
            numbers_a[second] = numbers_b[first]

            positions = batch.positions
            start = positions.detach().clone()
            graph = batch.batch_idx.to(torch.long)
            n_graphs = batch.num_graphs
            temperature = self._temperature_for(batch)
            kt = (KB_EV * temperature)[graph, None]
            # The switching trajectory needs one mass matrix for the forward and the
            # reverse move; species masses differ, so both swapped atoms carry the
            # mean of their two masses (any swap-symmetric choice would do).
            mass = (
                batch.atomic_masses.detach().reshape(-1, 1).to(positions.dtype).clone()
            )
            pair_mass = 0.5 * (mass[first] + mass[second])
            mass[first] = pair_mass
            mass[second] = pair_mass
            dt = float(fs_to_internal_time(self.dt))
            velocity = torch.randn(
                positions.shape,
                device=batch.device,
                dtype=positions.dtype,
                generator=generator,
            ) * torch.sqrt(kt / mass)

            energy_a, forces_a = self._evaluate(batch, numbers_a)
            start_forces = forces_a
            energy_b, forces_b = self._evaluate(batch, numbers_b)
            h_start = energy_a + self._kinetic(velocity, mass, graph, n_graphs)
            for k in range(1, self.n_switch):
                lam = k / self.n_switch
                velocity = (
                    velocity + 0.5 * dt * ((1 - lam) * forces_a + lam * forces_b) / mass
                )
                with torch.no_grad():
                    positions.add_(dt * velocity)
                energy_a, forces_a = self._evaluate(batch, numbers_a)
                energy_b, forces_b = self._evaluate(batch, numbers_b)
                velocity = (
                    velocity + 0.5 * dt * ((1 - lam) * forces_a + lam * forces_b) / mass
                )
            h_end = energy_b + self._kinetic(velocity, mass, graph, n_graphs)
            work = h_end - h_start
            self._work = work
            log_probability = (-work / (KB_EV * temperature)).clamp(max=0.0)
            log_uniform = torch.log(
                torch.rand(
                    n_graphs, device=batch.device, generator=generator, dtype=work.dtype
                )
            )
            accepted = swapped & (log_uniform < log_probability)
            rejected = self._active & ~accepted
            with torch.no_grad():
                # The species on the batch are the swapped ones (last evaluation).
                self._restore_rejected(batch, rejected)
                # Every graph that did not accept -- inactive ones too, which were
                # integrated alongside -- gets its starting positions back.
                undo = (~accepted)[graph, None]
                positions.copy_(torch.where(undo, start, positions))
                forces = getattr(batch, "forces", None)
                if forces is not None:
                    forces.copy_(torch.where(undo, start_forces, forces_b))
                self._energy = torch.where(accepted, energy_b, self._energy)
                batch.energy.copy_(self._energy.reshape_as(batch.energy))
                batch.mc_accepted.copy_(accepted.reshape_as(batch.mc_accepted))
                if self._attempted is None:
                    self._attempted = self._active.sum().detach()
                    self._accepted = accepted.sum().detach()
                else:
                    self._attempted += self._active.sum().detach()
                    self._accepted += accepted.sum().detach()
            self._sync_masses(batch)
            velocities = getattr(batch, "velocities", None)
            if velocities is not None:
                changed = torch.zeros(
                    flat.shape[0], dtype=torch.bool, device=batch.device
                )
                changed[self._proposal_first] = accepted
                changed[self._proposal_second] |= accepted
                new_mass = (
                    batch.atomic_masses.detach().reshape(-1, 1).to(velocities.dtype)
                )
                fresh = torch.randn(
                    velocities.shape,
                    device=batch.device,
                    dtype=velocities.dtype,
                    generator=generator,
                ) * torch.sqrt(kt.to(velocities.dtype) / new_mass)
                with torch.no_grad():
                    velocities.copy_(torch.where(changed[:, None], fresh, velocities))
        self._call_hooks(DynamicsStage.AFTER_STEP, batch)
        self.step_count += 1
        return batch, None
