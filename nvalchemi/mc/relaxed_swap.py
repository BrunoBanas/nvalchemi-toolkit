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
"""Non-local swaps accepted on locally relaxed energies (biased, not Boltzmann)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.mc.base import _is_batch
from nvalchemi.mc.swap import NonlocalSwap

if TYPE_CHECKING:
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["RelaxedSwap", "fire_relax"]


def _per_graph_sum(
    values: torch.Tensor, graph: torch.Tensor, n_graphs: int
) -> torch.Tensor:
    return torch.zeros(n_graphs, dtype=values.dtype, device=values.device).index_add_(
        0, graph, values
    )


def fire_relax(
    batch: Batch,
    evaluate: Any,
    movable: torch.Tensor,
    n_steps: int,
    *,
    dt_start: float = 0.1,
    dt_max: float = 1.0,
    max_step: float = 0.2,
    n_min: int = 5,
    alpha_start: float = 0.1,
) -> torch.Tensor:
    """Relax ``batch.positions`` in place with masked, batched FIRE (unit masses).

    Parameters
    ----------
    batch
        Batch whose positions are relaxed in place.
    evaluate
        ``evaluate(batch) -> (energy [B], forces [V, 3])`` at the current positions.
    movable
        ``[V]`` mask of atoms allowed to move; the others are frozen.
    n_steps
        FIRE iterations; each costs one ``evaluate`` call.
    dt_start, dt_max, max_step, n_min, alpha_start
        FIRE parameters (Bitzek et al., PRL 97, 170201 (2006)), ASE defaults;
        ``max_step`` caps each atom's displacement per iteration in Å.

    Returns
    -------
    torch.Tensor
        Energy per graph at the final positions, shape ``[B]``.
    """
    energy, forces = evaluate(batch)
    if n_steps <= 0:
        return energy
    positions = batch.positions
    graph = batch.batch_idx.to(torch.long)
    n_graphs = batch.num_graphs
    mask = movable[:, None].to(positions.dtype)
    velocity = torch.zeros_like(positions)
    dt = torch.full(
        (n_graphs,), dt_start, dtype=positions.dtype, device=positions.device
    )
    alpha = torch.full_like(dt, alpha_start)
    n_positive = torch.zeros(n_graphs, dtype=torch.long, device=positions.device)
    for _ in range(n_steps):
        force = forces * mask
        power = _per_graph_sum((force * velocity).sum(dim=1), graph, n_graphs)
        v_norm = _per_graph_sum(velocity.square().sum(dim=1), graph, n_graphs).sqrt()
        f_norm = _per_graph_sum(force.square().sum(dim=1), graph, n_graphs).sqrt()
        mix = alpha * v_norm / f_norm.clamp(min=1e-12)
        velocity = (1.0 - alpha[graph, None]) * velocity + mix[graph, None] * force
        downhill = power > 0
        n_positive = torch.where(downhill, n_positive + 1, torch.zeros_like(n_positive))
        grow = downhill & (n_positive > n_min)
        dt = torch.where(
            grow, (dt * 1.1).clamp(max=dt_max), torch.where(downhill, dt, dt * 0.5)
        )
        alpha = torch.where(
            grow,
            alpha * 0.99,
            torch.where(downhill, alpha, torch.full_like(alpha, alpha_start)),
        )
        velocity = torch.where(
            downhill[graph, None], velocity, torch.zeros_like(velocity)
        )
        velocity = velocity + dt[graph, None] * force
        step = dt[graph, None] * velocity
        length = step.norm(dim=1, keepdim=True)
        step = step * (max_step / length.clamp(min=max_step))
        with torch.no_grad():
            positions.add_(step * mask)
        energy, forces = evaluate(batch)
    return energy


class RelaxedSwap(NonlocalSwap):
    r"""Non-local swaps judged on locally relaxed energies: fast, but **not** Boltzmann.

    Each trial swaps one random unlike pair (as
    :class:`~nvalchemi.mc.NonlocalSwap`), relaxes the atoms within
    ``relax_radius`` of either swapped site with ``relax_steps`` FIRE
    iterations (everything else frozen), and applies the Metropolis test to

    * ``relax_baseline=True`` (default): the relaxed trial energy minus the
      energy of the *current* species relaxed the same way around the same
      pair -- a Metropolis walk on the local inherent-structure energies;
    * ``relax_baseline=False``: the relaxed trial energy minus the current,
      unrelaxed energy -- half the cost, but every trial gets the relaxation
      bonus, so acceptance is inflated further.

    An accepted trial keeps its relaxed positions; a rejected one gets its
    species and positions back. Neither variant satisfies detailed balance
    with respect to the Boltzmann distribution at ``temperature``: relaxation
    removes the strain penalty a swap of mismatched atoms pays at fixed
    positions, so swaps are accepted far more often, at the price of sampling
    a different (inherent-structure-like) distribution. Use it to pre-order a
    system quickly, and measure the bias against an exact sampler.

    Cost per trial: ``relax_steps + 1`` energy-and-force calls, twice with
    ``relax_baseline``. The model must return forces (run a
    :class:`~nvalchemi.hybrid.HybridMCMD` with ``mc_energy_only=False``). The
    region uses plain Cartesian distances: meant for free clusters or
    ``relax_radius`` well below half the cell. After a step, ``batch.forces``
    of rejected graphs describe the last relaxed trial, not the restored
    state; ``HybridMCMD`` re-evaluates forces before MD.
    """

    def __init__(
        self,
        model: BaseModelMixin,
        temperature: float | torch.Tensor,
        relax_steps: int = 5,
        relax_radius: float | None = 6.0,
        relax_baseline: bool = True,
        max_step: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a relaxed-swap sampler.

        Parameters
        ----------
        model
            Model that returns energies and forces.
        temperature
            Positive scalar temperature in K, or one temperature per graph.
        relax_steps
            FIRE iterations per relaxation (0 = plain swap).
        relax_radius
            Atoms within this distance (Å) of either swapped atom move; ``None``
            relaxes the whole graph.
        relax_baseline
            Relax the current species around the same pair as the reference
            energy (see class notes).
        max_step
            Per-atom displacement cap per FIRE iteration (Å).
        **kwargs
            Forwarded to :class:`~nvalchemi.mc.NonlocalSwap`.
        """
        super().__init__(model=model, temperature=temperature, **kwargs)
        if relax_steps < 0:
            raise ValueError("relax_steps must be non-negative")
        if relax_radius is not None and relax_radius <= 0:
            raise ValueError("relax_radius must be positive or None")
        self.relax_steps = int(relax_steps)
        self.relax_radius = relax_radius
        self.relax_baseline = relax_baseline
        self.max_step = float(max_step)

    def _validate_batch(self, batch: Batch) -> None:
        """Require forces from the model."""
        if "forces" not in self.model.model_config.active_outputs:
            raise ValueError(
                "RelaxedSwap needs forces: keep 'forces' in the model's "
                "active_outputs (HybridMCMD mc_energy_only=False)"
            )

    def _evaluate(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.compute(batch)
        energy = outputs["energy"].detach().reshape(batch.num_graphs).clone()
        return energy, outputs["forces"].detach().clone()

    def _region(self, batch: Batch, swapped: torch.Tensor) -> torch.Tensor:
        """Atoms that may move: near either swapped atom, in graphs that swapped."""
        graph = batch.batch_idx.to(torch.long)
        in_trial = swapped[graph]
        if self.relax_radius is None:
            return in_trial
        positions = batch.positions.detach()
        first = positions[self._proposal_first][graph]
        second = positions[self._proposal_second][graph]
        distance = torch.minimum(
            (positions - first).norm(dim=1), (positions - second).norm(dim=1)
        )
        return in_trial & (distance <= self.relax_radius)

    def _swap_pairs(self, batch: Batch, which: torch.Tensor) -> None:
        """Exchange the species of the proposed pairs in graphs ``which``."""
        flat = batch.atomic_numbers.reshape(-1)
        first = self._proposal_first[which]
        second = self._proposal_second[which]
        first_z = flat[first].clone()
        flat[first] = flat[second]
        flat[second] = first_z

    def step(self, batch: Batch) -> tuple[Batch, torch.Tensor | None]:
        """Propose, relax, and accept or restore one swap per active graph."""
        self._ensure_state_initialized(batch)
        self._ensure_observables(batch)
        self._call_hooks(DynamicsStage.BEFORE_STEP, batch)
        with self._stream_scope(batch.device):
            if self._energy is None or not _is_batch(self._energy_batch, batch):
                self._initialize_energy(batch)
            self._ensure_mass_table(batch)
            self._active = self._active_mask(batch)
            with torch.no_grad():
                self._propose(batch, self._ensure_generator(batch), self._active)
            swapped = self._proposal_swapped
            start = batch.positions.detach().clone()
            movable = self._region(batch, swapped)
            if self.relax_baseline:
                with torch.no_grad():
                    self._swap_pairs(batch, swapped)  # back to the current species
                reference = fire_relax(
                    batch,
                    self._evaluate,
                    movable,
                    self.relax_steps,
                    max_step=self.max_step,
                )
                with torch.no_grad():
                    batch.positions.copy_(start)
                    self._swap_pairs(batch, swapped)
            else:
                reference = self._energy
            trial = fire_relax(
                batch, self._evaluate, movable, self.relax_steps, max_step=self.max_step
            )
            log_probability = (
                -(trial - reference) / (KB_EV * self._temperature_for(batch))
            ).clamp(max=0.0)
            log_uniform = torch.log(
                torch.rand(
                    batch.num_graphs,
                    device=batch.device,
                    generator=self._ensure_generator(batch),
                    dtype=log_probability.dtype,
                )
            )
            accepted = swapped & (log_uniform < log_probability)
            rejected = self._active & ~accepted
            with torch.no_grad():
                self._restore_rejected(batch, rejected)
                undo = rejected[batch.batch_idx.to(torch.long), None]
                batch.positions.copy_(torch.where(undo, start, batch.positions))
                self._energy = torch.where(accepted, trial, self._energy)
                batch.energy.copy_(self._energy.reshape_as(batch.energy))
                batch.mc_accepted.copy_(accepted.reshape_as(batch.mc_accepted))
                if self._attempted is None:
                    self._attempted = self._active.sum().detach()
                    self._accepted = accepted.sum().detach()
                else:
                    self._attempted += self._active.sum().detach()
                    self._accepted += accepted.sum().detach()
            self._sync_masses(batch)
        self._call_hooks(DynamicsStage.AFTER_STEP, batch)
        self.step_count += 1
        return batch, None
