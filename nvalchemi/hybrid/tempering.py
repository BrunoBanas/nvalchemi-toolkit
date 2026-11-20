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
"""Parallel tempering (temperature replica exchange) over hybrid MC-MD blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.dynamics.hooks.swap import even_odd_pairs
from nvalchemi.hybrid.scheduler import HybridMCMD

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from nvalchemi.dynamics.base import BaseDynamics
    from nvalchemi.mc.base import BaseMonteCarlo

__all__ = ["ParallelTemperingMCMD"]

# Per-atom state that belongs to a configuration and travels with it.
_ATOM_FIELDS = ("positions", "velocities", "forces", "atomic_numbers", "atomic_masses")


class ParallelTemperingMCMD(HybridMCMD):
    r"""Hybrid MC-MD on a temperature ladder with replica exchange between blocks.

    Graph ``s`` of the batch is the slot at ``temperatures[s]`` (ascending);
    the MC sampler and the MD integrator must be built with that same
    per-graph temperature tensor. At the start of every
    ``exchange_every``-th MC block, neighbouring slots on alternating even/odd
    pairs (:func:`~nvalchemi.dynamics.hooks.swap.even_odd_pairs`) swap
    configurations with probability

    .. math::

        \min\bigl(1, \exp[(\beta_i - \beta_j)(U_i - U_j)]\bigr),

    using the potential energies left by the previous MD block.

    Configurations move, temperatures do not: positions, velocities (rescaled
    by :math:`\sqrt{T_\mathrm{new}/T_\mathrm{old}}`), forces, species, masses
    and energy are permuted between graph rows, so each sampler keeps its
    per-graph temperature tensor and each thermostat its own chain state, and
    graph 0 is always the coldest (target) replica -- the one to analyse.
    This differs from :class:`~nvalchemi.enhanced_sampling.ReplicaExchange`,
    which moves temperature labels between walkers through
    ``apply_per_system_params``; copying a few thousand atoms per exchange is
    negligible next to one model call. All graphs must have the same number
    of atoms.

    :attr:`replica_ids` follows each lineage (which starting configuration
    is in each slot), for round-trip diagnostics.
    """

    def __init__(
        self,
        mc: BaseMonteCarlo,
        md: BaseDynamics,
        mc_steps: int,
        md_steps: int,
        temperatures: Sequence[float] | torch.Tensor,
        exchange_every: int = 1,
        random_seed: int = 1234,
        mc_energy_only: bool = False,
        before_md_block: Callable[[Batch], None] | None = None,
    ) -> None:
        """Initialize the tempering scheduler.

        Parameters
        ----------
        mc, md, mc_steps, md_steps, mc_energy_only, before_md_block
            As for :class:`~nvalchemi.hybrid.HybridMCMD`.
        temperatures
            Strictly increasing ladder in K, one per graph.
        exchange_every
            Attempt exchanges before every this-many MC blocks.
        random_seed
            Seed of the exchange acceptance draws.
        """
        super().__init__(
            mc=mc,
            md=md,
            mc_steps=mc_steps,
            md_steps=md_steps,
            mc_energy_only=mc_energy_only,
            before_md_block=before_md_block,
        )
        ladder = torch.as_tensor(temperatures, dtype=torch.float64).detach().cpu()
        if ladder.ndim != 1 or ladder.numel() < 2:
            raise ValueError("temperatures must be a 1-D ladder of at least two values")
        if torch.any(ladder <= 0) or torch.any(ladder[1:] <= ladder[:-1]):
            raise ValueError("temperatures must be positive and strictly increasing")
        if exchange_every < 1:
            raise ValueError("exchange_every must be at least 1")
        self.temperatures = ladder
        self.exchange_every = int(exchange_every)
        self.random_seed = int(random_seed)
        self._generator: torch.Generator | None = None
        self._calls = 0
        self._segment = 0
        self._atoms_per_graph: int | None = None
        n_pairs = ladder.numel() - 1
        self.pair_attempts = [0] * n_pairs
        self._pair_accepted: torch.Tensor | None = None
        self.replica_ids: torch.Tensor | None = None

    def _check_batch(self, batch: Batch) -> int:
        if self._atoms_per_graph is not None:
            return self._atoms_per_graph
        if batch.num_graphs != self.temperatures.numel():
            raise ValueError(
                f"batch has {batch.num_graphs} graphs but the ladder has "
                f"{self.temperatures.numel()} temperatures"
            )
        counts = batch.num_nodes_per_graph
        if not bool((counts == counts[0]).all()):
            raise ValueError(
                "parallel tempering needs the same atom count in every graph"
            )
        self._atoms_per_graph = int(counts[0])
        return self._atoms_per_graph

    def exchange(self, batch: Batch) -> torch.Tensor | None:
        """Attempt one round of neighbour exchanges; return the slot permutation applied.

        Parameters
        ----------
        batch
            The ladder batch, modified in place.

        Returns
        -------
        torch.Tensor | None
            ``perm`` such that slot ``s`` now holds the configuration that was
            in slot ``perm[s]``; ``None`` if this round proposes no pair.
        """
        n_atoms = self._check_batch(batch)
        device = batch.device
        n_slots = batch.num_graphs
        if self.replica_ids is None:
            self.replica_ids = torch.arange(n_slots, device=device)
            self._pair_accepted = torch.zeros(
                n_slots - 1, dtype=torch.long, device=device
            )
        pairs = even_odd_pairs(self._segment, n_slots)
        self._segment += 1
        if not pairs:
            return None
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device)
            self._generator.manual_seed(self.random_seed)
        rows_i = torch.tensor([p[0] for p in pairs], device=device)
        rows_j = torch.tensor([p[1] for p in pairs], device=device)
        temperature = self.temperatures.to(device)
        beta = 1.0 / (KB_EV * temperature)
        energy = batch.energy.detach().reshape(n_slots).to(torch.float64)
        log_accept = (
            (beta[rows_i] - beta[rows_j]) * (energy[rows_i] - energy[rows_j])
        ).clamp(max=0.0)
        uniform = torch.rand(
            len(pairs), device=device, dtype=torch.float64, generator=self._generator
        )
        accepted = torch.log(uniform) < log_accept
        perm = torch.arange(n_slots, device=device)
        perm[rows_i] = torch.where(accepted, rows_j, rows_i)
        perm[rows_j] = torch.where(accepted, rows_i, rows_j)
        for i, _ in pairs:
            self.pair_attempts[i] += 1
        self._pair_accepted[rows_i] += accepted.to(torch.long)
        self._permute(batch, perm, temperature, n_atoms)
        self.replica_ids = self.replica_ids[perm]
        # The MC baseline must describe the permuted configurations (a cached
        # neighbour graph is rebuilt by the sampler's own synchronize).
        self.mc.synchronize(batch)
        return perm

    @staticmethod
    def _permute(
        batch: Batch, perm: torch.Tensor, temperature: torch.Tensor, n_atoms: int
    ) -> None:
        n_slots = perm.numel()
        with torch.no_grad():
            for name in _ATOM_FIELDS:
                tensor = getattr(batch, name, None)
                if tensor is None:
                    continue
                view = tensor.reshape(n_slots, n_atoms, *tensor.shape[1:])
                moved = view[perm]
                if name == "velocities":
                    scale = torch.sqrt(temperature / temperature[perm]).to(tensor.dtype)
                    moved = moved * scale.reshape(n_slots, *([1] * (moved.ndim - 1)))
                tensor.copy_(moved.reshape(tensor.shape))
            energy = batch.energy
            energy.copy_(energy.reshape(n_slots, -1)[perm].reshape(energy.shape))

    def run_mc_block(self, batch: Batch) -> None:
        """Exchange (every ``exchange_every``-th call), then run one MC block."""
        if self._calls % self.exchange_every == 0:
            self.exchange(batch)
        self._calls += 1
        super().run_mc_block(batch)

    @property
    def acceptance_rates(self) -> list[float]:
        """Exchange acceptance of each neighbouring pair ``(s, s + 1)``; NaN if never tried."""
        accepted = (
            [0] * len(self.pair_attempts)
            if self._pair_accepted is None
            else self._pair_accepted.tolist()
        )
        return [
            acc / att if att else float("nan")
            for acc, att in zip(accepted, self.pair_attempts, strict=True)
        ]
