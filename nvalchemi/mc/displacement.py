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
"""Off-lattice Metropolis displacement moves and a mixed-move MC sampler."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

import torch

from nvalchemi.data import Batch
from nvalchemi.mc.base import BaseMonteCarlo, MonteCarloStats

if TYPE_CHECKING:
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["CompositeMC", "Displacement"]


class Displacement(BaseMonteCarlo):
    r"""Batched off-lattice Metropolis MC with random Cartesian displacements.

    ``mode="single"`` moves one uniformly chosen atom per graph by a vector
    drawn uniformly from the cube :math:`[-\delta, \delta]^3`;
    ``mode="collective"`` moves every atom of the graph at once, each by its
    own independent cube draw. Both proposals are symmetric, so plain
    Metropolis acceptance is exact. Species are never changed: combine with a
    swap or transmutation sampler through :class:`CompositeMC` for an
    off-lattice MC that also samples chemical order.

    With a message-passing potential whose receptive field spans the whole
    system (UMA on a few-nm particle), a single-atom trial costs one full
    forward pass, so one displacement sweep costs ``N`` passes -- the reason
    MD (one force call moves all atoms) is normally preferred. This sampler
    exists to measure that cost, not to hide it.

    The step size :math:`\delta` (Å) is one value per graph and may be changed
    between blocks (:attr:`step_size`), e.g. to tune acceptance during a
    burn-in; changing it during production breaks detailed balance.
    """

    def __init__(
        self,
        model: BaseModelMixin,
        temperature: float | torch.Tensor,
        step_size: float | torch.Tensor = 0.1,
        mode: Literal["single", "collective"] = "single",
        **kwargs: Any,
    ) -> None:
        """Initialize a displacement sampler.

        Parameters
        ----------
        model
            Model that returns one energy per graph.
        temperature
            Positive scalar temperature in K, or one temperature per graph.
        step_size
            Half-width of the cubic displacement box in Å, scalar or one per
            graph.
        mode
            ``"single"`` (one atom per trial) or ``"collective"`` (all atoms).
        **kwargs
            Forwarded to :class:`~nvalchemi.mc.base.BaseMonteCarlo`.
        """
        super().__init__(model=model, temperature=temperature, **kwargs)
        if mode not in ("single", "collective"):
            raise ValueError("mode must be 'single' or 'collective'")
        step = torch.as_tensor(step_size, dtype=torch.float64)
        if step.ndim > 1 or torch.any(step <= 0):
            raise ValueError("step_size must be positive and scalar or one-dimensional")
        self.mode = mode
        self._step_size = step
        self._moved: torch.Tensor | None = None
        self._old_positions: torch.Tensor | None = None

    @property
    def step_size(self) -> torch.Tensor:
        """Half-width of the displacement box in Å (scalar or one per graph)."""
        return self._step_size

    @step_size.setter
    def step_size(self, value: float | torch.Tensor) -> None:
        step = torch.as_tensor(value, dtype=torch.float64).detach().cpu()
        if step.ndim > 1 or torch.any(step <= 0):
            raise ValueError("step_size must be positive and scalar or one-dimensional")
        self._step_size = step

    def _step_for(self, batch: Batch) -> torch.Tensor:
        """Return one step size per graph on the batch device."""
        step = self._step_size.to(device=batch.device, dtype=batch.positions.dtype)
        if step.ndim == 0:
            return step.expand(batch.num_graphs)
        if step.shape != (batch.num_graphs,):
            raise ValueError(
                "step_size must be scalar or have one value per graph "
                f"(expected {(batch.num_graphs,)}, got {tuple(step.shape)})"
            )
        return step

    def _propose(
        self,
        batch: Batch,
        generator: torch.Generator,
        active: torch.Tensor,
    ) -> None:
        """Displace one atom (or all atoms) of every active graph in place."""
        positions = batch.positions
        step = self._step_for(batch)
        if self.mode == "single":
            ptr = batch.batch_ptr.to(torch.long)
            starts = ptr[:-1]
            sizes = ptr[1:] - starts
            draw = torch.rand(
                batch.num_graphs, device=batch.device, generator=generator
            )
            atom = torch.minimum(
                starts + torch.floor(draw * sizes).to(torch.long), ptr[1:] - 1
            )
            delta = (
                torch.rand(
                    batch.num_graphs,
                    3,
                    device=batch.device,
                    generator=generator,
                    dtype=positions.dtype,
                )
                * 2.0
                - 1.0
            ) * step[:, None]
            delta = torch.where(active[:, None], delta, torch.zeros_like(delta))
            self._moved = atom
            self._old_positions = positions[atom].detach().clone()
            with torch.no_grad():
                # One atom per graph, so the indices are distinct.
                positions.index_add_(0, atom, delta)
        else:
            graph_of_atom = batch.batch_idx.to(torch.long)
            delta = (
                torch.rand(
                    positions.shape,
                    device=batch.device,
                    generator=generator,
                    dtype=positions.dtype,
                )
                * 2.0
                - 1.0
            ) * step[graph_of_atom, None]
            delta = torch.where(
                active[graph_of_atom, None], delta, torch.zeros_like(delta)
            )
            self._moved = None
            self._old_positions = positions.detach().clone()
            with torch.no_grad():
                positions.add_(delta)

    def _restore_rejected(self, batch: Batch, rejected: torch.Tensor) -> None:
        """Put the displaced atoms of rejected graphs back."""
        if self._old_positions is None:
            raise RuntimeError("Displacement rejection attempted without a proposal")
        positions = batch.positions
        with torch.no_grad():
            if self._moved is not None:
                keep = rejected[:, None]
                positions[self._moved] = torch.where(
                    keep, self._old_positions, positions[self._moved]
                )
            else:
                undo = rejected[batch.batch_idx.to(torch.long), None]
                positions.copy_(torch.where(undo, self._old_positions, positions))


class CompositeMC(BaseMonteCarlo):
    """Mix several MC move types into one sampler, one move type per step.

    Each step picks one of ``moves`` with probability proportional to
    ``weights`` -- the same move for every graph of the step -- and runs its
    proposal, chemical term and rejection. The choice uses a host-side seeded
    generator, so it never synchronizes the device, and it does not depend on
    the state, so a mixture of detailed-balance moves keeps detailed balance.
    The energy baseline, acceptance test, statistics and mass bookkeeping are
    this sampler's own; the sub-samplers only supply their move.

    Typical use: ``CompositeMC(model, T, [Displacement(...), NonlocalSwap(...)],
    weights=[0.5, 0.5])`` -- off-lattice MC sampling positions and chemical
    order together, with no MD. Moves that cache geometry between steps
    (:class:`~nvalchemi.mc.Kawasaki`'s neighbour graph) go stale when a
    displacement move is accepted, so combine displacements only with
    geometry-free swaps or transmutations
    (:class:`~nvalchemi.mc.NonlocalSwap`, :class:`~nvalchemi.mc.SGC`,
    :class:`~nvalchemi.mc.VCSGC`).
    """

    def __init__(
        self,
        model: BaseModelMixin,
        temperature: float | torch.Tensor,
        moves: Sequence[BaseMonteCarlo],
        weights: Sequence[float] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a mixed-move sampler.

        Parameters
        ----------
        model
            Model that returns one energy per graph (shared with the moves).
        temperature
            Positive scalar temperature in K, or one temperature per graph.
        moves
            Sub-samplers whose proposals are mixed. Their own temperature,
            seed and hooks are unused.
        weights
            Relative selection probability of each move; uniform by default.
        **kwargs
            Forwarded to :class:`~nvalchemi.mc.base.BaseMonteCarlo`.
        """
        super().__init__(model=model, temperature=temperature, **kwargs)
        if not moves:
            raise ValueError("CompositeMC needs at least one move")
        if weights is None:
            weights = [1.0] * len(moves)
        if (
            len(weights) != len(moves)
            or any(w < 0 for w in weights)
            or not any(w > 0 for w in weights)
        ):
            raise ValueError("weights must be non-negative, not all zero, one per move")
        self.moves = list(moves)
        total = float(sum(weights))
        self.weights = [float(w) / total for w in weights]
        self._move_rng = random.Random(self._random_seed)
        self._current: int | None = None
        self._move_attempted: list[torch.Tensor | None] = [None] * len(self.moves)
        self._move_accepted: list[torch.Tensor | None] = [None] * len(self.moves)

    def _validate_batch(self, batch: Batch) -> None:
        """Validate the batch against every sub-move."""
        for move in self.moves:
            move._validate_batch(batch)

    def _propose(
        self,
        batch: Batch,
        generator: torch.Generator,
        active: torch.Tensor,
    ) -> None:
        """Pick one move type and let it propose on every active graph."""
        index = self._move_rng.choices(range(len(self.moves)), weights=self.weights)[0]
        self._current = index
        move = self.moves[index]
        # Sub-moves read their own ``_active`` (e.g. SGC's chemical term); hand
        # them this step's mask, which they may narrow in place.
        move._active = active
        move._propose(batch, generator, active)

    def _current_move(self) -> BaseMonteCarlo:
        if self._current is None:
            raise RuntimeError("CompositeMC used without a proposal")
        return self.moves[self._current]

    def _restore_rejected(self, batch: Batch, rejected: torch.Tensor) -> None:
        """Delegate the rejection to the move that proposed."""
        self._current_move()._restore_rejected(batch, rejected)

    def _chemical_delta(self, batch: Batch) -> torch.Tensor:
        """Delegate the chemical term to the move that proposed."""
        return self._current_move()._chemical_delta(batch)

    def post_update(self, batch: Batch) -> None:
        """Accept or restore, then tally the outcome for the move that proposed."""
        super().post_update(batch)
        index = self._current
        if index is None or self._active is None:
            return
        attempted = self._active.sum().detach()
        accepted = (batch.mc_accepted.reshape(-1) & self._active).sum().detach()
        if self._move_attempted[index] is None:
            self._move_attempted[index] = attempted
            self._move_accepted[index] = accepted
        else:
            self._move_attempted[index] += attempted
            self._move_accepted[index] += accepted

    def reset_statistics(self) -> None:
        """Reset the overall and per-move counters."""
        super().reset_statistics()
        self._move_attempted = [None] * len(self.moves)
        self._move_accepted = [None] * len(self.moves)

    @property
    def move_stats(self) -> list[MonteCarloStats]:
        """Host-side attempted/accepted counts for each move, in ``moves`` order."""
        return [
            MonteCarloStats(
                attempted=0 if att is None else int(att.item()),
                accepted=0 if acc is None else int(acc.item()),
            )
            for att, acc in zip(self._move_attempted, self._move_accepted, strict=True)
        ]
