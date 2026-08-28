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
"""Semi-grand-canonical Monte Carlo transmutation moves."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.data import Batch
from nvalchemi.mc.base import BaseMonteCarlo

if TYPE_CHECKING:
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["SGC"]


class SGC(BaseMonteCarlo):
    r"""Batched semi-grand-canonical MC with single-site transmutations.

    The sampled potential is :math:`\Omega = E - \sum_i \mu_i N_i`. A
    proposal from species ``old`` to ``new`` therefore uses
    :math:`\Delta\Omega = \Delta E - (\mu_{new} - \mu_{old})`.
    """

    def __init__(
        self,
        model: BaseModelMixin,
        temperature: float | torch.Tensor,
        species: Sequence[int],
        chemical_potentials: Mapping[int, float | torch.Tensor],
        **kwargs: Any,
    ) -> None:
        """Initialize an SGC sampler.

        Parameters
        ----------
        model
            Model that returns one energy per graph.
        temperature
            Positive scalar temperature in K, or one temperature per graph.
        species
            Allowed atomic numbers. At least two unique species are required.
        chemical_potentials
            Chemical potential in eV for every allowed atomic number. Each
            value may be a scalar shared by the batch or a one-dimensional
            tensor with one value per graph, enabling different reservoirs in
            the same batched SGC calculation.
        **kwargs
            Forwarded to :class:`~nvalchemi.mc.base.BaseMonteCarlo`.
        """
        super().__init__(model=model, temperature=temperature, **kwargs)
        self.species = tuple(int(value) for value in species)
        if len(self.species) < 2 or len(set(self.species)) != len(self.species):
            raise ValueError("species must contain at least two unique atomic numbers")
        self.chemical_potentials = {
            int(number): torch.as_tensor(value).detach().clone()
            for number, value in chemical_potentials.items()
        }
        if any(value.ndim > 1 for value in self.chemical_potentials.values()):
            raise ValueError("chemical potentials must be scalar or one-dimensional per graph")
        missing = set(self.species) - self.chemical_potentials.keys()
        if missing:
            raise ValueError(f"missing chemical potentials for species: {sorted(missing)}")
        self._proposal_indices: torch.Tensor | None = None
        self._proposal_old: torch.Tensor | None = None
        self._proposal_new: torch.Tensor | None = None

    def _species_tensor(self, batch: Batch) -> torch.Tensor:
        """Return configured species on the batch device."""
        return torch.tensor(self.species, dtype=batch.atomic_numbers.dtype, device=batch.device)

    def _validate_batch(self, batch: Batch) -> None:
        """Reject an initial batch containing species outside the SGC reservoir."""
        present = torch.isin(batch.atomic_numbers, self._species_tensor(batch)).all()
        if not bool(present):
            raise ValueError("batch contains atomic numbers outside the configured species")

    def _propose(
        self,
        batch: Batch,
        generator: torch.Generator,
        active: torch.Tensor,
    ) -> None:
        """Change one random active site to a uniformly chosen other species."""
        counts = batch.num_nodes_per_graph
        local_indices = torch.floor(
            torch.rand(batch.num_graphs, device=batch.device, generator=generator) * counts
        ).to(torch.long)
        indices = batch.batch_ptr[:-1].to(torch.long) + local_indices
        old = batch.atomic_numbers[indices].clone()
        species = self._species_tensor(batch)
        old_rank = (old[:, None] == species[None, :]).to(torch.long).argmax(dim=1)
        offset = torch.randint(
            len(self.species) - 1,
            (batch.num_graphs,),
            device=batch.device,
            generator=generator,
        )
        new = species[(old_rank + offset + 1) % len(self.species)]
        with torch.no_grad():
            batch.atomic_numbers[indices[active]] = new[active]
        self._proposal_indices = indices
        self._proposal_old = old
        self._proposal_new = new

    def _restore_rejected(self, batch: Batch, rejected: torch.Tensor) -> None:
        """Restore atom types for rejected trial transmutations."""
        if self._proposal_indices is None or self._proposal_old is None:
            raise RuntimeError("SGC rejection attempted without a proposal")
        batch.atomic_numbers[self._proposal_indices[rejected]] = self._proposal_old[rejected]

    def _chemical_potentials_for(self, batch: Batch) -> torch.Tensor:
        """Return one chemical-potential vector per graph in the batch."""
        rows: list[torch.Tensor] = []
        for number in self.species:
            value = self.chemical_potentials[number].to(
                dtype=batch.positions.dtype,
                device=batch.device,
            )
            if value.ndim == 0:
                value = value.expand(batch.num_graphs)
            elif value.shape != (batch.num_graphs,):
                raise ValueError(
                    "chemical-potential tensors must have one value per graph "
                    f"(expected {(batch.num_graphs,)}, got {tuple(value.shape)})"
                )
            rows.append(value)
        return torch.stack(rows, dim=1)

    def _chemical_delta(self, batch: Batch) -> torch.Tensor:
        """Return ``-(mu_new - mu_old)`` for each current SGC proposal."""
        if self._proposal_old is None or self._proposal_new is None or self._active is None:
            raise RuntimeError("SGC chemical contribution requested without a proposal")
        potentials = self._chemical_potentials_for(batch)
        species = self._species_tensor(batch)
        old_rank = (self._proposal_old[:, None] == species[None, :]).to(torch.long).argmax(dim=1)
        new_rank = (self._proposal_new[:, None] == species[None, :]).to(torch.long).argmax(dim=1)
        graph_indices = torch.arange(batch.num_graphs, device=batch.device)
        chemical_delta = (
            potentials[graph_indices, old_rank] - potentials[graph_indices, new_rank]
        )
        return torch.where(self._active, chemical_delta, torch.zeros_like(chemical_delta))
