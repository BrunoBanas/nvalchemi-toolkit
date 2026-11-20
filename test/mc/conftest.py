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
"""Analytic tether model with an exactly known canonical distribution, for MC tests."""

from __future__ import annotations

import itertools
import math

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.models.base import BaseModelMixin, ModelConfig

SPECIES = (1, 2)
MASSES = {1: 1.0, 2: 4.0}


class TetherModel(torch.nn.Module, BaseModelMixin):
    r"""Each atom is tied to its site by a species-dependent spring, plus a site field.

    For site :math:`s` holding species :math:`z`,
    :math:`u = \tfrac12 k_{s z} |r - a_s|^2 + h_{s z}`. Every graph has the
    same ``n`` sites, so the canonical distribution is known in closed form:
    positions are Gaussian with variance :math:`k_BT/k_{sz}` per component,
    and an arrangement of species over sites has weight
    :math:`\prod_s e^{-\beta h_{s z_s}} (2\pi k_BT/k_{s z_s})^{3/2}` --
    springs depending on the species make the arrangement probability
    depend on the positional entropy, which a swap move that ignores the
    positions cannot get right.
    """

    def __init__(
        self, anchors: torch.Tensor, springs: torch.Tensor, fields: torch.Tensor
    ) -> None:
        super().__init__()
        self.anchors = anchors.double()  # [n, 3]
        self.springs = springs.double()  # [n, 2]
        self.fields = fields.double()  # [n, 2]
        self.calls = 0
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            autograd_outputs=frozenset(),
            autograd_inputs=frozenset(),
            neighbor_config=None,
            needs_pbc=False,
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """No embeddings."""
        return {}

    def compute_embeddings(self, data: Batch) -> Batch:
        """No embeddings."""
        return data

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        """Energy per graph and analytic forces."""
        self.calls += 1
        positions = batch.positions.double()
        graph = batch.batch_idx.to(torch.long)
        site = torch.arange(
            positions.shape[0], device=positions.device
        ) - batch.batch_ptr[graph].to(torch.long)
        species = (batch.atomic_numbers.reshape(-1) == SPECIES[1]).to(torch.long)
        spring = self.springs[site, species]
        field = self.fields[site, species]
        offset = positions - self.anchors[site]
        per_atom = 0.5 * spring * offset.square().sum(dim=1) + field
        energy = torch.zeros(batch.num_graphs, dtype=torch.float64).index_add_(
            0, graph, per_atom
        )
        out = {"energy": energy.reshape(-1, 1).to(batch.positions.dtype)}
        if "forces" in self.model_config.active_outputs:
            out["forces"] = (-spring[:, None] * offset).to(batch.positions.dtype)
        return out


def default_model() -> TetherModel:
    """Three well-separated sites with species-dependent springs and fields."""
    anchors = torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    springs = torch.tensor([[1.0, 4.0], [2.0, 1.0], [1.0, 1.0]])
    fields = torch.tensor([[0.0, 0.0], [0.0, 0.05], [0.0, -0.03]])
    return TetherModel(anchors, springs, fields)


def tether_batch(
    model: TetherModel, numbers: list[int], n_graphs: int, *, seed: int = 0
) -> Batch:
    """``n_graphs`` copies of one arrangement at the anchors, with masses and velocities."""
    generator = torch.Generator().manual_seed(seed)
    graphs = []
    for _ in range(n_graphs):
        positions = model.anchors.float() + 0.05 * torch.randn(
            model.anchors.shape, generator=generator
        )
        graphs.append(
            AtomicData(
                atomic_numbers=torch.tensor(numbers, dtype=torch.long),
                positions=positions,
            )
        )
    batch = Batch.from_data_list(graphs)
    batch.energy = torch.zeros(batch.num_graphs, 1)
    batch.forces = torch.zeros_like(batch.positions)
    batch.atomic_masses = torch.tensor(
        [MASSES[int(z)] for z in batch.atomic_numbers.reshape(-1)],
        dtype=batch.positions.dtype,
    )
    batch.velocities = torch.zeros_like(batch.positions)
    return batch


def exact_arrangement_probabilities(
    model: TetherModel, numbers: list[int], temperature: float
) -> dict[tuple[int, ...], float]:
    """Canonical probability of every distinct arrangement of ``numbers`` over the sites."""
    kt = KB_EV * temperature
    weights: dict[tuple[int, ...], float] = {}
    for arrangement in set(itertools.permutations(numbers)):
        log_w = 0.0
        for site, z in enumerate(arrangement):
            column = int(z == SPECIES[1])
            k = float(model.springs[site, column])
            log_w += -float(model.fields[site, column]) / kt + 1.5 * math.log(
                2 * math.pi * kt / k
            )
        weights[arrangement] = math.exp(log_w)
    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def arrangement_counts(batch: Batch, counts: dict[tuple[int, ...], int]) -> None:
    """Add the current arrangement of every graph to ``counts``."""
    n = int(batch.num_nodes_per_graph[0])
    for row in batch.atomic_numbers.reshape(-1, n).tolist():
        key = tuple(int(z) for z in row)
        counts[key] = counts.get(key, 0) + 1
