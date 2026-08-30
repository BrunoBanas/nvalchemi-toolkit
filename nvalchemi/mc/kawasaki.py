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
"""Canonical (fixed-composition) nearest-neighbour Kawasaki Monte Carlo."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.torch.neighbors import neighbor_list
from nvalchemiops.torch.neighbors.neighbor_utils import (
    get_neighbor_list_from_neighbor_matrix,
)

from nvalchemi.data import Batch
from nvalchemi.mc.base import BaseMonteCarlo

if TYPE_CHECKING:
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["Kawasaki"]


def _nearest_neighbor_edges(batch: Batch, cutoff: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a deduplicated, undirected nearest-neighbour pair list.

    Uses the same GPU cutoff neighbor-search kernel as
    :func:`nvalchemi.neighbors.compute_neighbors`, kept off the batch so it
    never collides with the energy model's own (typically longer-range)
    interaction neighbor list. The search is a plain radius cutoff over the
    current positions with no lattice-site assumption, so it is robust to
    thermally disordered geometry from a finite-temperature MD block.

    Parameters
    ----------
    batch : Batch
        Batch whose current positions define the proposal geometry.
    cutoff : float
        Nearest-neighbour proposal cutoff radius, in Angstrom.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(edges, counts)`` where ``edges`` has shape ``(E, 2)`` int64 (each
        undirected pair listed once) and ``counts`` has shape
        ``(num_graphs,)`` int64 with the per-graph edge count.
    """
    N = batch.num_nodes
    device = batch.device
    pbc = getattr(batch, "pbc", None)
    cell = getattr(batch, "cell", None)
    if pbc is not None and not bool(pbc.any()):
        pbc = None
        cell = None

    max_neighbors = estimate_max_neighbors(cutoff=cutoff)
    if pbc is None and batch.max_num_nodes > 0:
        cap = ((batch.max_num_nodes + 15) // 16) * 16
        max_neighbors = min(max_neighbors, cap)

    batch_ptr = batch.batch_ptr.to(torch.int32)
    batch_idx = batch.batch_idx.to(torch.int32)

    while True:
        nb_matrix = torch.full((N, max_neighbors), N, dtype=torch.int32, device=device)
        nb_counts = torch.zeros(N, dtype=torch.int32, device=device)
        neighbor_list(
            positions=batch.positions,
            cutoff=cutoff,
            cell=cell,
            pbc=pbc,
            max_neighbors=max_neighbors,
            half_fill=True,
            batch_ptr=batch_ptr,
            batch_idx=batch_idx,
            neighbor_matrix=nb_matrix,
            num_neighbors=nb_counts,
            neighbor_matrix_shifts=None,
            rebuild_flags=None,
        )
        actual_max = int(nb_counts.max())
        if actual_max < max_neighbors:
            break
        max_neighbors = int(actual_max * 1.5) + 1

    neighbor_list_coo = get_neighbor_list_from_neighbor_matrix(
        neighbor_matrix=nb_matrix,
        num_neighbors=nb_counts,
        neighbor_shift_matrix=None,
        fill_value=N,
    )
    edges = neighbor_list_coo[0].T.contiguous().to(torch.long)  # (E, 2)
    graph_per_edge = batch.batch_idx[edges[:, 0]]
    counts = torch.bincount(graph_per_edge, minlength=batch.num_graphs).to(torch.long)
    return edges, counts


class Kawasaki(BaseMonteCarlo):
    r"""Batched canonical MC with nearest-neighbour site-swap moves.

    Composition is fixed. A proposal draws one nearest-neighbour edge per
    active graph, uniformly, from a fixed-geometry proposal graph and swaps
    the pair's identities when the two species differ; a same-species draw
    leaves the state unchanged. The edge is chosen independently of species,
    so every ordered configuration has exactly one reverse proposal of equal
    probability -- plain Metropolis acceptance is therefore detailed-balance
    exact and no chemical-potential correction applies (the inherited
    :meth:`~nvalchemi.mc.base.BaseMonteCarlo._chemical_delta` stays zero).

    The proposal graph is a short-range nearest-neighbour list, independent
    of any interaction cutoff the energy model uses for its own neighbor
    list and never written to the batch. It is built once per accepted
    geometry and cached; call :meth:`synchronize` after positions change
    outside this sampler (e.g. an interleaved MD block) to rebuild it.
    """

    def __init__(
        self,
        model: BaseModelMixin,
        temperature: float | torch.Tensor,
        cutoff: float,
        **kwargs: Any,
    ) -> None:
        """Initialize a canonical Kawasaki sampler.

        Parameters
        ----------
        model
            Model that returns one energy per graph.
        temperature
            Positive scalar temperature in K, or one temperature per graph.
        cutoff
            Nearest-neighbour proposal cutoff radius in Angstrom, e.g. the
            system's first RDF minimum. Independent of the model's own
            interaction cutoff.
        **kwargs
            Forwarded to :class:`~nvalchemi.mc.base.BaseMonteCarlo`.
        """
        super().__init__(model=model, temperature=temperature, **kwargs)
        if cutoff <= 0.0:
            raise ValueError("Kawasaki proposal cutoff must be positive")
        self.cutoff = cutoff
        self._edges: torch.Tensor | None = None
        self._edge_offsets: torch.Tensor | None = None
        self._graph_batch_id: int | None = None
        self._proposal_first: torch.Tensor | None = None
        self._proposal_second: torch.Tensor | None = None
        self._proposal_swapped: torch.Tensor | None = None

    def _build_proposal_graph(self, batch: Batch) -> None:
        """(Re)build the fixed nearest-neighbour proposal graph in-place."""
        edges, counts = _nearest_neighbor_edges(batch, self.cutoff)
        if bool((counts == 0).any()):
            raise ValueError(
                f"Kawasaki proposal cutoff {self.cutoff:g} produced a graph with no "
                "neighbour pairs for at least one system"
            )
        offsets = torch.zeros(counts.numel() + 1, dtype=torch.long, device=batch.device)
        offsets[1:] = torch.cumsum(counts, dim=0)
        self._edges = edges
        self._edge_offsets = offsets
        self._graph_batch_id = id(batch)

    def _ensure_proposal_graph(self, batch: Batch) -> None:
        """Build the proposal graph on first use for a given batch object."""
        if self._edges is None or self._graph_batch_id != id(batch):
            self._build_proposal_graph(batch)

    def synchronize(self, batch: Batch) -> None:
        """Adopt a trusted current energy and rebuild the proposal graph.

        Call this after positions change outside this sampler (e.g. an MD
        block), since the fixed proposal graph must reflect the new geometry
        before further Kawasaki proposals.
        """
        super().synchronize(batch)
        self._build_proposal_graph(batch)

    def _propose(
        self,
        batch: Batch,
        generator: torch.Generator,
        active: torch.Tensor,
    ) -> None:
        """Swap one random nearest-neighbour pair per active graph."""
        self._ensure_proposal_graph(batch)
        counts = self._edge_offsets[1:] - self._edge_offsets[:-1]
        local_indices = torch.floor(
            torch.rand(batch.num_graphs, device=batch.device, generator=generator) * counts
        ).to(torch.long)
        rows = self._edge_offsets[:-1] + local_indices
        first = self._edges[rows, 0]
        second = self._edges[rows, 1]
        numbers = batch.atomic_numbers
        first_z = numbers[first].clone()
        second_z = numbers[second].clone()
        swap = active & (first_z != second_z)
        with torch.no_grad():
            numbers[first[swap]] = second_z[swap]
            numbers[second[swap]] = first_z[swap]
        self._proposal_first = first
        self._proposal_second = second
        self._proposal_swapped = swap

    def _restore_rejected(self, batch: Batch, rejected: torch.Tensor) -> None:
        """Swap back rejected trial pairs that were actually exchanged."""
        if (
            self._proposal_first is None
            or self._proposal_second is None
            or self._proposal_swapped is None
        ):
            raise RuntimeError("Kawasaki rejection attempted without a proposal")
        undo = rejected & self._proposal_swapped
        first = self._proposal_first[undo]
        second = self._proposal_second[undo]
        numbers = batch.atomic_numbers
        first_z = numbers[first].clone()
        numbers[first] = numbers[second]
        numbers[second] = first_z
