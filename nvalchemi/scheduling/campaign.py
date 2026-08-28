# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-aware simulation campaigns and persistent final-state storage."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

if TYPE_CHECKING:
    from nvalchemi.data import AtomicData

__all__ = [
    "CampaignScheduler",
    "CampaignSpec",
    "FinalStateStore",
    "RunSpec",
]

_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class RunSpec:
    """One independently restartable simulation state point.

    ``parent_id`` creates an explicit continuation edge.  Temperature,
    pressure, and chemical potentials deliberately do not enter
    :attr:`compatibility_key`: the supported SGC and NPT implementations can
    carry these quantities per graph in one batch.  Different model, method,
    reservoir species, or user-defined ``batch_group`` values are never mixed.
    """

    run_id: str
    temperature_k: float
    chemical_potentials_ev: Mapping[int, float]
    pressure_ev_per_a3: float = 0.0
    parent_id: str | None = None
    depends_on: tuple[str, ...] = ()
    model_key: str = "default"
    method_key: str = "hybrid_sgc_npt"
    species: tuple[int, ...] = ()
    batch_group: str = "default"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _RUN_ID.fullmatch(self.run_id):
            raise ValueError(
                "run_id must contain only letters, numbers, '.', '_', or '-' "
                "and must not begin with punctuation"
            )
        if self.temperature_k <= 0.0:
            raise ValueError("temperature_k must be positive")
        if not self.model_key or not self.method_key or not self.batch_group:
            raise ValueError("model_key, method_key, and batch_group must be non-empty")
        potentials = {int(number): float(value) for number, value in self.chemical_potentials_ev.items()}
        if not potentials:
            raise ValueError("chemical_potentials_ev must contain at least one species")
        species = tuple(int(number) for number in (self.species or tuple(potentials)))
        if len(species) != len(set(species)):
            raise ValueError("species must not contain duplicates")
        if set(species) != set(potentials):
            raise ValueError("species and chemical_potentials_ev must contain the same atomic numbers")
        if self.parent_id is not None and not _RUN_ID.fullmatch(self.parent_id):
            raise ValueError("parent_id has invalid characters")
        dependencies = tuple(self.depends_on)
        if len(dependencies) != len(set(dependencies)):
            raise ValueError("depends_on must not contain duplicates")
        if any(not _RUN_ID.fullmatch(dependency) for dependency in dependencies):
            raise ValueError("depends_on contains invalid run_id characters")
        if self.run_id in dependencies:
            raise ValueError("run_id cannot depend on itself")
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "chemical_potentials_ev", MappingProxyType(potentials))
        object.__setattr__(self, "depends_on", dependencies)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def compatibility_key(self) -> tuple[str, str, tuple[int, ...], str]:
        """Return the properties that must agree within a batched runner."""
        return (self.model_key, self.method_key, self.species, self.batch_group)

    @property
    def dependency_ids(self) -> tuple[str, ...]:
        """Return completion dependencies, including the continuation parent."""
        return ((self.parent_id,) if self.parent_id is not None else ()) + self.depends_on


@dataclass(frozen=True)
class CampaignSpec:
    """A validated directed acyclic graph of :class:`RunSpec` nodes."""

    runs: tuple[RunSpec, ...]
    name: str = "campaign"

    def __post_init__(self) -> None:
        object.__setattr__(self, "runs", tuple(self.runs))
        if not self.runs:
            raise ValueError("campaign must contain at least one run")
        ids = [run.run_id for run in self.runs]
        if len(ids) != len(set(ids)):
            raise ValueError("campaign run_id values must be unique")
        run_ids = set(ids)
        for run in self.runs:
            unknown = set(run.dependency_ids) - run_ids
            if unknown:
                raise ValueError(f"run {run.run_id!r} has unknown dependency {sorted(unknown)!r}")
            if run.parent_id == run.run_id:
                raise ValueError(f"run {run.run_id!r} cannot be its own parent")
        self.topological_runs()

    @property
    def by_id(self) -> Mapping[str, RunSpec]:
        """Return the campaign nodes indexed by stable run identifier."""
        return MappingProxyType({run.run_id: run for run in self.runs})

    def topological_runs(self) -> tuple[RunSpec, ...]:
        """Return nodes in dependency order and reject cyclic continuation."""
        remaining = {run.run_id: run for run in self.runs}
        ordered: list[RunSpec] = []
        completed: set[str] = set()
        while remaining:
            ready = [
                run
                for run in remaining.values()
                if set(run.dependency_ids).issubset(completed)
            ]
            if not ready:
                raise ValueError("campaign continuation graph contains a cycle")
            for run in ready:
                ordered.append(run)
                completed.add(run.run_id)
                del remaining[run.run_id]
        return tuple(ordered)

    def ready(self, completed_ids: set[str] | frozenset[str]) -> tuple[RunSpec, ...]:
        """Return uncompleted nodes whose parents have persistent final states."""
        completed = set(completed_ids)
        return tuple(
            run
            for run in self.topological_runs()
            if run.run_id not in completed
            and set(run.dependency_ids).issubset(completed)
        )

    @classmethod
    def cooling_from_reference(
        cls,
        reference_runs: Sequence[RunSpec],
        temperatures_k: Sequence[float],
        *,
        name: str = "cooling_campaign",
        start_after: Sequence[str] = (),
    ) -> CampaignSpec:
        """Build sequential cooling branches from high-temperature reference runs.

        ``temperatures_k`` must begin at the temperature of every reference run
        and then decrease strictly. Reference runs usually represent the final
        states of a high-temperature bidirectional chemical-potential scan.
        ``start_after`` adds a completion barrier to every cooling child; use
        the terminal nodes of both high-temperature scan directions to prevent
        cooling from beginning before the complete reference line is available.
        """
        if not reference_runs:
            raise ValueError("at least one high-temperature reference run is required")
        temperatures = tuple(float(value) for value in temperatures_k)
        if len(temperatures) < 2:
            raise ValueError("cooling requires the reference temperature and at least one lower temperature")
        if any(left <= right for left, right in zip(temperatures, temperatures[1:])):
            raise ValueError("temperatures_k must decrease strictly after the reference point")
        if any(run.temperature_k != temperatures[0] for run in reference_runs):
            raise ValueError("every reference run must be at temperatures_k[0]")
        barrier = tuple(start_after)
        reference_ids = {run.run_id for run in reference_runs}
        if set(barrier) - reference_ids:
            raise ValueError("start_after must refer to a supplied reference run")

        runs = list(reference_runs)
        for reference in reference_runs:
            parent_id = reference.run_id
            for temperature in temperatures[1:]:
                child_id = f"{reference.run_id}.cool.T{temperature:g}"
                runs.append(
                    replace(
                        reference,
                        run_id=child_id,
                        temperature_k=temperature,
                        parent_id=parent_id,
                        depends_on=barrier,
                    )
                )
                parent_id = child_id
        return cls(runs=tuple(runs), name=name)


class FinalStateStore:
    """Atomic on-disk final states for continuation or campaign restart.

    The stored ``AtomicData`` includes positions, atom types, cell, velocities,
    and any graph fields in the completed result. ``runtime_state`` is an
    optional tensor-only payload for future integrator/MC restart state.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, run_id: str) -> Path:
        """Return the validated checkpoint path for one campaign node."""
        if not _RUN_ID.fullmatch(run_id):
            raise ValueError("run_id has invalid characters")
        return self.root / f"{run_id}.pt"

    def exists(self, run_id: str) -> bool:
        """Return whether a durable final-state record exists for ``run_id``."""
        return self.path_for(run_id).is_file()

    def save(
        self,
        run_id: str,
        state: AtomicData,
        *,
        runtime_state: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Persist one final atomic state using atomic replacement."""
        state_cpu = state.clone().cpu()
        payload = {
            "run_id": run_id,
            "state": state_cpu.model_dump(exclude_none=True),
            "runtime_state": dict(runtime_state or {}),
        }
        path = self.path_for(run_id)
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def load(
        self,
        run_id: str,
        *,
        device: torch.device | str = "cpu",
    ) -> AtomicData:
        """Load one final atomic state onto the requested device."""
        from nvalchemi.data import AtomicData

        payload = torch.load(self.path_for(run_id), map_location=device, weights_only=True)
        if payload.get("run_id") != run_id:
            raise RuntimeError(f"checkpoint identity mismatch for {run_id!r}")
        return AtomicData.model_validate(payload["state"])


class CampaignScheduler:
    """Expose ready campaign nodes, continuation parents, and completion API."""

    def __init__(self, campaign: CampaignSpec, state_store: FinalStateStore) -> None:
        self.campaign = campaign
        self.state_store = state_store
        self._completed = {
            run.run_id for run in campaign.runs if state_store.exists(run.run_id)
        }

    @property
    def completed_ids(self) -> frozenset[str]:
        """Return durable completion state reconstructed from the store."""
        return frozenset(self._completed)

    def ready(self) -> tuple[RunSpec, ...]:
        """Return all currently runnable nodes in dependency order."""
        return self.campaign.ready(self._completed)

    def ready_batches(self, max_batch_size: int) -> tuple[tuple[RunSpec, ...], ...]:
        """Group ready compatible nodes into batches without breaking dependencies."""
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        groups: dict[tuple[str, str, tuple[int, ...], str], list[RunSpec]] = {}
        for run in self.ready():
            groups.setdefault(run.compatibility_key, []).append(run)
        batches: list[tuple[RunSpec, ...]] = []
        for runs in groups.values():
            batches.extend(
                tuple(runs[index : index + max_batch_size])
                for index in range(0, len(runs), max_batch_size)
            )
        return tuple(batches)

    def ready_batch_waves(
        self,
        max_batch_size: int,
        gpu_ids: Sequence[int],
    ) -> tuple[tuple[Any, tuple[RunSpec, ...]], ...]:
        """Assign complete ready batches to GPU waves without splitting them.

        The returned assignment object is the ``RunAssignment`` emitted by
        :class:`~nvalchemi.scheduling.SimulationBatchPlanner`; its ``start``
        index selects the corresponding ready batch. A process-per-GPU runner
        can execute same-wave assignments concurrently and later waves serially.
        """
        from nvalchemi.scheduling.batching import SimulationBatchPlanner

        batches = self.ready_batches(max_batch_size)
        if not batches:
            return ()
        assignments = SimulationBatchPlanner.assign_runs(
            total_runs=len(batches),
            batch_width=1,
            gpu_ids=gpu_ids,
        )
        return tuple((assignment, batches[assignment.start]) for assignment in assignments)

    def parent_state(self, run: RunSpec, *, device: torch.device | str = "cpu") -> AtomicData | None:
        """Load the final parent configuration required for continuation."""
        if run.parent_id is None:
            return None
        if run.parent_id not in self._completed:
            raise RuntimeError(f"parent {run.parent_id!r} is not complete")
        return self.state_store.load(run.parent_id, device=device)

    def complete(
        self,
        run_id: str,
        state: AtomicData,
        *,
        runtime_state: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Durably finalize a ready node and unlock its dependent child."""
        run = self.campaign.by_id.get(run_id)
        if run is None:
            raise KeyError(f"unknown run_id {run_id!r}")
        if run_id in self._completed:
            raise RuntimeError(f"run {run_id!r} is already complete")
        if run not in self.ready():
            raise RuntimeError(f"run {run_id!r} is not ready")
        self.state_store.save(run_id, state, runtime_state=runtime_state)
        self._completed.add(run_id)
