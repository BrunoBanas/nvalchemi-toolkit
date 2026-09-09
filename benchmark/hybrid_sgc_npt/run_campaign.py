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
"""Hybrid SGC-NPT efficiency benchmark: coarse Au-Pt T / delta_mu grid.

One process runs one system size end-to-end (see ``--n-atoms``); submit
``submit_campaign.slurm`` to fill three single-GPU jobs, one per size.

Grid
----
* Temperature: 3000 K -> 1600 K in 200 K steps (8 points).
* Chemical potential: DELTA_MU_EV, -1.0 -> 1.0 eV in 0.2 eV steps (11
  points). What this quantity actually means depends on
  --reference-energies-json (see "Chemical-potential calibration" below):
  uncalibrated, it is the literal, raw mu(Pt, Z=78) - mu(Au, Z=79); with a
  calibration file it is delta_mu_excess added on top of the calibrated
  delta_mu_ref(T) at each run's own temperature.
* Sizes: 500 / 1372 / 2048 atom conventional-cubic Au fcc supercells
  (5x5x5 / 7x7x7 / 8x8x8 conventional cells).

That is 8 * 11 = 88 state points per size, 264 total across all three sizes.

Chemical-potential calibration
-------------------------------
`nvalchemi.mc.SGC`'s chemical_potentials is a literal, absolute per-atom
energy, not a relative bias: {Au: 0.0, Pt: delta_mu} does not mean "Au and
Pt are equally favorable," it means "whatever this checkpoint's own raw
energy convention already encodes between the two species, uncorrected."
That raw offset is 1-3 eV/atom on this UMA checkpoint -- large enough to
swamp the +-1.0 eV DELTA_MU_EV sweep above and drive every run to one pure
phase regardless of delta_mu (nvalchemi-toolkit-quest-deploy's
scout_sgc_temperature_composition_drift.py job 5484379 is a documented
example of exactly this failure mode, at delta_mu=0.0).

--reference-energies-json <reference_energy_calibration.py output>
(recommended; see nvalchemi-toolkit-quest-deploy/phase_diagram_guide.md
section 3) fixes this: every run's chemical_potentials_ev is rebuilt as
{Au: 0.0, Pt: reference[T]["delta_mu_ref_eV"] + delta_mu_excess}, looked up
at THAT RUN'S OWN temperature_k -- not just the 3000 K reference row's --
since delta_mu_ref(T) genuinely varies with T. delta_mu_excess is
DELTA_MU_EV's per-column value, recovered from each run's
metadata["delta_mu_ev"] (identical across every reference and
continuation-child row in that column, since cooling_from_reference and the
flat-grid branch both copy metadata unchanged). Refuses to run if a
requested temperature is missing from the file, if its species order
doesn't match SPECIES, or if either species' equilibration_gate.resolved
at that temperature isn't true, unless --allow-unresolved-reference is
passed (not recommended). Omitting --reference-energies-json keeps the old
literal, uncalibrated behavior (a runtime warning is printed) -- fine for a
pure throughput/memory benchmark, not for drawing conclusions about the
real Au-Pt phase boundary.

Continuation
------------
Each delta_mu column is one ``CampaignSpec.cooling_from_reference`` chain: an
independent 3000 K reference state seeds seven cooling children (2800 K,
..., 1600 K) through ``RunSpec.parent_id``. 200 K is a coarse step for this
alloy, so a cooled child is not already equilibrated at its new set point,
but it inherits a plausible composition and a lattice constant close to the
target, which still cuts the required re-equilibration relative to a random
start. Reference runs therefore get the full block budget
(``N_BLOCKS_REFERENCE``); continuation children get a reduced budget
(``N_BLOCKS_CONTINUATION``) that keeps the same ~50-block equilibration
window described below. Set ``USE_CONTINUATION = False`` to run every state
point independently from a fresh random 50/50 composition instead (a clean
per-state-point throughput comparison, at roughly 2x the total block count).

Hybrid block
------------
One block is ``MD_STEPS_PER_BLOCK`` MD steps at ``DT_FS`` followed by
``round(MC_STEP_FRACTION * n_atoms)`` MC trials (one attempted transmutation
per graph per MC step). Reference runs run ``N_BLOCKS_REFERENCE`` blocks
total, continuation children run ``N_BLOCKS_CONTINUATION``.

Equilibration gate
-------------------
Equilibration used to be *assumed* within the first ``EQUILIBRATION_BLOCKS``
(50) and never checked. ``_run_hybrid_with_observables`` now records each
block's per-graph Pt fraction and energy/atom (one extra small GPU->CPU
transfer per block -- see its docstring for the cost/why), and after each
run ``_equilibration_gate`` (PHASE_DIAGRAM_MANUAL.md section 7, the same
one-shot pattern reference_energy_calibration.py already uses) compares the
last two ``EQUILIBRATION_WINDOW_BLOCKS``-block windows of both series
against twice their combined standard error. This is a single end-of-run
check, not the manual's full three-consecutive-checks promotion protocol --
a run that fails it is logged as unresolved (throughput CSV's ``resolved``
column, plus a per-run ``<run_id>.equilibration.json`` sidecar in the
checkpoint directory with both gates' window means/difference/standard
error), not retried or extended automatically.

Batch width
-----------
``SimulationBatchPlanner`` profiles representative reference-row workloads at
candidate widths and recommends the smallest width within
``BATCH_THROUGHPUT_FRACTION`` of peak throughput while reserving
``BATCH_MEMORY_FRACTION`` of device memory. A 500-atom walker is expected to
reserve about 3.5 GB; ``_verify_memory_floor`` fits a linear memory model
from the profile and raises if the fitted per-walker cost falls far below
that (atom-count-scaled) expectation, since a profiler that under-counts
memory would otherwise pick an unsafely large width for an unattended run.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import torch
from ase import Atoms
from ase.build import bulk
from ase.data import chemical_symbols

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.integrators.npt import NPT
from nvalchemi.hybrid import HybridMCMD
from nvalchemi.mc import SGC
from nvalchemi.models.uma import UMAWrapper
from nvalchemi.scheduling import (
    BatchMeasurement,
    CampaignScheduler,
    CampaignSpec,
    FinalStateStore,
    RunSpec,
    SimulationBatchPlanner,
)

# %%
# Grid definition
# ---------------

CHECKPOINT = "uma-s-1p2"
TASK = "omat"
INFERENCE_SETTINGS = "batch"  # SGC changes atomic composition per step.

TEMPLATE_SYMBOL = "Au"
CRYSTAL_STRUCTURE = "fcc"
LATTICE_A_ANG = 4.00
CONVENTIONAL_CELL = True
PT_FRACTION = 0.50
PRESSURE_EV_PER_A3 = 1.01325 / 1.602176634e6  # 1 atmosphere
SPECIES = (79, 78)  # Au, Pt

SIZE_REPEATS: dict[int, tuple[int, int, int]] = {
    500: (5, 5, 5),
    1372: (7, 7, 7),
    2048: (8, 8, 8),
    4000: (10, 10, 10),
}
BATCH_WIDTH_CANDIDATES: dict[int, tuple[int, ...]] = {
    500: (2, 4, 6, 8, 10, 12, 14, 16),
    1372: (1, 2, 3, 4, 5, 6),
    2048: (1, 2, 3, 4),
    # Existing entries follow max_width * n_atoms ~= 8000-8200 (500->16,
    # 1372->6, 2048->4); extrapolating gives ~2 at 4000. Widths beyond what
    # fits are not a crash risk -- SimulationBatchPlanner.profile() reports
    # a per-width "oom" status and moves on -- so 3 is included as a free
    # extra data point above the naive extrapolation, not a verified-safe one.
    4000: (1, 2, 3),
}

TEMPERATURES_K: tuple[float, ...] = tuple(float(t) for t in range(3000, 1599, -200))
DELTA_MU_EV: tuple[float, ...] = tuple(round(-1.0 + 0.2 * i, 2) for i in range(11))

MC_STEP_FRACTION = 0.2  # MC trials per block = round(MC_STEP_FRACTION * n_atoms)
MD_STEPS_PER_BLOCK = 50
DT_FS = 3.0
THERMOSTAT_TIME_FS = 100.0
BAROSTAT_TIME_FS = 1000.0
SEED = 20260827

EQUILIBRATION_BLOCKS = 50
N_BLOCKS_REFERENCE = 200  # 50 equilibration + 150 production; no parent state.
N_BLOCKS_CONTINUATION = 100  # 50 equilibration + 50 production; warm-started.
EQUILIBRATION_WINDOW_BLOCKS = 25  # _equilibration_gate check interval, PHASE_DIAGRAM_MANUAL.md section 7.

USE_CONTINUATION = True

BATCH_MEMORY_FRACTION = 0.85
BATCH_THROUGHPUT_FRACTION = 0.95
PROFILE_WARMUP_BLOCKS = 1
PROFILE_MEASURED_BLOCKS = 2

KB_EV = 8.617333262e-5  # Boltzmann constant, eV/K.
EXPECTED_BYTES_PER_ATOM = (3.5 * 1024**3) / 500  # ~3.5 GB reserved @ 500 atoms.
MEMORY_FLOOR_TOLERANCE = 0.75  # Require >= 75% of the atom-count-scaled estimate.


def build_ase_structure(
    symbol: str,
    crystal_structure: str,
    lattice_a: float,
    repeats: tuple[int, int, int],
    *,
    cubic: bool,
) -> Atoms:
    """Build a periodic ASE crystal template for one independent run."""
    unit_cell = bulk(symbol, crystalstructure=crystal_structure, a=lattice_a, cubic=cubic)
    return unit_cell * repeats


def _walker(
    template: Atoms,
    run: RunSpec,
    seed: int,
    parent_state: AtomicData | None,
    device: torch.device,
) -> AtomicData:
    """Create a fresh state or restore the parent state for continuation."""
    if parent_state is not None:
        return parent_state.to(device)

    data = AtomicData.from_atoms(template, device=device)
    n_atoms = data.num_nodes
    generator = torch.Generator(device=device).manual_seed(seed)
    numbers = torch.full_like(data.atomic_numbers, SPECIES[0])
    pt_fraction = float(run.metadata.get("pt_fraction", PT_FRACTION))
    pt_count = round(pt_fraction * n_atoms)
    numbers[torch.randperm(n_atoms, device=device, generator=generator)[:pt_count]] = SPECIES[1]
    data.atomic_numbers = numbers
    data.atomic_masses = None
    data.use_default_masses()
    velocity_std = torch.sqrt(
        torch.as_tensor(KB_EV * run.temperature_k, device=device) / data.atomic_masses
    )
    data.velocities = (
        torch.randn((n_atoms, 3), device=device, generator=generator) * velocity_std[:, None]
    )
    data.velocities -= data.velocities.mean(dim=0, keepdim=True)
    data.forces = torch.zeros_like(data.positions)
    data.energy = torch.zeros(1, 1, device=device)
    data.stress = torch.zeros(1, 3, 3, device=device)
    return data


def _make_batch(
    template: Atoms,
    runs: tuple[RunSpec, ...],
    parent_states: tuple[AtomicData | None, ...],
    device: torch.device,
) -> Batch:
    """Build a batch of fresh states or restored continuation states."""
    if len(runs) != len(parent_states):
        raise ValueError("runs and parent_states must have equal lengths")
    return Batch.from_data_list(
        [
            _walker(template, run, SEED + index, parent_state, device)
            for index, (run, parent_state) in enumerate(zip(runs, parent_states))
        ]
    )


def make_workload(
    model: UMAWrapper,
    template: Atoms,
    runs: tuple[RunSpec, ...],
    parent_states: tuple[AtomicData | None, ...],
    device: torch.device,
) -> tuple[HybridMCMD, Batch]:
    """Return one per-graph hybrid MC-MD workload for compatible runs.

    The same model object is deliberately passed to both stages: ``HybridMCMD``
    requires every trial energy and MD force to come from one potential.
    """
    if not runs:
        raise ValueError("a workload requires at least one run")
    species = runs[0].species
    if any(run.species != species for run in runs):
        raise ValueError("all runs in a workload must share reservoir species")
    temperatures = torch.tensor([run.temperature_k for run in runs], device=device)
    pressures = torch.tensor([run.pressure_ev_per_a3 for run in runs], device=device)
    chemical_potentials = {
        number: torch.tensor([run.chemical_potentials_ev[number] for run in runs], device=device)
        for number in species
    }
    mc_steps = max(1, round(MC_STEP_FRACTION * len(template)))
    sgc = SGC(
        model=model,
        temperature=temperatures,
        species=species,
        chemical_potentials=chemical_potentials,
        random_seed=SEED,
    )
    npt = NPT(
        model=model,
        dt=DT_FS,
        temperature=temperatures,
        pressure=pressures,
        thermostat_time=THERMOSTAT_TIME_FS,
        barostat_time=BAROSTAT_TIME_FS,
        pressure_coupling="isotropic",
    )
    hybrid = HybridMCMD(mc=sgc, md=npt, mc_steps=mc_steps, md_steps=MD_STEPS_PER_BLOCK)
    return hybrid, _make_batch(template, runs, parent_states, device)


def _build_reference_runs(n_atoms: int) -> tuple[RunSpec, ...]:
    """Return the 3000 K reference row: one run per delta_mu column."""
    return tuple(
        RunSpec(
            run_id=f"atoms{n_atoms}.mu{index}",
            temperature_k=TEMPERATURES_K[0],
            pressure_ev_per_a3=PRESSURE_EV_PER_A3,
            chemical_potentials_ev={SPECIES[0]: 0.0, SPECIES[1]: delta_mu},
            species=SPECIES,
            batch_group=f"atoms{n_atoms}",
            metadata={"delta_mu_ev": delta_mu, "n_atoms": n_atoms},
        )
        for index, delta_mu in enumerate(DELTA_MU_EV)
    )


def _build_campaign(n_atoms: int, reference_runs: tuple[RunSpec, ...]) -> CampaignSpec:
    """Build the temperature-continuation campaign, or a flat independent grid."""
    if USE_CONTINUATION:
        return CampaignSpec.cooling_from_reference(
            reference_runs,
            TEMPERATURES_K,
            name=f"aupt_{n_atoms}atoms_cooling",
        )
    runs = tuple(
        replace(reference, run_id=f"{reference.run_id}.T{temperature:g}", temperature_k=temperature)
        for reference in reference_runs
        for temperature in TEMPERATURES_K
    )
    return CampaignSpec(runs=runs, name=f"aupt_{n_atoms}atoms_independent")


def _load_delta_mu_ref(
    reference_path: Path, temperatures_k: Sequence[float], symbols: list[str], *, allow_unresolved: bool,
) -> dict[float, float]:
    """delta_mu_ref_eV per requested temperature, from
    reference_energy_calibration.py's reference_energies.json (this file's
    own TEMPERATURES_K, imported there as ``campaign.TEMPERATURES_K``, so the
    default calibration grid already covers every temperature this campaign
    needs). Raises on a missing temperature, a species mismatch, or an
    unresolved equilibration_gate (unless allow_unresolved) -- silently
    proceeding on any of those would reproduce exactly the uncalibrated-
    delta_mu problem this loader exists to fix. Ported from
    nvalchemi-toolkit-quest-deploy's
    tests/scout_sgc_temperature_composition_drift.py (different repo, no
    shared import path).
    """
    reference_data = json.loads(reference_path.read_text())
    reference = reference_data["reference"]
    if list(reference_data.get("symbols", [])) != list(symbols):
        raise ValueError(
            f"{reference_path} was calibrated for symbols {reference_data.get('symbols')}, "
            f"this run uses {symbols} -- refusing to mix calibrations across species orderings"
        )
    resolved: dict[float, float] = {}
    for t in temperatures_k:
        key = f"{t:g}"
        entry = reference.get(key)
        if entry is None:
            raise ValueError(
                f"{reference_path} has no entry for T={t:g} K -- rerun "
                f"reference_energy_calibration.py with --temperatures-k including {t:g}, "
                "or this campaign's TEMPERATURES_K no longer matches its calibration"
            )
        if "delta_mu_ref_eV" not in entry:
            raise ValueError(
                f"{reference_path}'s T={t:g} K entry has no delta_mu_ref_eV "
                "(only written for the 2-species case)"
            )
        gate_status = {
            symbol: entry[symbol]["equilibration_gate"]["resolved"] for symbol in symbols
        }
        if not allow_unresolved and not all(status is True for status in gate_status.values()):
            raise ValueError(
                f"T={t:g} K: equilibration_gate.resolved is {gate_status}, not all True -- "
                "this reference value isn't trustworthy enough to anchor delta_mu on. Pass "
                "--allow-unresolved-reference to proceed anyway (not recommended)."
            )
        resolved[t] = entry["delta_mu_ref_eV"]
    return resolved


def _apply_calibrated_delta_mu(
    campaign: CampaignSpec, delta_mu_ref_by_t: dict[float, float],
) -> CampaignSpec:
    """Replace every run's literal chemical_potentials_ev with
    {Au: 0.0, Pt: delta_mu_ref_by_t[T] + delta_mu_excess}, using each run's
    OWN temperature_k -- not just the 3000 K reference row's -- since
    delta_mu_ref(T) genuinely varies with T (phase_diagram_guide.md section
    3). delta_mu_excess is recovered from each run's
    metadata["delta_mu_ev"] -- the DELTA_MU_EV column identity that
    _build_campaign already carries unchanged through every reference and
    continuation-child row (``dataclasses.replace`` only overrides the
    fields it's given), so this is a pure post-processing pass: it does not
    change which runs exist or how they're connected, only the chemical
    potentials attached to each.
    """
    runs = tuple(
        replace(
            run,
            chemical_potentials_ev={
                SPECIES[0]: 0.0,
                SPECIES[1]: delta_mu_ref_by_t[run.temperature_k] + float(run.metadata["delta_mu_ev"]),
            },
        )
        for run in campaign.runs
    )
    return replace(campaign, runs=runs)


def profile_workload(
    model: UMAWrapper,
    template: Atoms,
    reference_runs: tuple[RunSpec, ...],
    width: int,
    device: torch.device,
) -> tuple[HybridMCMD, Batch]:
    """Build representative reference-row states for a capacity profile."""
    runs = tuple(
        replace(reference_runs[index % len(reference_runs)], run_id=f"profile.{index}")
        for index in range(width)
    )
    return make_workload(model, template, runs, (None,) * width, device)


def _verify_memory_floor(measurements: list[BatchMeasurement], n_atoms: int) -> None:
    """Reject an implausibly low per-walker memory fit before trusting it.

    A 500-atom walker is expected to reserve about 3.5 GB; scale that
    linearly with atom count and require the profiler's fitted per-walker
    cost stay above ``MEMORY_FLOOR_TOLERANCE`` of the expectation. A profiler
    that under-counts memory (e.g. lazy allocation, a warm-up that skipped
    the largest tensors) would otherwise select a batch width that OOMs once
    a long unattended production run reaches a worse-case composition.
    """
    estimate = SimulationBatchPlanner.infer_memory_model(measurements)
    expected_bytes = EXPECTED_BYTES_PER_ATOM * n_atoms
    floor_bytes = MEMORY_FLOOR_TOLERANCE * expected_bytes
    print(
        f"[memory] n_atoms={n_atoms} fitted {estimate.bytes_per_walker / 1024**3:.2f} GB/walker "
        f"(resident {estimate.model_resident_bytes / 1024**3:.2f} GB); "
        f"expected ~{expected_bytes / 1024**3:.2f} GB/walker"
    )
    if estimate.bytes_per_walker < floor_bytes:
        raise RuntimeError(
            f"profiled per-walker memory {estimate.bytes_per_walker / 1024**3:.2f} GB is "
            f"implausibly below the {floor_bytes / 1024**3:.2f} GB floor scaled from the "
            "~3.5 GB/500-atom reference; refusing to trust this batch-width estimate."
        )


def select_batch_width(
    model: UMAWrapper,
    template: Atoms,
    reference_runs: tuple[RunSpec, ...],
    n_atoms: int,
    device: torch.device,
) -> int:
    """Profile candidate widths, verify the memory floor, and recommend one."""
    planner = SimulationBatchPlanner(
        memory_fraction=BATCH_MEMORY_FRACTION,
        throughput_fraction=BATCH_THROUGHPUT_FRACTION,
    )
    measurements = planner.profile(
        lambda width, dev: profile_workload(model, template, reference_runs, width, dev),
        BATCH_WIDTH_CANDIDATES[n_atoms],
        device=device,
        warmup_blocks=PROFILE_WARMUP_BLOCKS,
        measured_blocks=PROFILE_MEASURED_BLOCKS,
    )
    _verify_memory_floor(measurements, n_atoms)
    width = planner.recommend_width(
        measurements, total_memory_bytes=torch.cuda.get_device_properties(device).total_memory
    )
    print(f"[batch-width] n_atoms={n_atoms} selected width={width} from {measurements}")
    if n_atoms == 500 and not 7 <= width <= 13:
        print(
            f"[batch-width] WARNING: expected width ~10 for 500 atoms on a 40 GB A100, "
            f"measured {width}; check GPU occupancy and model version."
        )
    return width


def _pt_fraction_per_graph(batch: Batch, pt_number: int, n_graphs: int) -> list[float]:
    """Per-graph Pt atomic fraction, one GPU->CPU sync. Ported from
    nvalchemi-toolkit-quest-deploy's scout_sgc_temperature_composition_drift.py
    (different repo, no shared import path).
    """
    dtype = batch.positions.dtype
    pt_mask = (batch.atomic_numbers == pt_number).to(dtype)
    n_pt = torch.bincount(batch.batch_idx, weights=pt_mask, minlength=n_graphs)
    n_total = torch.bincount(batch.batch_idx, minlength=n_graphs).to(dtype)
    return (n_pt / n_total).detach().cpu().tolist()


def _equilibration_gate(series: list[float], window: int) -> dict:
    """PHASE_DIAGRAM_MANUAL.md section 7's live gate, one-shot: compare the
    last two `window`-block windows' means against twice their combined
    standard error. A single check run once at the end of a fixed-length run
    -- NOT that section's full three-consecutive-checks protocol. Identical
    to reference_energy_calibration.py's helper of the same name (different
    repo, no shared import path).
    """
    if len(series) < 2 * window:
        return {
            "resolved": False,
            "reason": f"fewer than {2 * window} blocks recorded ({len(series)})",
        }
    penultimate, last = series[-2 * window : -window], series[-window:]
    mean_a, mean_b = statistics.fmean(penultimate), statistics.fmean(last)
    se_a = statistics.pstdev(penultimate) / (window**0.5) if window > 1 else 0.0
    se_b = statistics.pstdev(last) / (window**0.5) if window > 1 else 0.0
    combined_se = (se_a**2 + se_b**2) ** 0.5
    difference = mean_b - mean_a
    return {
        "resolved": bool(abs(difference) < 2 * combined_se) if combined_se > 0 else None,
        "window_blocks": window,
        "mean_penultimate_window": mean_a,
        "mean_last_window": mean_b,
        "difference": difference,
        "combined_standard_error": combined_se,
    }


def _run_hybrid_with_observables(
    hybrid: HybridMCMD, batch: Batch, n_blocks: int, n_graphs: int, pt_number: int,
) -> tuple[Batch, list[list[float]], list[list[float]]]:
    """Run n_blocks hybrid MC-MD blocks, recording each block's per-graph Pt
    fraction and energy/atom -- the observable series _equilibration_gate
    needs, which a plain ``hybrid.run(batch, n_blocks=n_blocks)`` call
    doesn't expose (it returns only the final batch).

    Reimplements ``HybridMCMD.run``'s loop body locally, using only its
    public ``mc``/``md``/``mc_steps``/``md_steps`` attributes, instead of
    calling ``hybrid.run(batch, n_blocks=1)`` in a Python loop: that would
    re-enter ``with self.md:`` (a CUDA-stream context, see BaseDynamics) and
    re-run ``md.compute``/``mc.synchronize`` once per recorded block instead
    of once for the whole run, roughly doubling the compute cost. This
    version enters the stream context exactly once, matching the original
    method's cost, at the price of one extra small GPU->CPU transfer per
    block for the recorded series (the same per-block-transfer pattern
    reference_energy_calibration.py already uses).
    """
    if n_blocks < 1:
        raise ValueError("n_blocks must be positive")
    if getattr(batch, "forces", None) is None:
        raise ValueError("hybrid MC-MD requires preallocated batch.forces")
    atoms_per_graph = batch.num_nodes // n_graphs
    pt_fraction_series: list[list[float]] = []
    energy_per_atom_series: list[list[float]] = []

    def _record() -> None:
        pt_fraction_series.append(_pt_fraction_per_graph(batch, pt_number, n_graphs))
        energies = batch.energy.flatten().detach().cpu().tolist()
        energy_per_atom_series.append([e / atoms_per_graph for e in energies])

    with hybrid.md:
        hybrid.md.compute(batch)
        hybrid.mc.synchronize(batch)
        for _ in range(n_blocks):
            hybrid.mc.run(batch, n_steps=hybrid.mc_steps)
            hybrid.md.compute(batch)
            hybrid.md.run(batch, n_steps=hybrid.md_steps)
            hybrid.mc.synchronize(batch)
            _record()
    return batch, pt_fraction_series, energy_per_atom_series


def _run_campaign(
    model: UMAWrapper,
    template: Atoms,
    scheduler: CampaignScheduler,
    campaign: CampaignSpec,
    batch_width: int,
    device: torch.device,
    log_path: Path,
) -> None:
    """Run every ready batch to completion, logging per-batch throughput."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with log_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(
                [
                    "run_id",
                    "n_atoms",
                    "batch_width",
                    "n_blocks",
                    "wall_seconds",
                    "walker_blocks_per_second",
                    "mc_acceptance",
                    "continuation",
                    "composition_gate_resolved",
                    "energy_gate_resolved",
                    "resolved",
                ]
            )
        pt_number = SPECIES[1]
        while ready := scheduler.ready_batches(batch_width):
            for runs in ready:
                n_blocks = N_BLOCKS_REFERENCE if runs[0].parent_id is None else N_BLOCKS_CONTINUATION
                parents = tuple(scheduler.parent_state(run, device=device) for run in runs)
                hybrid, batch = make_workload(model, template, runs, parents, device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                result, pt_fraction_series, energy_per_atom_series = _run_hybrid_with_observables(
                    hybrid, batch, n_blocks, len(runs), pt_number,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                acceptance = hybrid.mc.stats.acceptance
                atoms_per_walker = result.num_nodes // len(runs)
                unresolved_run_ids = []
                for graph_index, (run, final_state) in enumerate(zip(runs, result.to_data_list())):
                    scheduler.complete(run.run_id, final_state)
                    composition_series = [block[graph_index] for block in pt_fraction_series]
                    energy_series = [block[graph_index] for block in energy_per_atom_series]
                    composition_gate = _equilibration_gate(composition_series, EQUILIBRATION_WINDOW_BLOCKS)
                    energy_gate = _equilibration_gate(energy_series, EQUILIBRATION_WINDOW_BLOCKS)
                    resolved = (
                        bool(composition_gate.get("resolved")) and bool(energy_gate.get("resolved"))
                        if composition_gate.get("resolved") is not None and energy_gate.get("resolved") is not None
                        else None
                    )
                    if resolved is not True:
                        unresolved_run_ids.append(run.run_id)
                    (scheduler.state_store.root / f"{run.run_id}.equilibration.json").write_text(
                        json.dumps(
                            {
                                "run_id": run.run_id,
                                "temperature_K": run.temperature_k,
                                "n_blocks": n_blocks,
                                "window_blocks": EQUILIBRATION_WINDOW_BLOCKS,
                                "composition_gate": composition_gate,
                                "energy_gate": energy_gate,
                                "resolved": resolved,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                    writer.writerow(
                        [
                            run.run_id,
                            atoms_per_walker,
                            batch_width,
                            n_blocks,
                            f"{elapsed:.3f}",
                            f"{len(runs) * n_blocks / elapsed:.4f}",
                            f"{acceptance:.4f}",
                            run.parent_id is not None,
                            composition_gate.get("resolved"),
                            energy_gate.get("resolved"),
                            resolved,
                        ]
                    )
                handle.flush()
                print(
                    f"Completed {[run.run_id for run in runs]} in {elapsed:.2f} s "
                    f"({len(runs) * n_blocks / elapsed:.3f} walker-blocks/s); "
                    f"SGC acceptance={acceptance:.3f}"
                )
                if unresolved_run_ids:
                    print(
                        f"[equilibration] WARNING: unresolved after {n_blocks} blocks (see "
                        f"<run_id>.equilibration.json for diagnostics): {unresolved_run_ids}"
                    )
    print(
        f"Campaign {campaign.name!r} complete: "
        f"{len(scheduler.completed_ids)} final states in {scheduler.state_store.root}"
    )


def main() -> None:
    """Entry point: run one system size's cooling campaign end-to-end."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-atoms", type=int, required=True, choices=sorted(SIZE_REPEATS))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("hybrid_sgc_npt_checkpoints"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch-width",
        type=int,
        default=None,
        help="Skip auto-profiling and use this batch width instead.",
    )
    parser.add_argument(
        "--reference-energies-json", type=Path, default=None,
        help="reference_energy_calibration.py output. Rebuilds every run's "
        "chemical_potentials_ev as {Au: 0.0, Pt: delta_mu_ref_eV(T) + delta_mu_excess} "
        "at each run's own temperature, replacing the literal, uncalibrated DELTA_MU_EV "
        "sweep. Recommended -- see module docstring 'Chemical-potential calibration'. "
        "Omit to keep the old literal behavior (a warning is printed).",
    )
    parser.add_argument(
        "--allow-unresolved-reference", action="store_true",
        help="Proceed even if a needed temperature's calibration entry has "
        "equilibration_gate.resolved != true. Off by default.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    n_atoms = args.n_atoms
    repeats = SIZE_REPEATS[n_atoms]
    checkpoint_dir = args.checkpoint_root / f"atoms{n_atoms}"
    log_path = args.checkpoint_root / f"atoms{n_atoms}_throughput.csv"

    template = build_ase_structure(TEMPLATE_SYMBOL, CRYSTAL_STRUCTURE, LATTICE_A_ANG, repeats, cubic=CONVENTIONAL_CELL)
    if len(template) != n_atoms:
        raise ValueError(f"expected {n_atoms} atoms, repeats={repeats} built {len(template)}")

    model = UMAWrapper.from_checkpoint(
        CHECKPOINT, task_name=TASK, device=str(device), inference_settings=INFERENCE_SETTINGS
    )

    reference_runs = _build_reference_runs(n_atoms)
    campaign = _build_campaign(n_atoms, reference_runs)
    if args.reference_energies_json is not None:
        symbols = [chemical_symbols[z] for z in SPECIES]
        delta_mu_ref_by_t = _load_delta_mu_ref(
            args.reference_energies_json, TEMPERATURES_K, symbols,
            allow_unresolved=args.allow_unresolved_reference,
        )
        campaign = _apply_calibrated_delta_mu(campaign, delta_mu_ref_by_t)
        print(f"[delta_mu] calibrated from {args.reference_energies_json}")
    else:
        print(
            "[delta_mu] WARNING: no --reference-energies-json given -- using the "
            "literal, uncalibrated DELTA_MU_EV sweep (chemical_potentials_ev="
            "{Au: 0.0, Pt: delta_mu}). This does not locate the real Au-Pt phase "
            "boundary; see module docstring 'Chemical-potential calibration'."
        )
    scheduler = CampaignScheduler(campaign, FinalStateStore(checkpoint_dir))
    print(
        f"Campaign {campaign.name!r}: {len(campaign.runs)} runs, "
        f"{len(scheduler.completed_ids)} already complete."
    )

    if args.batch_width is not None:
        batch_width = args.batch_width
    elif device.type == "cuda":
        batch_width = select_batch_width(model, template, reference_runs, n_atoms, device)
    else:
        batch_width = 1
        print("CUDA unavailable; using batch_width=1")

    _run_campaign(model, template, scheduler, campaign, batch_width, device, log_path)


if __name__ == "__main__":
    main()
