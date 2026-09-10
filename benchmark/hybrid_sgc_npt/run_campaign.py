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

Modes
-----
``--mode cooling`` (default) is the grid described above. ``--mode
delta-mu-scan`` instead runs ``CampaignSpec.delta_mu_scan_runs`` (see also
its convenience wrapper ``delta_mu_scan_from_endpoints``, for a single
temperature with no cross-temperature continuation) across one or more
``--scan-temperatures-k`` (highest first): at the highest
temperature, two fresh A-rich/B-rich endpoints are built from scratch; at
every next (lower) temperature, new endpoints are built as CONTINUATION
CHILDREN of the immediately preceding temperature's own endpoints; then EACH
temperature's own endpoint pair seeds its own two-branch delta_mu_excess scan,
marching from +-``--delta-mu-excess-bracket-ev`` toward 0.0
(``--delta-mu-excess-min-step-ev`` / ``--delta-mu-excess-refine-ratio``;
non-uniform -- see "Delta_mu ladder spacing" below) (PHASE_DIAGRAM_MANUAL.md
section 4, all of steps 1-4). Every temperature is built into ONE combined
``CampaignSpec`` in a single script invocation -- a ``CampaignSpec`` must be
a self-contained dependency graph, so chaining across separate script
invocations by a bare run_id string does not work: the later invocation's
campaign would reference a parent outside its own run set, which
``CampaignSpec`` rejects at construction. A single-element
``--scan-temperatures-k`` degenerates to one from-scratch scan with no
cross-temperature continuation.

Cross-temperature continuation chains each temperature's two SEED runs to
the previous temperature's two seeds only -- not to that temperature's
whole delta_mu ladder -- so a lower temperature becomes ready once the
higher temperature's seeds finish, and from there ``ready_batches`` can
pack that ladder together with the next temperature's seeds (batch
compatibility ignores temperature). Generation 1 is still only the highest
temperature's 2 seeds, though, regardless of how many temperatures are
requested. Pass ``--scan-independent-temperatures`` to drop the chaining
altogether: every temperature's seed pair gets ``parent_id=None`` and all
of them (``2 * len(--scan-temperatures-k)`` walkers) are ready at once in
generation 1, at the cost of a fresh random-composition start at each
temperature instead of a warm start from a neighboring one.

Delta_mu ladder spacing
------------------------
Each branch's ladder is NON-uniform (``_endpoint_ladder``): every new point
sits ``--delta-mu-excess-refine-ratio`` (default 0.5, i.e. halving) of the
way from the previous point to 0.0, so the absolute step size shrinks
geometrically approaching 0.0 and refinement stops once it would fall
tighter than ``--delta-mu-excess-min-step-ev`` (default 0.01 eV) -- coarse
sampling near the safe bracket endpoint (deep in one pure/near-pure phase,
flat response, extra points buy nothing), fine sampling near the
transition (steep response, where the coexistence boundary actually sits).
E.g. bracket=0.2, min_step=0.01: -0.2, -0.1, -0.05, -0.025, -0.0125, 0.0.

Auto-calibration
-----------------
``--reference-energies-json`` is now OPTIONAL in ``--mode delta-mu-scan``
(it remains required-if-given-must-be-complete for ``--mode cooling``, via
``_load_delta_mu_ref``). In scan mode, ``_load_available_delta_mu_ref`` uses
whatever entries the file already has (or nothing, if the flag is omitted)
and reports which requested ``--scan-temperatures-k`` are still missing;
``compute_reference_energies`` then runs the missing temperatures' pure-Au/
pure-Pt NPT calibration in-process, with the model already loaded for the
alloy runs above -- one script invocation, one GPU, no manual pre-step. The
result is written to ``<--checkpoint-root>/atoms<n>/auto_reference_energies.json``
for provenance and reuse (pass it back in as ``--reference-energies-json``
next time to skip recalibrating). Each auto-calibrated temperature's own
``equilibration_gate`` is still checked, exactly as for a file-supplied one
(``--allow-unresolved-reference`` governs both).

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
total, continuation children run ``N_BLOCKS_CONTINUATION``. ``--md-steps-
per-block`` overrides ``MD_STEPS_PER_BLOCK`` for the whole campaign (both
modes); ``--md-steps-per-block 0`` runs pure SGC (MC moves only, no MD
sub-step) for a direct SGC-vs-hybrid comparison -- use a different
``--checkpoint-root`` for each, since run_ids (and therefore every output
filename) are otherwise identical between the two.

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
N_BLOCKS_SCAN_SEED = 200  # Fresh A-rich/B-rich endpoint burn-in; no parent state.
N_BLOCKS_SCAN_STEP = 100  # Per delta_mu_excess step along a scan branch; warm-started.
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
    md_steps_per_block: int = MD_STEPS_PER_BLOCK,
) -> tuple[HybridMCMD, Batch]:
    """Return one per-graph hybrid MC-MD workload for compatible runs.

    The same model object is deliberately passed to both stages: ``HybridMCMD``
    requires every trial energy and MD force to come from one potential.
    ``md_steps_per_block=0`` degenerates to pure SGC (MC moves only, no MD
    sub-step) -- see ``--md-steps-per-block``.
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
    hybrid = HybridMCMD(mc=sgc, md=npt, mc_steps=mc_steps, md_steps=md_steps_per_block)
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


def _endpoint_ladder(seed_value: float, min_step_ev: float, refine_ratio: float = 0.5) -> tuple[float, ...]:
    """Values from ``seed_value`` in toward (and including) 0.0, NON-UNIFORMLY
    spaced -- the per-branch delta_mu_excess ladder for
    ``CampaignSpec.delta_mu_scan_from_endpoints``. ``seed_value`` is the safe,
    extreme starting point (PHASE_DIAGRAM_MANUAL.md section 4 step 1); walking
    it toward 0.0 is section 4 step 3's "sweep outward from each endpoint,"
    i.e. away from the safe corner and toward the transition.

    Each new point is ``refine_ratio`` (default 0.5, i.e. halving) times the
    PREVIOUS point's own remaining distance to 0.0, so the absolute step size
    shrinks geometrically every point: coarse near the endpoint (deep in one
    pure/near-pure phase, where composition responds flatly to delta_mu_excess
    and extra resolution buys nothing) and increasingly fine near 0.0 (the
    interesting, steep-response region close to the coexistence boundary).
    Stops refining once the NEXT point would fall closer than ``min_step_ev``
    to its predecessor, and always lands exactly on 0.0.
    """
    if min_step_ev <= 0.0:
        raise ValueError("min_step_ev must be positive")
    if not 0.0 < refine_ratio < 1.0:
        raise ValueError("refine_ratio must be strictly between 0.0 and 1.0")
    values = [seed_value]
    remaining = abs(seed_value)
    sign = 1.0 if seed_value > 0.0 else -1.0
    while remaining * refine_ratio > min_step_ev:
        remaining *= refine_ratio
        values.append(round(sign * remaining, 10))
    if values[-1] != 0.0:
        values.append(0.0)
    return tuple(values)


def _build_delta_mu_scan_schedule(
    n_atoms: int,
    temperatures_k: Sequence[float],
    delta_mu_ref_by_t: dict[float, float],
    bracket_ev: float,
    min_step_ev: float,
    refine_ratio: float = 0.5,
    *,
    independent_temperatures: bool = False,
) -> tuple[CampaignSpec, tuple[RunSpec, ...]]:
    """Two-branch chemical-potential scans across one or more temperatures,
    combined into ONE ``CampaignSpec`` (PHASE_DIAGRAM_MANUAL.md section 4,
    all of steps 1-4).

    At ``temperatures_k[0]`` (the highest), two fresh A-rich/B-rich endpoints
    are built from scratch (step 1). At every next, lower temperature, new
    endpoints are built as continuation children of the immediately
    preceding temperature's own endpoints via ``RunSpec.parent_id`` --
    ``cooling_from_reference``'s pattern, applied to just the two endpoints
    -- never an interior, history-dependent state (step 4). Independently at
    EACH temperature, that temperature's own endpoint pair seeds a two-branch
    delta_mu_excess scan marching toward 0.0
    (``CampaignSpec.delta_mu_scan_from_endpoints``, step 3). Every run across
    every temperature lands in the one returned ``CampaignSpec`` -- required,
    since a ``CampaignSpec`` must be a self-contained dependency graph (see
    module docstring "Modes").

    ``temperatures_k`` must be strictly decreasing (a single-element sequence
    is a from-scratch scan with no cross-temperature continuation).
    ``delta_mu_ref_by_t`` must already have every requested temperature's
    calibrated reference (section 3), keyed exactly as ``_load_delta_mu_ref``
    returns it. Returns the campaign plus every temperature's two endpoint
    ``RunSpec``s, in schedule order, for batch-width profiling (see
    ``select_batch_width``).

    Concurrency note: cross-temperature continuation only chains each
    temperature's two SEED runs to the previous temperature's two seeds --
    not to that temperature's entire delta_mu ladder. So a lower temperature
    becomes ready as soon as the higher temperature's seeds finish, not after
    its whole scan finishes; from there on, ``ready_batches`` can pack that
    temperature's own ladder steps together with the next temperature's
    seeds (same ``compatibility_key``, which ignores temperature). The one
    real bottleneck is generation 1: only the highest temperature's 2 seeds
    are ready at the very start, so early batch width is 2 regardless of how
    many temperatures are requested. Pass ``independent_temperatures=True``
    to remove cross-temperature continuation altogether -- every
    temperature's seed pair becomes ``parent_id=None`` and all of them are
    ready simultaneously in generation 1 (``2 * len(temperatures_k)`` walkers
    at once), at the cost of each temperature's endpoints starting from a
    fresh random composition instead of a warm start from a nearby T.
    """
    temperatures = tuple(float(t) for t in temperatures_k)
    if not temperatures:
        raise ValueError("at least one temperature is required")
    if any(left <= right for left, right in zip(temperatures, temperatures[1:])):
        raise ValueError("temperatures_k must decrease strictly")
    if bracket_ev <= 0.0:
        raise ValueError("bracket_ev must be positive")
    missing = set(temperatures) - set(delta_mu_ref_by_t)
    if missing:
        raise ValueError(f"delta_mu_ref_by_t has no entry for {sorted(missing)!r}")

    ladder_a_excess = _endpoint_ladder(-bracket_ev, min_step_ev, refine_ratio)
    ladder_b_excess = _endpoint_ladder(bracket_ev, min_step_ev, refine_ratio)

    runs: list[RunSpec] = []
    endpoints: list[RunSpec] = []
    parent_a: str | None = None
    parent_b: str | None = None
    for temperature_k in temperatures:
        delta_mu_ref_ev = delta_mu_ref_by_t[temperature_k]
        ladder_a = tuple(delta_mu_ref_ev + excess for excess in ladder_a_excess)
        ladder_b = tuple(delta_mu_ref_ev + excess for excess in ladder_b_excess)
        seed_a = RunSpec(
            run_id=f"atoms{n_atoms}.T{temperature_k:g}.Arich",
            temperature_k=temperature_k,
            pressure_ev_per_a3=PRESSURE_EV_PER_A3,
            chemical_potentials_ev={SPECIES[0]: 0.0, SPECIES[1]: ladder_a[0]},
            parent_id=parent_a,
            species=SPECIES,
            batch_group=f"atoms{n_atoms}",
            metadata={
                "seed_delta_mu_excess_ev": ladder_a_excess[0],
                "n_atoms": n_atoms,
                "branch": "A_rich",
                "pt_fraction": 0.05,
            },
        )
        seed_b = RunSpec(
            run_id=f"atoms{n_atoms}.T{temperature_k:g}.Brich",
            temperature_k=temperature_k,
            pressure_ev_per_a3=PRESSURE_EV_PER_A3,
            chemical_potentials_ev={SPECIES[0]: 0.0, SPECIES[1]: ladder_b[0]},
            parent_id=parent_b,
            species=SPECIES,
            batch_group=f"atoms{n_atoms}",
            metadata={
                "seed_delta_mu_excess_ev": ladder_b_excess[0],
                "n_atoms": n_atoms,
                "branch": "B_rich",
                "pt_fraction": 0.95,
            },
        )
        # delta_mu_scan_runs (not delta_mu_scan_from_endpoints): when
        # temperature_k is not the first in the schedule, seed_a/seed_b's own
        # parent_id points at the PREVIOUS temperature's endpoints, which are
        # not part of this call's own seeds -- delta_mu_scan_from_endpoints
        # would reject that immediately. Every temperature's runs are
        # accumulated here and validated together in exactly one
        # CampaignSpec(...) call below, once every parent_id (both the
        # within-temperature delta_mu ladder and the cross-temperature
        # endpoint chain) resolves inside the same combined run set.
        runs.extend(
            CampaignSpec.delta_mu_scan_runs((seed_a, seed_b), (ladder_a, ladder_b), SPECIES[1])
        )
        endpoints.extend((seed_a, seed_b))
        if not independent_temperatures:
            # Chains to the SEED, not the last ladder child (see docstring above).
            parent_a, parent_b = seed_a.run_id, seed_b.run_id

    campaign = CampaignSpec(runs=tuple(runs), name=f"aupt_{n_atoms}atoms_deltamu_scan_schedule")
    return campaign, tuple(endpoints)


def _build_pure_element_template(symbol: str, n_atoms: int) -> Atoms:
    """A pure-element supercell at ASE's own reference lattice constant,
    matching the alloy's crystal structure/repeat count. Deliberately omits
    ``a=`` (unlike ``build_ase_structure``, which always uses the shared,
    Au-based ``LATTICE_A_ANG``): each end-member should start near its OWN
    equilibrium volume -- reusing the shared lattice constant would just
    reintroduce the volume-relaxation skew this calibration removes. NPT's
    barostat does the actual equilibration; this only sets a physically
    reasonable starting point. Adapted from nvalchemi-toolkit-quest-deploy's
    reference_energy_calibration.py (different repo, no shared import path;
    duplicated here, rather than imported, so ``--mode delta-mu-scan`` can
    auto-calibrate in-process with the model already loaded for the alloy
    runs -- see that script's module docstring for the full physical
    rationale, and ``compute_reference_energies`` below).
    """
    repeats = SIZE_REPEATS[n_atoms]
    unit_cell = bulk(symbol, crystalstructure=CRYSTAL_STRUCTURE, cubic=CONVENTIONAL_CELL)
    template = unit_cell * repeats
    if len(template) != n_atoms:
        raise ValueError(f"expected {n_atoms} atoms for pure {symbol}, repeats={repeats} built {len(template)}")
    return template


def _pure_element_endpoint(
    template: Atoms, temperature_k: float, velocity_seed: int, device: torch.device,
) -> AtomicData:
    """One pure-element ``AtomicData`` with a Maxwell-Boltzmann velocity draw
    at ``temperature_k``. Adapted from reference_energy_calibration.py."""
    data = AtomicData.from_atoms(template, device=device)
    n = data.num_nodes
    data.atomic_masses = None
    data.use_default_masses()
    generator = torch.Generator(device=device).manual_seed(velocity_seed)
    velocity_std = torch.sqrt(torch.as_tensor(KB_EV * temperature_k, device=device) / data.atomic_masses)
    data.velocities = torch.randn((n, 3), device=device, generator=generator) * velocity_std[:, None]
    data.velocities -= data.velocities.mean(dim=0, keepdim=True)
    data.forces = torch.zeros_like(data.positions)
    data.energy = torch.zeros(1, 1, device=device)
    data.stress = torch.zeros(1, 3, 3, device=device)
    return data


def _cell_volumes(batch: Batch) -> torch.Tensor:
    """Per-graph cell volume (A^3), robust to a possibly-unbatched cell tensor."""
    cell = batch.cell if batch.cell.ndim == 3 else batch.cell.unsqueeze(0)
    return torch.linalg.det(cell)


def compute_reference_energies(
    model: UMAWrapper,
    temperatures_k: Sequence[float],
    n_atoms: int,
    *,
    n_blocks: int,
    md_steps_per_block: int,
    equilibration_window_blocks: int,
    velocity_seed: int,
    device: torch.device,
) -> dict:
    """Pure Au/Pt NPT reference-energy calibration, in-process, on an
    ALREADY-LOADED model -- PHASE_DIAGRAM_MANUAL.md section 6.2's
    ``delta_mu_ref(T) = g_Pt(T) - g_Au(T)``, computed at every requested
    temperature. This is what lets ``--mode delta-mu-scan`` run completely
    automatically on a single GPU: no second checkpoint load, no separate
    script invocation, no manually-prepared ``reference_energies.json``.

    Ported from nvalchemi-toolkit-quest-deploy's
    reference_energy_calibration.py (different repo, no shared import path;
    kept here, rather than imported, precisely so this project's toolkit-side
    entry point stays self-sufficient -- see that script's module docstring
    for the full physical rationale: each pure element is equilibrated
    independently, from its own ASE reference lattice constant, through a
    real NPT trajectory at the target temperature and the alloy's own
    pressure/thermostat/barostat settings, one graph per (element,
    temperature) in a single batch).

    Returns the same ``reference_energies.json``-shaped dict that script
    writes (``reference[T][symbol]`` plus, for the 2-species case,
    ``delta_mu_ref_eV`` per temperature) -- the caller decides whether/where
    to persist it and whether its ``equilibration_gate`` is trustworthy
    enough to use.
    """
    temperatures = tuple(float(t) for t in temperatures_k)
    if not temperatures:
        raise ValueError("at least one temperature is required")
    symbols = [chemical_symbols[z] for z in SPECIES]
    templates = {
        number: _build_pure_element_template(symbol, n_atoms) for number, symbol in zip(SPECIES, symbols)
    }

    graph_index: list[tuple[int, float]] = [(number, t) for number in SPECIES for t in temperatures]
    data_list = [
        _pure_element_endpoint(templates[number], t, velocity_seed + i, device)
        for i, (number, t) in enumerate(graph_index)
    ]
    batch = Batch.from_data_list(data_list)
    n_graphs = len(graph_index)

    temperatures_tensor = torch.tensor([t for _, t in graph_index], device=device)
    npt = NPT(
        model=model,
        dt=DT_FS,
        temperature=temperatures_tensor,
        pressure=torch.full((n_graphs,), PRESSURE_EV_PER_A3, device=device),
        thermostat_time=THERMOSTAT_TIME_FS,
        barostat_time=BAROSTAT_TIME_FS,
        pressure_coupling="isotropic",
    )

    energy_per_atom_series: list[list[float]] = [[] for _ in range(n_graphs)]
    volume_per_atom_series: list[list[float]] = [[] for _ in range(n_graphs)]

    def _record() -> None:
        energies = batch.energy.detach().reshape(-1)
        counts = batch.num_nodes_per_graph.to(energies.dtype)
        volumes = _cell_volumes(batch)
        for i in range(n_graphs):
            energy_per_atom_series[i].append(float(energies[i] / counts[i]))
            volume_per_atom_series[i].append(float(volumes[i] / counts[i]))

    with npt:
        npt.compute(batch)
        _record()
        for _ in range(n_blocks):
            npt.run(batch, n_steps=md_steps_per_block)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            _record()

    window = equilibration_window_blocks
    n_rep = SIZE_REPEATS[n_atoms][0]
    reference: dict[str, dict] = {}
    for i, (number, t) in enumerate(graph_index):
        symbol = chemical_symbols[number]
        energy_gate = _equilibration_gate(energy_per_atom_series[i][1:], window)
        volume_gate = _equilibration_gate(volume_per_atom_series[i][1:], window)
        resolved = (
            bool(energy_gate.get("resolved")) and bool(volume_gate.get("resolved"))
            if energy_gate.get("resolved") is not None and volume_gate.get("resolved") is not None
            else None
        )
        tail = min(window, len(energy_per_atom_series[i]))
        mean_energy = statistics.fmean(energy_per_atom_series[i][-tail:])
        mean_volume = statistics.fmean(volume_per_atom_series[i][-tail:])
        se_energy = (
            statistics.pstdev(energy_per_atom_series[i][-tail:]) / (tail**0.5) if tail > 1 else float("nan")
        )
        lattice_a = (mean_volume * n_atoms) ** (1 / 3) / n_rep
        reference.setdefault(f"{t:g}", {})[symbol] = {
            "atomic_number": number,
            "energy_eV_per_atom": mean_energy,
            "energy_eV_per_atom_standard_error": se_energy,
            "volume_A3_per_atom": mean_volume,
            "lattice_constant_a_ang": lattice_a,
            "averaging_window_blocks": tail,
            "equilibration_gate": {"energy": energy_gate, "volume": volume_gate, "resolved": resolved},
        }

    s0, s1 = symbols
    for t_entry in reference.values():
        t_entry["delta_mu_ref_eV"] = t_entry[s1]["energy_eV_per_atom"] - t_entry[s0]["energy_eV_per_atom"]
        t_entry["delta_mu_ref_definition"] = f"mu({s1}) - mu({s0}), per PHASE_DIAGRAM_MANUAL.md section 6.2"

    return {
        "checkpoint": CHECKPOINT,
        "task_name": TASK,
        "species": list(SPECIES),
        "symbols": symbols,
        "crystal_structure": CRYSTAL_STRUCTURE,
        "conventional_cell": CONVENTIONAL_CELL,
        "n_atoms_per_graph": n_atoms,
        "pressure_ev_per_a3": PRESSURE_EV_PER_A3,
        "n_blocks": n_blocks,
        "md_steps_per_block": md_steps_per_block,
        "equilibration_window_blocks": window,
        "reference": reference,
    }


def _load_available_delta_mu_ref(
    reference_path: Path | None, temperatures_k: Sequence[float], symbols: list[str], *, allow_unresolved: bool,
) -> tuple[dict[float, float], tuple[float, ...]]:
    """Like ``_load_delta_mu_ref``, but tolerant of a MISSING temperature:
    returns whatever calibrated ``delta_mu_ref_eV`` values ARE already in
    ``reference_path`` (or an empty dict, with every requested temperature
    reported missing, if ``reference_path`` is None), plus the requested
    temperatures still needing calibration, highest first (ready to feed
    straight into ``compute_reference_energies``). A species-order mismatch
    or an unresolved ``equilibration_gate`` (without ``allow_unresolved``)
    remain fatal -- those are correctness problems, not merely absent data,
    so silently proceeding would reproduce the uncalibrated-delta_mu problem
    this whole calibration step exists to fix; only a temperature's outright
    absence is treated as "auto-calibrate it" rather than an error.
    """
    if reference_path is None:
        return {}, tuple(sorted({float(t) for t in temperatures_k}, reverse=True))
    reference_data = json.loads(reference_path.read_text())
    reference = reference_data["reference"]
    if list(reference_data.get("symbols", [])) != list(symbols):
        raise ValueError(
            f"{reference_path} was calibrated for symbols {reference_data.get('symbols')}, "
            f"this run uses {symbols} -- refusing to mix calibrations across species orderings"
        )
    found: dict[float, float] = {}
    missing: list[float] = []
    for t in temperatures_k:
        t = float(t)
        entry = reference.get(f"{t:g}")
        if entry is None:
            missing.append(t)
            continue
        if "delta_mu_ref_eV" not in entry:
            raise ValueError(
                f"{reference_path}'s T={t:g} K entry has no delta_mu_ref_eV "
                "(only written for the 2-species case)"
            )
        gate_status = {symbol: entry[symbol]["equilibration_gate"]["resolved"] for symbol in symbols}
        if not allow_unresolved and not all(status is True for status in gate_status.values()):
            raise ValueError(
                f"T={t:g} K: equilibration_gate.resolved is {gate_status}, not all True -- "
                "this reference value isn't trustworthy enough to anchor delta_mu on. Pass "
                "--allow-unresolved-reference to proceed anyway (not recommended)."
            )
        found[t] = entry["delta_mu_ref_eV"]
    return found, tuple(sorted(set(missing), reverse=True))


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
    md_steps_per_block: int = MD_STEPS_PER_BLOCK,
) -> tuple[HybridMCMD, Batch]:
    """Build representative reference-row states for a capacity profile."""
    runs = tuple(
        replace(reference_runs[index % len(reference_runs)], run_id=f"profile.{index}")
        for index in range(width)
    )
    return make_workload(model, template, runs, (None,) * width, device, md_steps_per_block)


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
    md_steps_per_block: int = MD_STEPS_PER_BLOCK,
) -> int:
    """Profile candidate widths, verify the memory floor, and recommend one."""
    planner = SimulationBatchPlanner(
        memory_fraction=BATCH_MEMORY_FRACTION,
        throughput_fraction=BATCH_THROUGHPUT_FRACTION,
    )
    measurements = planner.profile(
        lambda width, dev: profile_workload(model, template, reference_runs, width, dev, md_steps_per_block),
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
    n_blocks_root: int,
    n_blocks_continuation: int,
    md_steps_per_block: int = MD_STEPS_PER_BLOCK,
) -> None:
    """Run every ready batch to completion, logging per-batch throughput.

    ``n_blocks_root`` applies to a run with no parent (a fresh reference row
    or scan endpoint); ``n_blocks_continuation`` applies to every run with a
    parent_id, whether that continuation is across temperature (cooling),
    across delta_mu_excess (a scan branch step), or an endpoint carried
    forward to a new temperature (scan section 4 steps 2/4) -- the budget
    only depends on whether a run is warm-started, not which axis moved.
    ``md_steps_per_block=0`` runs pure SGC (MC moves only, no MD sub-step)
    for every batch -- see ``--md-steps-per-block``.
    """
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
                n_blocks = n_blocks_root if runs[0].parent_id is None else n_blocks_continuation
                parents = tuple(scheduler.parent_state(run, device=device) for run in runs)
                hybrid, batch = make_workload(model, template, runs, parents, device, md_steps_per_block)
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
                                "chemical_potentials_ev": {
                                    chemical_symbols[z]: run.chemical_potentials_ev[z] for z in SPECIES
                                },
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
    """Entry point: run one system size's cooling campaign, or its
    two-branch delta_mu-scan schedule across one or more temperatures
    (--mode delta-mu-scan), end-to-end."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-atoms", type=int, required=True, choices=sorted(SIZE_REPEATS))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("hybrid_sgc_npt_checkpoints"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--md-steps-per-block", type=int, default=None,
        help="MD steps per hybrid block for the actual campaign runs (both "
        "--mode cooling and --mode delta-mu-scan) -- NOT the calibration "
        "sub-step (see --calibration-md-steps-per-block for that). Defaults "
        "to MD_STEPS_PER_BLOCK (currently 50). Pass 0 for pure SGC (MC moves "
        "only, no MD dynamics) -- e.g. to compare a pure-SGC job against a "
        "hybrid job at the same --checkpoint-root-adjacent path (use "
        "DIFFERENT --checkpoint-root values for the two, since run_ids and "
        "therefore output filenames are otherwise identical).",
    )
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
    parser.add_argument(
        "--mode", choices=["cooling", "delta-mu-scan"], default="cooling",
        help="'cooling' (default): the existing 3000->1600 K, fixed-delta_mu "
        "campaign. 'delta-mu-scan': a two-branch (A-rich/B-rich) "
        "delta_mu_excess continuation across one or more "
        "--scan-temperatures-k (PHASE_DIAGRAM_MANUAL.md section 4, all of "
        "steps 1-4). Fully automatic: any requested temperature missing "
        "from --reference-energies-json (or its complete absence) is "
        "calibrated in-process first, on the same GPU and the same already-"
        "loaded model, via compute_reference_energies -- see 'Auto-"
        "calibration' below.",
    )
    parser.add_argument(
        "--scan-temperatures-k", type=float, nargs="+", default=None,
        help="--mode delta-mu-scan only: one or more temperatures, HIGHEST "
        "FIRST. A single value is a from-scratch scan at that temperature; "
        "two or more chain each next (lower) temperature's two endpoints as "
        "continuation children of the previous temperature's own endpoints "
        "(section 4 steps 2 and 4), UNLESS --scan-independent-temperatures "
        "is passed -- run every temperature in this ONE invocation, not as "
        "separate script calls (see module docstring 'Modes'). Required in "
        "delta-mu-scan mode.",
    )
    parser.add_argument(
        "--scan-independent-temperatures", action="store_true",
        help="--mode delta-mu-scan only: drop cross-temperature continuation "
        "-- every requested temperature's A-rich/B-rich seed pair gets "
        "parent_id=None instead of chaining to the previous temperature's "
        "seeds, so all 2 * len(--scan-temperatures-k) seeds are ready "
        "simultaneously in generation 1 instead of only the highest "
        "temperature's 2. Trades a warm-started composition at each lower "
        "temperature for full generation-1 batch width. Off by default.",
    )
    parser.add_argument(
        "--delta-mu-excess-bracket-ev", type=float, default=0.2,
        help="--mode delta-mu-scan only: half-width of the scan -- the seed "
        "endpoints sit at delta_mu_excess = -bracket (A-rich) and +bracket "
        "(B-rich), each marching toward 0.0. Default 0.2 eV.",
    )
    parser.add_argument(
        "--delta-mu-excess-min-step-ev", type=float, default=0.01,
        help="--mode delta-mu-scan only: finest (last, closest to 0.0) step "
        "size in each branch's NON-UNIFORM ladder -- see "
        "--delta-mu-excess-refine-ratio. Default 0.01 eV.",
    )
    parser.add_argument(
        "--delta-mu-excess-refine-ratio", type=float, default=0.5,
        help="--mode delta-mu-scan only: each new ladder point sits this "
        "fraction of the way from the previous point to 0.0 (default 0.5, "
        "i.e. halving), so steps shrink geometrically approaching the "
        "transition and stay coarse near the safe bracket endpoint. Must be "
        "in (0, 1).",
    )
    parser.add_argument(
        "--calibration-n-blocks", type=int, default=100,
        help="--mode delta-mu-scan only, auto-calibration: MD blocks for each "
        "missing temperature's pure-Au/pure-Pt NPT run. Default 100 (matches "
        "reference_energy_calibration.py's own default).",
    )
    parser.add_argument(
        "--calibration-md-steps-per-block", type=int, default=None,
        help="--mode delta-mu-scan only, auto-calibration: MD steps/block "
        "for the calibration run. Defaults to MD_STEPS_PER_BLOCK (the "
        "alloy's own value).",
    )
    parser.add_argument(
        "--calibration-equilibration-window-blocks", type=int, default=25,
        help="--mode delta-mu-scan only, auto-calibration: equilibration-gate "
        "window, PHASE_DIAGRAM_MANUAL.md section 7. Default 25.",
    )
    parser.add_argument(
        "--calibration-velocity-seed", type=int, default=None,
        help="--mode delta-mu-scan only, auto-calibration: base velocity "
        "seed. Defaults to SEED (the alloy's own base seed).",
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
    md_steps_per_block = (
        args.md_steps_per_block if args.md_steps_per_block is not None else MD_STEPS_PER_BLOCK
    )
    print(f"[hybrid] md_steps_per_block={md_steps_per_block}" + (" (pure SGC)" if md_steps_per_block == 0 else ""))

    if args.mode == "delta-mu-scan":
        if not args.scan_temperatures_k:
            parser.error("--mode delta-mu-scan requires --scan-temperatures-k")
        symbols = [chemical_symbols[z] for z in SPECIES]
        delta_mu_ref_by_t, missing_temperatures = _load_available_delta_mu_ref(
            args.reference_energies_json, args.scan_temperatures_k, symbols,
            allow_unresolved=args.allow_unresolved_reference,
        )
        if missing_temperatures:
            print(
                f"[calibration] no reference energy for T={list(missing_temperatures)} K -- "
                f"auto-calibrating pure {symbols[0]}/{symbols[1]} there now (same process, "
                "same GPU, same model already loaded above)"
            )
            calibration = compute_reference_energies(
                model, missing_temperatures, n_atoms,
                n_blocks=args.calibration_n_blocks,
                md_steps_per_block=args.calibration_md_steps_per_block or MD_STEPS_PER_BLOCK,
                equilibration_window_blocks=args.calibration_equilibration_window_blocks,
                velocity_seed=args.calibration_velocity_seed or SEED,
                device=device,
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            calibration_path = checkpoint_dir / "auto_reference_energies.json"
            calibration_path.write_text(json.dumps(calibration, indent=2) + "\n")
            for t in missing_temperatures:
                entry = calibration["reference"][f"{t:g}"]
                gate_status = {symbol: entry[symbol]["equilibration_gate"]["resolved"] for symbol in symbols}
                if not args.allow_unresolved_reference and not all(v is True for v in gate_status.values()):
                    raise SystemExit(
                        f"[calibration] auto-calibration at T={t:g} K did not pass the "
                        f"equilibration gate ({gate_status}) -- rerun with a larger "
                        "--calibration-n-blocks, or pass --allow-unresolved-reference to "
                        "proceed anyway (not recommended)."
                    )
                delta_mu_ref_by_t[t] = entry["delta_mu_ref_eV"]
                print(f"  T={t:g} K: delta_mu_ref = {entry['delta_mu_ref_eV']:.6f} eV (auto-calibrated)")
            print(f"[calibration] wrote {calibration_path}")
        campaign, endpoints = _build_delta_mu_scan_schedule(
            n_atoms, args.scan_temperatures_k, delta_mu_ref_by_t,
            args.delta_mu_excess_bracket_ev, args.delta_mu_excess_min_step_ev,
            args.delta_mu_excess_refine_ratio,
            independent_temperatures=args.scan_independent_temperatures,
        )
        reference_runs = endpoints
        print(f"[delta_mu] scan ready across T={args.scan_temperatures_k} K")
    else:
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
        batch_width = select_batch_width(model, template, reference_runs, n_atoms, device, md_steps_per_block)
    else:
        batch_width = 1
        print("CUDA unavailable; using batch_width=1")

    n_blocks_root = N_BLOCKS_SCAN_SEED if args.mode == "delta-mu-scan" else N_BLOCKS_REFERENCE
    n_blocks_continuation = N_BLOCKS_SCAN_STEP if args.mode == "delta-mu-scan" else N_BLOCKS_CONTINUATION
    _run_campaign(
        model, template, scheduler, campaign, batch_width, device, log_path,
        n_blocks_root, n_blocks_continuation, md_steps_per_block,
    )


if __name__ == "__main__":
    main()
