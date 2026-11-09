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
"""Au-Pt hybrid SGC-NPT campaigns: T x delta_mu grid, delta_mu scans, boundary tracing.

One process runs one system size (``--n-atoms``) end to end. Completed state
points are checkpointed with ``FinalStateStore`` and skipped on restart.

Modes
-----
``cooling`` (default)
    8 temperatures (3000 -> 1600 K) x 11 delta_mu columns (-1 -> 1 eV); each
    column cools from a 3000 K reference run through ``RunSpec.parent_id``.
``delta-mu-scan``
    Two-branch (A-rich / B-rich) delta_mu_excess ladders at every
    ``--scan-temperatures-k`` (highest first), endpoints chained across
    temperatures; missing reference energies are calibrated in-process.
``trace-boundary``
    Integrate the coexistence line from one known point (``boundary_tracer.py``).

Pass ``--reference-energies-json`` (or let delta-mu-scan calibrate) so that
delta_mu is an excess over the calibrated delta_mu_ref(T); the literal sweep
drives every run to a pure end member. Background (calibration, ladder spacing,
continuation, equilibration gate, batch width) is in this directory's
``README.md``; shared helpers are in ``campaign_lib.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch
import torch._inductor.config
from ase import Atoms
from ase.data import chemical_symbols
from campaign_lib import (
    CHECKPOINT,
    CONVENTIONAL_CELL,
    CRYSTAL_STRUCTURE,
    EQUILIBRATION_WINDOW_BLOCKS,
    LATTICE_A_ANG,
    MC_ENERGY_ONLY,
    MC_STEP_FRACTION,
    MD_STEPS_PER_BLOCK,
    PRESSURE_EV_PER_A3,
    SEED,
    SIZE_REPEATS,
    SPECIES,
    TASK,
    TEMPLATE_SYMBOL,
    NvalchemiTraceEngine,
    build_ase_structure,
    compute_reference_energies,
    equilibration_gate,
    make_workload,
    resolve_inference_settings,
    run_hybrid_with_observables,
)

from nvalchemi.data import Batch
from nvalchemi.hybrid import HybridMCMD
from nvalchemi.models.uma import UMAWrapper
from nvalchemi.scheduling import (
    BatchMeasurement,
    CampaignScheduler,
    CampaignSpec,
    FinalStateStore,
    RunSpec,
    SimulationBatchPlanner,
)

# Inductor's pad_mm pass benchmarks real, padded copies of large batched
# matmul operands while compiling (a ~20 GiB one-shot allocation at wide
# batches), so a compiled model can OOM during compilation at a batch width
# whose steady-state run fits. Applied in main(), not at import, so modules
# that import this one for its constants keep their own inductor config.
INDUCTOR_SHAPE_PADDING = False
BATCH_WIDTH_CANDIDATES: dict[int, tuple[int, ...]] = {
    108: (4, 8, 16, 24, 32, 48, 64),
    256: (2, 4, 8, 12, 16, 24, 32),
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
N_BLOCKS_REFERENCE = 200  # 50 equilibration + 150 production; no parent state.
N_BLOCKS_CONTINUATION = 100  # 50 equilibration + 50 production; warm-started.
N_BLOCKS_SCAN_SEED = 200  # Fresh A-rich/B-rich endpoint burn-in; no parent state.
N_BLOCKS_SCAN_STEP = 100  # Per delta_mu_excess step along a scan branch; warm-started.
USE_CONTINUATION = True
BATCH_MEMORY_FRACTION = 0.85
BATCH_THROUGHPUT_FRACTION = 0.95
PROFILE_WARMUP_BLOCKS = 1
PROFILE_MEASURED_BLOCKS = 2
EXPECTED_BYTES_PER_ATOM = (3.5 * 1024**3) / 500  # ~3.5 GB reserved @ 500 atoms.
MEMORY_FLOOR_TOLERANCE = 0.75  # Require >= 75% of the atom-count-scaled estimate.


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
        replace(
            reference,
            run_id=f"{reference.run_id}.T{temperature:g}",
            temperature_k=temperature,
        )
        for reference in reference_runs
        for temperature in TEMPERATURES_K
    )
    return CampaignSpec(runs=runs, name=f"aupt_{n_atoms}atoms_independent")


def _endpoint_ladder(
    seed_value: float,
    min_step_ev: float,
    refine_ratio: float = 0.5,
    overshoot_ev: float = 0.0,
) -> tuple[float, ...]:
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

    ``overshoot_ev`` (0.0 = the old behaviour) continues the ladder past 0.0 by
    that much, with the step size doubling away from 0.0, so both branches cover
    a shared delta_mu window and hysteresis becomes measurable.
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
    if overshoot_ev > 0.0:
        # Continue PAST 0.0 into the opposite branch's territory, so the two
        # branches share a dmu window instead of only touching at 0.0. Without
        # that overlap neither walker is ever metastable at a dmu the other one
        # also visited, so a first-order transition cannot show hysteresis and
        # van de Walle & Asta's phi_alpha = phi_gamma crossing has nothing to
        # bracket (exactly what left the 1200/1400 K scans undecidable).
        # Spacing doubles away from 0.0: the finest resolution belongs next to
        # the transition, and points deep inside the other phase's basin are
        # only there to extend the metastable branch.
        step = min_step_ev
        total = 0.0
        while total < overshoot_ev - 1e-12:
            total = min(total + step, overshoot_ev)
            values.append(round(-sign * total, 10))
            step *= 2.0
    return tuple(values)


def _build_delta_mu_scan_schedule(
    n_atoms: int,
    temperatures_k: Sequence[float],
    delta_mu_ref_by_t: dict[float, float],
    bracket_ev: float,
    min_step_ev: float,
    refine_ratio: float = 0.5,
    overshoot_ev: float = 0.0,
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
    README "Modes").

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

    ladder_a_excess = _endpoint_ladder(
        -bracket_ev, min_step_ev, refine_ratio, overshoot_ev
    )
    ladder_b_excess = _endpoint_ladder(
        bracket_ev, min_step_ev, refine_ratio, overshoot_ev
    )

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
            CampaignSpec.delta_mu_scan_runs(
                (seed_a, seed_b), (ladder_a, ladder_b), SPECIES[1]
            )
        )
        endpoints.extend((seed_a, seed_b))
        if not independent_temperatures:
            # Chains to the SEED, not the last ladder child (see docstring above).
            parent_a, parent_b = seed_a.run_id, seed_b.run_id

    campaign = CampaignSpec(
        runs=tuple(runs), name=f"aupt_{n_atoms}atoms_deltamu_scan_schedule"
    )
    return campaign, tuple(endpoints)


def _load_available_delta_mu_ref(
    reference_path: Path | None,
    temperatures_k: Sequence[float],
    symbols: list[str],
    *,
    allow_unresolved: bool,
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
        gate_status = {
            symbol: entry[symbol]["equilibration_gate"]["resolved"]
            for symbol in symbols
        }
        if not allow_unresolved and not all(
            status is True for status in gate_status.values()
        ):
            raise ValueError(
                f"T={t:g} K: equilibration_gate.resolved is {gate_status}, not all True -- "
                "this reference value isn't trustworthy enough to anchor delta_mu on. Pass "
                "--allow-unresolved-reference to proceed anyway (not recommended)."
            )
        found[t] = entry["delta_mu_ref_eV"]
    return found, tuple(sorted(set(missing), reverse=True))


def _load_delta_mu_ref(
    reference_path: Path,
    temperatures_k: Sequence[float],
    symbols: list[str],
    *,
    allow_unresolved: bool,
) -> dict[float, float]:
    """delta_mu_ref_eV per requested temperature, from
    reference_energy_calibration.py's reference_energies.json (this file's
    own TEMPERATURES_K, imported there as ``campaign.TEMPERATURES_K``, so the
    default calibration grid already covers every temperature this campaign
    needs). Raises on a missing temperature, a species mismatch, or an
    unresolved equilibration_gate (unless allow_unresolved) -- silently
    proceeding on any of those would reproduce exactly the uncalibrated-
    delta_mu problem this loader exists to fix.
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
            symbol: entry[symbol]["equilibration_gate"]["resolved"]
            for symbol in symbols
        }
        if not allow_unresolved and not all(
            status is True for status in gate_status.values()
        ):
            raise ValueError(
                f"T={t:g} K: equilibration_gate.resolved is {gate_status}, not all True -- "
                "this reference value isn't trustworthy enough to anchor delta_mu on. Pass "
                "--allow-unresolved-reference to proceed anyway (not recommended)."
            )
        resolved[t] = entry["delta_mu_ref_eV"]
    return resolved


def _apply_calibrated_delta_mu(
    campaign: CampaignSpec,
    delta_mu_ref_by_t: dict[float, float],
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
                SPECIES[1]: delta_mu_ref_by_t[run.temperature_k]
                + float(run.metadata["delta_mu_ev"]),
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
    mc_step_fraction: float = MC_STEP_FRACTION,
    mc_energy_only: bool = MC_ENERGY_ONLY,
) -> tuple[HybridMCMD, Batch]:
    """Build representative reference-row states for a capacity profile."""
    runs = tuple(
        replace(reference_runs[index % len(reference_runs)], run_id=f"profile.{index}")
        for index in range(width)
    )
    return make_workload(
        model,
        template,
        runs,
        (None,) * width,
        device,
        md_steps_per_block,
        mc_step_fraction,
        mc_energy_only,
    )


def _verify_memory_floor(measurements: list[BatchMeasurement], n_atoms: int) -> None:
    """Flag (no longer reject) an implausibly low per-walker memory fit.

    Originally raised when the profiler's fitted per-walker cost fell below
    ``MEMORY_FLOOR_TOLERANCE`` of a floor scaled LINEARLY from the ~3.5
    GB/500-atom reference (a profiler that under-counts memory -- lazy
    allocation, a warm-up that skipped the largest tensors -- would
    otherwise select a batch width that OOMs once a long unattended
    production run reaches a worst-case composition). Downgraded to a
    warning: the linear-in-n_atoms floor itself doesn't hold at larger
    sizes -- a dedicated memory-profile run (far more
    samples than this profiler's PROFILE_MEASURED_BLOCKS=2) measured
    real per-walker costs of 3.96 GB/500 atoms, 8.59 GB/1372, but only
    1.71 GB/2048 -- resident cost dominates and marginal per-walker cost
    grows sublinearly, so the naive floor overshoots badly by 2048 atoms
    (demanding 10.75 GB where 1.71 GB is the measured reality) and raising
    on that mismatch blocked otherwise-valid runs. Left in as a print so a
    genuinely bad profiler run (lazy allocation etc.) is still visible.
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
        print(
            f"[memory] WARNING: profiled per-walker memory {estimate.bytes_per_walker / 1024**3:.2f} "
            f"GB is below the {floor_bytes / 1024**3:.2f} GB floor naively scaled from the ~3.5 "
            "GB/500-atom reference -- proceeding anyway (this linear floor is known to be "
            "miscalibrated at larger n_atoms; see this function's docstring). If this batch width "
            "later OOMs mid-production, that's the real signal to distrust -- rerun with a smaller "
            "--batch-width override rather than relying on this check."
        )


def select_batch_width(
    model: UMAWrapper,
    template: Atoms,
    reference_runs: tuple[RunSpec, ...],
    n_atoms: int,
    device: torch.device,
    md_steps_per_block: int = MD_STEPS_PER_BLOCK,
    mc_step_fraction: float = MC_STEP_FRACTION,
    mc_energy_only: bool = MC_ENERGY_ONLY,
) -> int:
    """Profile candidate widths, verify the memory floor, and recommend one."""
    planner = SimulationBatchPlanner(
        memory_fraction=BATCH_MEMORY_FRACTION,
        throughput_fraction=BATCH_THROUGHPUT_FRACTION,
    )
    measurements = planner.profile(
        lambda width, dev: profile_workload(
            model,
            template,
            reference_runs,
            width,
            dev,
            md_steps_per_block,
            mc_step_fraction,
            mc_energy_only,
        ),
        BATCH_WIDTH_CANDIDATES[n_atoms],
        device=device,
        warmup_blocks=PROFILE_WARMUP_BLOCKS,
        measured_blocks=PROFILE_MEASURED_BLOCKS,
    )
    _verify_memory_floor(measurements, n_atoms)
    width = planner.recommend_width(
        measurements,
        total_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
    )
    print(f"[batch-width] n_atoms={n_atoms} selected width={width} from {measurements}")
    if n_atoms == 500 and not 7 <= width <= 13:
        print(
            f"[batch-width] WARNING: expected width ~10 for 500 atoms on a 40 GB A100, "
            f"measured {width}; check GPU occupancy and model version."
        )
    return width


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
    mc_step_fraction: float = MC_STEP_FRACTION,
    mc_energy_only: bool = MC_ENERGY_ONLY,
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
                    "timestamp_utc",
                    "n_atoms",
                    "batch_width",
                    "n_blocks",
                    "wall_seconds",
                    "walker_blocks_per_second",
                    "peak_gpu_allocated_gb",
                    "peak_gpu_reserved_gb",
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
                n_blocks = (
                    n_blocks_root
                    if runs[0].parent_id is None
                    else n_blocks_continuation
                )
                parents = tuple(
                    scheduler.parent_state(run, device=device) for run in runs
                )
                hybrid, batch = make_workload(
                    model,
                    template,
                    runs,
                    parents,
                    device,
                    md_steps_per_block,
                    mc_step_fraction,
                    mc_energy_only,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                result, pt_fraction_series, energy_per_atom_series = (
                    run_hybrid_with_observables(
                        hybrid,
                        batch,
                        n_blocks,
                        len(runs),
                        pt_number,
                    )
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                if device.type == "cuda":
                    peak_allocated_gb = (
                        torch.cuda.max_memory_allocated(device) / 1024**3
                    )
                    peak_reserved_gb = torch.cuda.max_memory_reserved(device) / 1024**3
                else:
                    peak_allocated_gb = 0.0
                    peak_reserved_gb = 0.0
                timestamp_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
                acceptance = hybrid.mc.stats.acceptance
                atoms_per_walker = result.num_nodes // len(runs)
                unresolved_run_ids = []
                for graph_index, (run, final_state) in enumerate(
                    zip(runs, result.to_data_list())
                ):
                    scheduler.complete(run.run_id, final_state)
                    composition_series = [
                        block[graph_index] for block in pt_fraction_series
                    ]
                    energy_series = [
                        block[graph_index] for block in energy_per_atom_series
                    ]
                    composition_gate = equilibration_gate(
                        composition_series, EQUILIBRATION_WINDOW_BLOCKS
                    )
                    energy_gate = equilibration_gate(
                        energy_series, EQUILIBRATION_WINDOW_BLOCKS
                    )
                    resolved = (
                        bool(composition_gate.get("resolved"))
                        and bool(energy_gate.get("resolved"))
                        if composition_gate.get("resolved") is not None
                        and energy_gate.get("resolved") is not None
                        else None
                    )
                    if resolved is not True:
                        unresolved_run_ids.append(run.run_id)
                    (
                        scheduler.state_store.root / f"{run.run_id}.equilibration.json"
                    ).write_text(
                        json.dumps(
                            {
                                "run_id": run.run_id,
                                "temperature_K": run.temperature_k,
                                "chemical_potentials_ev": {
                                    chemical_symbols[z]: run.chemical_potentials_ev[z]
                                    for z in SPECIES
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
                            timestamp_utc,
                            atoms_per_walker,
                            batch_width,
                            n_blocks,
                            f"{elapsed:.3f}",
                            f"{len(runs) * n_blocks / elapsed:.4f}",
                            f"{peak_allocated_gb:.3f}",
                            f"{peak_reserved_gb:.3f}",
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
                # Every workload enters a fresh CUDA stream (BaseDynamics.__enter__), so the caching
                # allocator strands the previous run's blocks on a stream nothing uses again: reserved
                # memory climbs by one run's working pool per batch while allocated stays flat. Release
                # them here so reserved tracks one run (larger with activation_checkpointing=false).
                del hybrid, batch, result
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
    print(
        f"Campaign {campaign.name!r} complete: "
        f"{len(scheduler.completed_ids)} final states in {scheduler.state_store.root}"
    )


def _run_trace(
    args, model, template, device, md_steps_per_block, checkpoint_dir
) -> None:
    """--mode trace-boundary: integrate eq. (29) from one coexistence point (resumable)."""
    from boundary_tracer import BoundaryTracer, TraceConfig

    trace_dir = checkpoint_dir / f"trace_{args.trace_label}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    store = FinalStateStore(trace_dir / "states")
    cfg = TraceConfig(
        t0=args.trace_t0,
        mu0=args.trace_mu0,
        mu0_se=args.trace_mu0_se,
        t_stop=args.trace_t_stop,
        dt=args.trace_dt,
        dt_min=args.trace_dt_min,
        dt_max=args.trace_dt_max,
        tol_mu=args.trace_tol_mu,
        min_gap=args.trace_min_gap,
        recenter=not args.trace_no_recenter,
        recenter_dmu=args.trace_recenter_dmu,
        x_a0=args.trace_x_alpha0,
        x_g0=args.trace_x_gamma0,
    )
    engine = NvalchemiTraceEngine(
        model,
        template,
        device,
        store,
        md_steps_per_block,
        args.mc_step_fraction,
        args.trace_n_blocks,
        args.trace_label,
        trace_dir / "runs.jsonl",
        replicas=args.trace_replicas,
        mc_energy_only=args.mc_energy_only,
    )
    print(
        f"[trace] {args.trace_label}: T {cfg.t0:g} -> {cfg.t_stop:g} K from dmu={cfg.mu0:.5f} eV, "
        f"{args.trace_n_blocks} blocks/run, md_steps_per_block={md_steps_per_block}, "
        f"mc_step_fraction={args.mc_step_fraction}, replicas/phase={args.trace_replicas} "
        f"(batch width {2 * args.trace_replicas}), inference_settings={args.inference_settings!r}, "
        f"mc_energy_only={args.mc_energy_only}; trace file {trace_dir / 'trace.json'}",
        flush=True,
    )
    result = BoundaryTracer(cfg, engine, trace_dir / "trace.json").run(
        str(args.trace_alpha_state), str(args.trace_gamma_state)
    )
    print(
        f"[trace] {result['status']}: {result['stop_reason']} ({len(result['points'])} points)",
        flush=True,
    )


def main() -> None:
    """Entry point: run one system size's cooling campaign, or its
    two-branch delta_mu-scan schedule across one or more temperatures
    (--mode delta-mu-scan), end-to-end."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--n-atoms", type=int, required=True, choices=sorted(SIZE_REPEATS)
    )
    parser.add_argument(
        "--checkpoint-root", type=Path, default=Path("hybrid_sgc_npt_checkpoints")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--mc-energy-only",
        action=argparse.BooleanOptionalAction,
        default=MC_ENERGY_ONLY,
        help="Evaluate energies only (no forces/stress autograd) during MC blocks; MD blocks keep "
        "full outputs. Exact: each MC block re-evaluates its baseline under the same outputs. "
        "On by default; --no-mc-energy-only restores full outputs in every phase.",
    )
    parser.add_argument(
        "--inference-settings",
        default="auto",
        help="UMA inference settings: 'auto' (default), a fairchem preset name or a key=value "
        "InferenceSettings spec. auto = INFERENCE_SETTINGS_COMPILED (compile, no merge) for "
        "hybrid runs and INFERENCE_SETTINGS (no compile, no merge) for pure SGC "
        "(--md-steps-per-block 0). Pass INFERENCE_SETTINGS' spec explicitly for short runs "
        "(under ~100 blocks per process), which do not repay the ~2 min compile.",
    )
    parser.add_argument(
        "--md-steps-per-block",
        type=int,
        default=None,
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
        "--mc-step-fraction",
        type=float,
        default=MC_STEP_FRACTION,
        help=f"MC trials per block = round(fraction * n_atoms); default {MC_STEP_FRACTION} "
        "(100 trials at 500 atoms). Raise it when MC acceptance is so low that a block "
        "accepts well under one move, which is what pins a hybrid MD+MC walker at a pure "
        "composition: on a RELAXED lattice at 700 K, Au-Pt acceptance is 0.002-0.03 "
        "(vs 0.20-0.32 for fixed-lattice SGC), i.e. ~0.5 accepted transmutations per "
        "100-trial block. Cost per block scales as (mc_steps + md_steps) / (100 + 50) at "
        "500 atoms, so 0.6 (300 trials) is ~2.3x a 0.2 block. Note this buys sampling, not "
        "an unbiased fix: the acceptance is low because inserting a solute into a pure, "
        "relaxed host costs strain energy, and a single-site transmutation cannot relax "
        "that. A move set that can (Kawasaki swaps of unlike neighbours, or a "
        "relaxation-aware proposal) would raise acceptance at its root.",
    )
    parser.add_argument(
        "--batch-width",
        type=int,
        default=None,
        help="Skip auto-profiling and use this batch width instead.",
    )
    parser.add_argument(
        "--reference-energies-json",
        type=Path,
        default=None,
        help="reference_energy_calibration.py output. Rebuilds every run's "
        "chemical_potentials_ev as {Au: 0.0, Pt: delta_mu_ref_eV(T) + delta_mu_excess} "
        "at each run's own temperature, replacing the literal, uncalibrated DELTA_MU_EV "
        "sweep. Recommended -- see README 'Chemical-potential calibration'. "
        "Omit to keep the old literal behavior (a warning is printed).",
    )
    parser.add_argument(
        "--allow-unresolved-reference",
        action="store_true",
        help="Proceed even if a needed temperature's calibration entry has "
        "equilibration_gate.resolved != true. Off by default.",
    )
    parser.add_argument(
        "--mode",
        choices=["cooling", "delta-mu-scan", "trace-boundary"],
        default="cooling",
        help="'cooling' (default): the existing 3000->1600 K, fixed-delta_mu "
        "campaign. 'delta-mu-scan': a two-branch (A-rich/B-rich) "
        "delta_mu_excess continuation across one or more "
        "--scan-temperatures-k (PHASE_DIAGRAM_MANUAL.md section 4, all of "
        "steps 1-4). Fully automatic: any requested temperature missing "
        "from --reference-energies-json (or its complete absence) is "
        "calibrated in-process first, on the same GPU and the same already-"
        "loaded model, via compute_reference_energies -- see README 'Modes'.",
    )
    parser.add_argument(
        "--scan-temperatures-k",
        type=float,
        nargs="+",
        default=None,
        help="--mode delta-mu-scan only: one or more temperatures, HIGHEST "
        "FIRST. A single value is a from-scratch scan at that temperature; "
        "two or more chain each next (lower) temperature's two endpoints as "
        "continuation children of the previous temperature's own endpoints "
        "(section 4 steps 2 and 4), UNLESS --scan-independent-temperatures "
        "is passed -- run every temperature in this ONE invocation, not as "
        "separate script calls (see README 'Modes'). Required in "
        "delta-mu-scan mode.",
    )
    parser.add_argument(
        "--scan-independent-temperatures",
        action="store_true",
        help="--mode delta-mu-scan only: drop cross-temperature continuation "
        "-- every requested temperature's A-rich/B-rich seed pair gets "
        "parent_id=None instead of chaining to the previous temperature's "
        "seeds, so all 2 * len(--scan-temperatures-k) seeds are ready "
        "simultaneously in generation 1 instead of only the highest "
        "temperature's 2. Trades a warm-started composition at each lower "
        "temperature for full generation-1 batch width. Off by default.",
    )
    parser.add_argument(
        "--delta-mu-excess-bracket-ev",
        type=float,
        default=0.2,
        help="--mode delta-mu-scan only: half-width of the scan -- the seed "
        "endpoints sit at delta_mu_excess = -bracket (A-rich) and +bracket "
        "(B-rich), each marching toward 0.0. Default 0.2 eV.",
    )
    parser.add_argument(
        "--delta-mu-excess-min-step-ev",
        type=float,
        default=0.01,
        help="--mode delta-mu-scan only: finest (last, closest to 0.0) step "
        "size in each branch's NON-UNIFORM ladder -- see "
        "--delta-mu-excess-refine-ratio. Default 0.01 eV.",
    )
    parser.add_argument(
        "--delta-mu-excess-refine-ratio",
        type=float,
        default=0.5,
        help="--mode delta-mu-scan only: each new ladder point sits this "
        "fraction of the way from the previous point to 0.0 (default 0.5, "
        "i.e. halving), so steps shrink geometrically approaching the "
        "transition and stay coarse near the safe bracket endpoint. Must be "
        "in (0, 1).",
    )
    parser.add_argument(
        "--delta-mu-excess-overshoot-ev",
        type=float,
        default=0.0,
        help="--mode delta-mu-scan only: continue each branch PAST delta_mu_excess "
        "= 0.0 by this much (step size doubling away from 0.0), so the A-rich and "
        "B-rich branches share a delta_mu window. Required to see hysteresis and "
        "therefore to locate a first-order boundary: with the default 0.0 the two "
        "branches meet at a single point and neither is ever metastable where the "
        "other has data. Costs roughly log2(overshoot/min_step)+1 extra runs per "
        "branch per temperature.",
    )
    parser.add_argument(
        "--n-blocks-scan-seed",
        type=int,
        default=N_BLOCKS_SCAN_SEED,
        help=f"Blocks for each from-scratch A-rich/B-rich endpoint (default {N_BLOCKS_SCAN_SEED}).",
    )
    parser.add_argument(
        "--n-blocks-scan-step",
        type=int,
        default=N_BLOCKS_SCAN_STEP,
        help=f"Blocks per warm-started delta_mu step (default {N_BLOCKS_SCAN_STEP}). Raise when runs "
        "near the transition fail the equilibration gate.",
    )
    trace = parser.add_argument_group(
        "--mode trace-boundary",
        "Integrate the coexistence line dmu_coex(T) from ONE known coexistence point with "
        "van de Walle & Asta eq. (29) (boundary_tracer.py, vendored from the sgc-phase-boundary "
        "skill). Two walkers -- one per phase -- run at the same (T, dmu) every step.",
    )
    trace.add_argument(
        "--trace-t0",
        type=float,
        help="starting temperature (K), a known coexistence point",
    )
    trace.add_argument(
        "--trace-mu0",
        type=float,
        help="coexistence dmu = mu_Pt - mu_Au at --trace-t0 (eV)",
    )
    trace.add_argument(
        "--trace-mu0-se",
        type=float,
        default=0.0,
        help="uncertainty of --trace-mu0 (eV)",
    )
    trace.add_argument(
        "--trace-t-stop",
        type=float,
        help="trace toward this temperature (K), up or down",
    )
    trace.add_argument(
        "--trace-alpha-state",
        type=Path,
        help="checkpoint .pt of a walker equilibrated in the Au-rich (low-x) phase near --trace-mu0",
    )
    trace.add_argument(
        "--trace-gamma-state",
        type=Path,
        help="checkpoint .pt of a walker equilibrated in the Pt-rich (high-x) phase near --trace-mu0",
    )
    trace.add_argument(
        "--trace-label",
        default="trace",
        help="names the trace file and its checkpoints",
    )
    trace.add_argument("--trace-dt", type=float, default=50.0, help="initial |dT| (K)")
    trace.add_argument("--trace-dt-min", type=float, default=5.0)
    trace.add_argument("--trace-dt-max", type=float, default=100.0)
    trace.add_argument(
        "--trace-n-blocks",
        type=int,
        default=200,
        help="blocks per walker per (T, dmu) run",
    )
    trace.add_argument(
        "--trace-tol-mu", type=float, default=0.002, help="corrector convergence (eV)"
    )
    trace.add_argument(
        "--trace-min-gap",
        type=float,
        default=0.05,
        help="stop when x_gamma - x_alpha drops below",
    )
    trace.add_argument(
        "--trace-x-alpha0",
        type=float,
        default=None,
        help="x of the alpha (low-x) phase at the start; with --trace-x-gamma0 it lets the tracer "
        "recenter a mis-centred start point (Fig. 6) instead of rejecting it",
    )
    trace.add_argument(
        "--trace-x-gamma0",
        type=float,
        default=None,
        help="x of the gamma phase at the start",
    )
    trace.add_argument(
        "--trace-no-recenter",
        action="store_true",
        help="disable the tracer's Fig. 6 recentering (reject + halve dT only, as in tracer 1.1)",
    )
    trace.add_argument(
        "--trace-recenter-dmu",
        type=float,
        default=0.0025,
        help="first dmu step (eV) of a recentering sweep; doubles every 4 runs up to 8x",
    )
    trace.add_argument(
        "--trace-replicas",
        type=int,
        default=1,
        choices=(1, 2),
        help="independent walkers per phase; batch width = 2 x replicas (GPU_MEMORY_GUIDE.md: 2-4 "
        "for the energy-only SGC spec). Keep fixed for one trace.json",
    )
    parser.add_argument(
        "--calibration-n-blocks",
        type=int,
        default=100,
        help="--mode delta-mu-scan only, auto-calibration: MD blocks for each "
        "missing temperature's pure-Au/pure-Pt NPT run. Default 100 (matches "
        "reference_energy_calibration.py's own default).",
    )
    parser.add_argument(
        "--calibration-md-steps-per-block",
        type=int,
        default=None,
        help="--mode delta-mu-scan only, auto-calibration: MD steps/block "
        "for the calibration run. Defaults to MD_STEPS_PER_BLOCK (the "
        "alloy's own value).",
    )
    parser.add_argument(
        "--calibration-equilibration-window-blocks",
        type=int,
        default=25,
        help="--mode delta-mu-scan only, auto-calibration: equilibration-gate "
        "window, PHASE_DIAGRAM_MANUAL.md section 7. Default 25.",
    )
    parser.add_argument(
        "--calibration-velocity-seed",
        type=int,
        default=None,
        help="--mode delta-mu-scan only, auto-calibration: base velocity "
        "seed. Defaults to SEED (the alloy's own base seed).",
    )
    args = parser.parse_args()
    torch._inductor.config.shape_padding = INDUCTOR_SHAPE_PADDING
    md_steps_per_block = (
        args.md_steps_per_block
        if args.md_steps_per_block is not None
        else MD_STEPS_PER_BLOCK
    )
    args.inference_settings = resolve_inference_settings(
        args.inference_settings, md_steps_per_block
    )
    print(
        f"[hybrid] mc_energy_only={args.mc_energy_only} inference_settings={args.inference_settings!r}",
        flush=True,
    )
    if "compile=true" in args.inference_settings and args.batch_width is None:
        print(
            "[hybrid] compiled model + batch-width auto-profiling: each profiled width "
            "compiles its own graphs once (a few minutes in total); pass --batch-width to skip.",
            flush=True,
        )

    device = torch.device(args.device)
    n_atoms = args.n_atoms
    repeats = SIZE_REPEATS[n_atoms]
    checkpoint_dir = args.checkpoint_root / f"atoms{n_atoms}"
    log_path = args.checkpoint_root / f"atoms{n_atoms}_throughput.csv"

    template = build_ase_structure(
        TEMPLATE_SYMBOL,
        CRYSTAL_STRUCTURE,
        LATTICE_A_ANG,
        repeats,
        cubic=CONVENTIONAL_CELL,
    )
    if len(template) != n_atoms:
        raise ValueError(
            f"expected {n_atoms} atoms, repeats={repeats} built {len(template)}"
        )

    model = UMAWrapper.from_checkpoint(
        CHECKPOINT,
        task_name=TASK,
        device=str(device),
        inference_settings=args.inference_settings,
    )
    print(
        f"[hybrid] md_steps_per_block={md_steps_per_block}"
        + (" (pure SGC)" if md_steps_per_block == 0 else "")
    )

    if args.mode == "trace-boundary":
        missing = [
            f
            for f in (
                "trace_t0",
                "trace_mu0",
                "trace_t_stop",
                "trace_alpha_state",
                "trace_gamma_state",
            )
            if getattr(args, f) is None
        ]
        if missing:
            parser.error(
                "--mode trace-boundary requires "
                + ", ".join("--" + m.replace("_", "-") for m in missing)
            )
        _run_trace(args, model, template, device, md_steps_per_block, checkpoint_dir)
        return

    if args.mode == "delta-mu-scan":
        if not args.scan_temperatures_k:
            parser.error("--mode delta-mu-scan requires --scan-temperatures-k")
        symbols = [chemical_symbols[z] for z in SPECIES]
        delta_mu_ref_by_t, missing_temperatures = _load_available_delta_mu_ref(
            args.reference_energies_json,
            args.scan_temperatures_k,
            symbols,
            allow_unresolved=args.allow_unresolved_reference,
        )
        if missing_temperatures:
            print(
                f"[calibration] no reference energy for T={list(missing_temperatures)} K -- "
                f"auto-calibrating pure {symbols[0]}/{symbols[1]} there now (same process, "
                "same GPU, same model already loaded above)"
            )
            calibration = compute_reference_energies(
                model,
                missing_temperatures,
                n_atoms,
                n_blocks=args.calibration_n_blocks,
                md_steps_per_block=args.calibration_md_steps_per_block
                or MD_STEPS_PER_BLOCK,
                equilibration_window_blocks=args.calibration_equilibration_window_blocks,
                velocity_seed=args.calibration_velocity_seed or SEED,
                device=device,
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            calibration_path = checkpoint_dir / "auto_reference_energies.json"
            calibration_path.write_text(json.dumps(calibration, indent=2) + "\n")
            for t in missing_temperatures:
                entry = calibration["reference"][f"{t:g}"]
                gate_status = {
                    symbol: entry[symbol]["equilibration_gate"]["resolved"]
                    for symbol in symbols
                }
                if not args.allow_unresolved_reference and not all(
                    v is True for v in gate_status.values()
                ):
                    raise SystemExit(
                        f"[calibration] auto-calibration at T={t:g} K did not pass the "
                        f"equilibration gate ({gate_status}) -- rerun with a larger "
                        "--calibration-n-blocks, or pass --allow-unresolved-reference to "
                        "proceed anyway (not recommended)."
                    )
                delta_mu_ref_by_t[t] = entry["delta_mu_ref_eV"]
                print(
                    f"  T={t:g} K: delta_mu_ref = {entry['delta_mu_ref_eV']:.6f} eV (auto-calibrated)"
                )
            print(f"[calibration] wrote {calibration_path}")
        campaign, endpoints = _build_delta_mu_scan_schedule(
            n_atoms,
            args.scan_temperatures_k,
            delta_mu_ref_by_t,
            args.delta_mu_excess_bracket_ev,
            args.delta_mu_excess_min_step_ev,
            args.delta_mu_excess_refine_ratio,
            args.delta_mu_excess_overshoot_ev,
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
                args.reference_energies_json,
                TEMPERATURES_K,
                symbols,
                allow_unresolved=args.allow_unresolved_reference,
            )
            campaign = _apply_calibrated_delta_mu(campaign, delta_mu_ref_by_t)
            print(f"[delta_mu] calibrated from {args.reference_energies_json}")
        else:
            print(
                "[delta_mu] WARNING: no --reference-energies-json given -- using the "
                "literal, uncalibrated DELTA_MU_EV sweep (chemical_potentials_ev="
                "{Au: 0.0, Pt: delta_mu}). This does not locate the real Au-Pt phase "
                "boundary; see README.md 'Chemical-potential calibration'."
            )
    scheduler = CampaignScheduler(campaign, FinalStateStore(checkpoint_dir))
    print(
        f"Campaign {campaign.name!r}: {len(campaign.runs)} runs, "
        f"{len(scheduler.completed_ids)} already complete."
    )

    if args.batch_width is not None:
        batch_width = args.batch_width
    elif device.type == "cuda":
        batch_width = select_batch_width(
            model,
            template,
            reference_runs,
            n_atoms,
            device,
            md_steps_per_block,
            args.mc_step_fraction,
            args.mc_energy_only,
        )
    else:
        batch_width = 1
        print("CUDA unavailable; using batch_width=1")

    n_blocks_root = (
        args.n_blocks_scan_seed if args.mode == "delta-mu-scan" else N_BLOCKS_REFERENCE
    )
    n_blocks_continuation = (
        args.n_blocks_scan_step
        if args.mode == "delta-mu-scan"
        else N_BLOCKS_CONTINUATION
    )
    _run_campaign(
        model,
        template,
        scheduler,
        campaign,
        batch_width,
        device,
        log_path,
        n_blocks_root,
        n_blocks_continuation,
        md_steps_per_block,
        args.mc_step_fraction,
        args.mc_energy_only,
    )


if __name__ == "__main__":
    main()
