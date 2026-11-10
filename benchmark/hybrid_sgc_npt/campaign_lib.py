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
"""Shared building blocks of the Au-Pt hybrid SGC-NPT drivers in this directory.

System and protocol constants (UMA checkpoint and inference settings, the fcc
Au-Pt cell, NPT and MC block parameters), walker construction
(``build_ase_structure``, ``make_workload``), between-block geometry helpers
(``relax_cell_volume``, ``wrap_batch_positions``), per-block observables and the
end-of-run equilibration gate (``run_hybrid_with_observables``,
``equilibration_gate``), in-process pure-element reference calibration
(``compute_reference_energies``) and the ``boundary_tracer.py`` engine
(``NvalchemiTraceEngine``). ``run_campaign.py`` is the command-line driver;
``run_mu_ladder.py``, ``run_vcsgc_scan.py``, ``run_phase_campaign.py`` and the
``../uma_efficiency`` benchmarks import from here. ``NVALCHEMI_UMA_CHECKPOINT``
overrides the UMA checkpoint for all of them.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Callable, Sequence

import torch
from ase import Atoms
from ase.build import bulk
from ase.data import chemical_symbols

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import DynamicsStage
from nvalchemi.dynamics.integrators.npt import NPT
from nvalchemi.hooks import WrapPeriodicHook
from nvalchemi.hooks.periodic import wrap_positions_into_cell
from nvalchemi.hybrid import HybridMCMD
from nvalchemi.mc import SGC
from nvalchemi.models.uma import UMAWrapper
from nvalchemi.scheduling import RunSpec

# Overridable per cluster (Perlmutter's env.sh sets uma-s-1p2p1); every driver importing CHECKPOINT follows it.
# Never mix checkpoints within one isotherm / boundary trace.
CHECKPOINT = os.environ.get("NVALCHEMI_UMA_CHECKPOINT", "uma-s-1p2")
TASK = "omat"
# SGC changes atomic composition every step, so no MoLE merge (it assumes a
# fixed composition). Activation checkpointing off and tf32 on: 1.9x faster per
# step than the old "batch" preset, same sampled chain -- see "UMA settings for
# SGC and SGC-NPT" in docs/userguide/dynamics_simulations.md. Key=value specs,
# which UMAWrapper.from_checkpoint parses into InferenceSettings.
#
# INFERENCE_SETTINGS is the uncompiled spec (also imported by other scripts here).
# --inference-settings auto (the default) uses INFERENCE_SETTINGS_COMPILED for
# hybrid runs: with UMAWrapper's dynamic compile shapes, SGC-NPT runs 1.06x faster
# with ~12% less memory after a ~2 min one-time compile per process (500 atoms,
# A100-SXM), which campaign processes of hundreds of blocks repay (break-even ~125
# blocks at width 1, ~45 at width 4). Pure SGC (--md-steps-per-block 0) stays
# uncompiled: compile measured slower there.
INFERENCE_SETTINGS = (
    "compile=false,merge_mole=false,tf32=true,activation_checkpointing=false"
)
INFERENCE_SETTINGS_COMPILED = (
    "compile=true,merge_mole=false,tf32=true,activation_checkpointing=false"
)


def resolve_inference_settings(spec: str, md_steps_per_block: int) -> str:
    """Resolve ``--inference-settings auto``: compiled for hybrids, eager for pure SGC.

    Parameters
    ----------
    spec : str
        ``"auto"``, a fairchem preset name or a key=value spec.
    md_steps_per_block : int
        MD steps per block; 0 is pure SGC MC.

    Returns
    -------
    str
        The spec passed to ``UMAWrapper.from_checkpoint``.
    """
    if spec != "auto":
        return spec
    return INFERENCE_SETTINGS_COMPILED if md_steps_per_block > 0 else INFERENCE_SETTINGS


# Evaluate energies only (no forces/stress autograd) during MC blocks; MD blocks
# keep full outputs. Measured 2.1x per MC step for Kawasaki (turbo) and ~1.4x for
# SGC on A100, same sampled chain. On by default; --no-mc-energy-only in main()
# turns it off.
MC_ENERGY_ONLY = True
TEMPLATE_SYMBOL = "Au"
CRYSTAL_STRUCTURE = "fcc"
LATTICE_A_ANG = 4.00
CONVENTIONAL_CELL = True
PT_FRACTION = 0.50
PRESSURE_EV_PER_A3 = 1.01325 / 1.602176634e6  # 1 atmosphere
SPECIES = (79, 78)  # Au, Pt
SIZE_REPEATS: dict[int, tuple[int, int, int]] = {
    # 108 and 256: small cells for batching benchmarks (the 108-atom cell is
    # 12 A across, twice UMA's 6 A cutoff; the graph builder handles images).
    108: (3, 3, 3),
    256: (4, 4, 4),
    500: (5, 5, 5),
    # 1000: the 500-atom cell doubled along z (the only fcc supercell with exactly 1000 atoms);
    # 20 A x 20 A x 40 A, so a slab of the other phase across the short axes costs the same
    # interface area as at 500 atoms while the volume doubles.
    1000: (5, 5, 10),
    1372: (7, 7, 7),
    2048: (8, 8, 8),
    4000: (10, 10, 10),
}
MC_STEP_FRACTION = 0.2  # MC trials per block = round(MC_STEP_FRACTION * n_atoms)
MD_STEPS_PER_BLOCK = 50
DT_FS = 3.0
THERMOSTAT_TIME_FS = 100.0
BAROSTAT_TIME_FS = 1000.0
SEED = 20260827
EQUILIBRATION_WINDOW_BLOCKS = 25  # equilibration_gate check interval.
KB_EV = 8.617333262e-5  # Boltzmann constant, eV/K.


def npt_wrap_hooks() -> list[WrapPeriodicHook]:
    """Keep positions inside the cell during NPT.

    NPT's position update (``r += (v + eps_dot . r) dt``) never wraps, and the
    UMA adapter hands raw positions to fairchem, whose periodic graph builder
    only scans image offsets of +-ceil(cutoff / L) (+-1 here). Once an atom has
    diffused more than about one cell length from a neighbour in *unwrapped*
    coordinates (liquid-like Au-rich cells at 1200-1400 K do this within one or
    two 15 ps ladder steps), that pair's minimum image is never generated. The
    two atoms feel no repulsion and can fall onto each other: every
    collapsed (< 1.5 A) pair in the scan_1200_1400_hybrid checkpoints needed an
    image offset > +-1. Wrapping every step keeps every pair within +-1 image.
    """
    return [WrapPeriodicHook(frequency=1, stage=DynamicsStage.AFTER_POST_UPDATE)]


def wrap_batch_positions(batch: Batch) -> None:
    """Wrap a (restored) batch into its cells once, before its first force call.

    The hook above only fires after an MD step, but ``md.compute`` and
    ``mc.synchronize`` (and every MC trial of the first block) run first, so a
    warm-started parent saved with unwrapped positions must be folded back here.
    """
    cell = batch.cell
    pbc = getattr(batch, "pbc", None)
    if pbc is None:
        pbc = torch.ones((batch.num_graphs, 3), dtype=torch.bool, device=batch.device)
    # Same shape normalisation as WrapPeriodicHook._wrap_positions.
    if cell.dim() == 4:
        cell = cell.squeeze(1)
    if pbc.dim() == 3:
        pbc = pbc.squeeze(1)
    with torch.no_grad():
        wrap_positions_into_cell(
            batch.positions, cell, pbc.to(torch.bool), batch.batch_idx
        )


def relax_cell_volume(
    md: NPT,
    batch: Batch,
    *,
    bulk_modulus_ev_per_a3: float = 1.2,
    tolerance_ev_per_a3: float = 2e-4,
    max_iterations: int = 4,
    target_pressure_ev_per_a3: float = 0.0,
) -> torch.Tensor:
    """Relax each graph's volume at fixed fractional coordinates (rigid-lattice MC).

    Scales cell and positions together, isotropically, so every atom keeps its lattice site and
    only the lattice constant follows the composition: one Newton step on V per iteration,
    ``V <- V (1 + (P - P_target) / B)`` with P = -tr(stress)/3 from the last ``md.compute`` (stress
    is tensile-positive, see NPT), re-evaluating after each step until every graph has
    ``|P - P_target| < tolerance``. A rough B only slows convergence (error shrinks by
    ``|1 - B_true/B|`` per step); B ~ 1.1-1.4 eV/A^3 for Au-Pt. Static lattice: no kinetic term.
    Expects ``batch.stress`` from a compute at the current geometry; leaves energy/forces/stress
    evaluated at the relaxed geometry. Returns the final per-graph pressure (eV/A^3).
    """
    n_graphs = batch.num_graphs
    for iteration in range(max_iterations + 1):
        pressure = (
            -batch.stress.reshape(n_graphs, 3, 3).diagonal(dim1=-2, dim2=-1).sum(-1)
            / 3.0
        )
        excess = pressure - target_pressure_ev_per_a3
        if iteration == max_iterations or bool(
            (excess.abs() < tolerance_ev_per_a3).all()
        ):
            return pressure.detach()
        scale = (1.0 + excess / bulk_modulus_ev_per_a3).clamp(0.85, 1.15) ** (1.0 / 3.0)
        with torch.no_grad():
            batch.cell.mul_(scale.view(-1, *([1] * (batch.cell.dim() - 1))))
            batch.positions.mul_(scale[batch.batch_idx].unsqueeze(-1))
        md.compute(batch)
    return pressure.detach()


def lattice_constant_per_graph(
    batch: Batch, atoms_per_cubic_cell: int = 4
) -> list[float]:
    """Cubic-equivalent lattice constant (4 atoms per fcc cube) of each graph's cell."""
    n_graphs = batch.num_graphs
    volumes = torch.linalg.det(batch.cell.reshape(n_graphs, 3, 3)).abs()
    counts = torch.bincount(batch.batch_idx, minlength=n_graphs).to(volumes.dtype)
    return (atoms_per_cubic_cell * volumes / counts).pow(1.0 / 3.0).tolist()


def build_ase_structure(
    symbol: str,
    crystal_structure: str,
    lattice_a: float,
    repeats: tuple[int, int, int],
    *,
    cubic: bool,
) -> Atoms:
    """Build a periodic ASE crystal template for one independent run."""
    unit_cell = bulk(
        symbol, crystalstructure=crystal_structure, a=lattice_a, cubic=cubic
    )
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
    numbers[torch.randperm(n_atoms, device=device, generator=generator)[:pt_count]] = (
        SPECIES[1]
    )
    data.atomic_numbers = numbers
    data.atomic_masses = None
    data.use_default_masses()
    velocity_std = torch.sqrt(
        torch.as_tensor(KB_EV * run.temperature_k, device=device) / data.atomic_masses
    )
    data.velocities = (
        torch.randn((n_atoms, 3), device=device, generator=generator)
        * velocity_std[:, None]
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
    # mc_accepted (per-graph MC bookkeeping) exists only on states saved after an MC run; drop it
    # so saved and freshly built states can share a batch. The sampler re-creates it on first use.
    return Batch.from_data_list(
        [
            _walker(template, run, SEED + index, parent_state, device)
            for index, (run, parent_state) in enumerate(zip(runs, parent_states))
        ],
        exclude_keys=["mc_accepted"],
    )


def make_workload(
    model: UMAWrapper,
    template: Atoms,
    runs: tuple[RunSpec, ...],
    parent_states: tuple[AtomicData | None, ...],
    device: torch.device,
    md_steps_per_block: int = MD_STEPS_PER_BLOCK,
    mc_step_fraction: float = MC_STEP_FRACTION,
    mc_energy_only: bool = MC_ENERGY_ONLY,
) -> tuple[HybridMCMD, Batch]:
    """Return one per-graph hybrid MC-MD workload for compatible runs.

    The same model object is deliberately passed to both stages: ``HybridMCMD``
    requires every trial energy and MD force to come from one potential.
    ``md_steps_per_block=0`` degenerates to pure SGC (MC moves only, no MD
    sub-step) -- see ``--md-steps-per-block``. ``mc_energy_only`` skips the
    forces/stress backward during MC blocks.
    """
    if not runs:
        raise ValueError("a workload requires at least one run")
    species = runs[0].species
    if any(run.species != species for run in runs):
        raise ValueError("all runs in a workload must share reservoir species")
    temperatures = torch.tensor([run.temperature_k for run in runs], device=device)
    pressures = torch.tensor([run.pressure_ev_per_a3 for run in runs], device=device)
    chemical_potentials = {
        number: torch.tensor(
            [run.chemical_potentials_ev[number] for run in runs], device=device
        )
        for number in species
    }
    mc_steps = max(1, round(mc_step_fraction * len(template)))
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
        hooks=npt_wrap_hooks(),
    )
    hybrid = HybridMCMD(
        mc=sgc,
        md=npt,
        mc_steps=mc_steps,
        md_steps=md_steps_per_block,
        mc_energy_only=mc_energy_only,
    )
    batch = _make_batch(template, runs, parent_states, device)
    wrap_batch_positions(batch)
    return hybrid, batch


def _build_pure_element_template(symbol: str, n_atoms: int) -> Atoms:
    """A pure-element supercell at ASE's own reference lattice constant,
    matching the alloy's crystal structure/repeat count. Deliberately omits
    ``a=`` (unlike ``build_ase_structure``, which always uses the shared,
    Au-based ``LATTICE_A_ANG``): each end-member should start near its OWN
    equilibrium volume -- reusing the shared lattice constant would just
    reintroduce the volume-relaxation skew this calibration removes. NPT's
    barostat does the actual equilibration; this only sets a physically
    reasonable starting point. Lets ``--mode delta-mu-scan`` auto-calibrate
    in-process with the model already loaded for the alloy runs; see
    ``compute_reference_energies`` below for the physical rationale.
    """
    repeats = SIZE_REPEATS[n_atoms]
    unit_cell = bulk(
        symbol, crystalstructure=CRYSTAL_STRUCTURE, cubic=CONVENTIONAL_CELL
    )
    template = unit_cell * repeats
    if len(template) != n_atoms:
        raise ValueError(
            f"expected {n_atoms} atoms for pure {symbol}, repeats={repeats} built {len(template)}"
        )
    return template


def pure_element_state(
    template: Atoms,
    temperature_k: float,
    velocity_seed: int,
    device: torch.device,
) -> AtomicData:
    """One pure-element ``AtomicData`` with a Maxwell-Boltzmann velocity draw
    at ``temperature_k``."""
    data = AtomicData.from_atoms(template, device=device)
    n = data.num_nodes
    data.atomic_masses = None
    data.use_default_masses()
    generator = torch.Generator(device=device).manual_seed(velocity_seed)
    velocity_std = torch.sqrt(
        torch.as_tensor(KB_EV * temperature_k, device=device) / data.atomic_masses
    )
    data.velocities = (
        torch.randn((n, 3), device=device, generator=generator) * velocity_std[:, None]
    )
    data.velocities -= data.velocities.mean(dim=0, keepdim=True)
    data.forces = torch.zeros_like(data.positions)
    data.energy = torch.zeros(1, 1, device=device)
    data.stress = torch.zeros(1, 3, 3, device=device)
    return data


def cell_volumes(batch: Batch) -> torch.Tensor:
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
    ALREADY-LOADED model -- the reference
    ``delta_mu_ref(T) = g_Pt(T) - g_Au(T)``, computed at every requested
    temperature. This is what lets ``--mode delta-mu-scan`` run completely
    automatically on a single GPU: no second checkpoint load, no separate
    script invocation, no manually-prepared ``reference_energies.json``.

    Kept in this script so the campaign entry point is self-sufficient. Each
    pure element is equilibrated independently, from its own ASE reference
    lattice constant, through a real NPT trajectory at the target temperature
    and the alloy's own pressure/thermostat/barostat settings, one graph per
    (element, temperature) in a single batch.

    Returns a ``reference_energies.json``-shaped dict
    (``reference[T][symbol]`` plus, for the 2-species case,
    ``delta_mu_ref_eV`` per temperature) -- the caller decides whether/where
    to persist it and whether its ``equilibration_gate`` is trustworthy
    enough to use.
    """
    temperatures = tuple(float(t) for t in temperatures_k)
    if not temperatures:
        raise ValueError("at least one temperature is required")
    symbols = [chemical_symbols[z] for z in SPECIES]
    templates = {
        number: _build_pure_element_template(symbol, n_atoms)
        for number, symbol in zip(SPECIES, symbols)
    }

    graph_index: list[tuple[int, float]] = [
        (number, t) for number in SPECIES for t in temperatures
    ]
    data_list = [
        pure_element_state(templates[number], t, velocity_seed + i, device)
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
        hooks=npt_wrap_hooks(),
    )

    energy_per_atom_series: list[list[float]] = [[] for _ in range(n_graphs)]
    volume_per_atom_series: list[list[float]] = [[] for _ in range(n_graphs)]

    def _record() -> None:
        energies = batch.energy.detach().reshape(-1)
        counts = batch.num_nodes_per_graph.to(energies.dtype)
        volumes = cell_volumes(batch)
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
        energy_gate = equilibration_gate(energy_per_atom_series[i][1:], window)
        volume_gate = equilibration_gate(volume_per_atom_series[i][1:], window)
        resolved = (
            bool(energy_gate.get("resolved")) and bool(volume_gate.get("resolved"))
            if energy_gate.get("resolved") is not None
            and volume_gate.get("resolved") is not None
            else None
        )
        tail = min(window, len(energy_per_atom_series[i]))
        mean_energy = statistics.fmean(energy_per_atom_series[i][-tail:])
        mean_volume = statistics.fmean(volume_per_atom_series[i][-tail:])
        se_energy = (
            statistics.pstdev(energy_per_atom_series[i][-tail:]) / (tail**0.5)
            if tail > 1
            else float("nan")
        )
        lattice_a = (mean_volume * n_atoms) ** (1 / 3) / n_rep
        reference.setdefault(f"{t:g}", {})[symbol] = {
            "atomic_number": number,
            "energy_eV_per_atom": mean_energy,
            "energy_eV_per_atom_standard_error": se_energy,
            "volume_A3_per_atom": mean_volume,
            "lattice_constant_a_ang": lattice_a,
            "averaging_window_blocks": tail,
            "equilibration_gate": {
                "energy": energy_gate,
                "volume": volume_gate,
                "resolved": resolved,
            },
        }

    s0, s1 = symbols
    for t_entry in reference.values():
        t_entry["delta_mu_ref_eV"] = (
            t_entry[s1]["energy_eV_per_atom"] - t_entry[s0]["energy_eV_per_atom"]
        )
        t_entry["delta_mu_ref_definition"] = (
            f"mu({s1}) - mu({s0}) = g_{s1}(T) - g_{s0}(T)"
        )

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


def pt_fraction_per_graph(batch: Batch, pt_number: int, n_graphs: int) -> list[float]:
    """Per-graph Pt atomic fraction, one GPU->CPU sync."""
    dtype = batch.positions.dtype
    pt_mask = (batch.atomic_numbers == pt_number).to(dtype)
    n_pt = torch.bincount(batch.batch_idx, weights=pt_mask, minlength=n_graphs)
    n_total = torch.bincount(batch.batch_idx, minlength=n_graphs).to(dtype)
    return (n_pt / n_total).detach().cpu().tolist()


def equilibration_gate(series: list[float], window: int) -> dict:
    """One-shot equilibration gate: compare the last two `window`-block
    windows' means against twice their combined standard error. A single
    check run once at the end of a fixed-length run, not a full
    three-consecutive-checks promotion protocol.
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
        "resolved": bool(abs(difference) < 2 * combined_se)
        if combined_se > 0
        else None,
        "window_blocks": window,
        "mean_penultimate_window": mean_a,
        "mean_last_window": mean_b,
        "difference": difference,
        "combined_standard_error": combined_se,
    }


def run_hybrid_with_observables(
    hybrid: HybridMCMD,
    batch: Batch,
    n_blocks: int,
    n_graphs: int,
    pt_number: int,
    cell_hook: Callable[[Batch], None] | None = None,
) -> tuple[Batch, list[list[float]], list[list[float]]]:
    """Run n_blocks hybrid MC-MD blocks, recording each block's per-graph Pt
    fraction and energy/atom -- the observable series equilibration_gate
    needs -- through ``HybridMCMD.run``'s ``on_block`` callback (one small
    GPU->CPU transfer per block).

    ``cell_hook(batch)``, if given, becomes the scheduler's ``after_md_block``:
    it runs after the initial compute and after every block's MD sub-step,
    before MC adopts the energy -- e.g. ``relax_cell_volume`` for rigid-lattice
    SGC (``md_steps=0``) whose lattice constant follows the composition. It must
    leave ``batch.energy`` evaluated at the geometry it leaves behind.
    """
    hybrid.after_md_block = cell_hook
    atoms_per_graph = batch.num_nodes // n_graphs
    pt_fraction_series: list[list[float]] = []
    energy_per_atom_series: list[list[float]] = []
    device = batch.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def record(batch: Batch, block_index: int) -> None:
        """Append this block's per-graph Pt fraction and energy/atom; print memory on CUDA."""
        pt_fraction_series.append(pt_fraction_per_graph(batch, pt_number, n_graphs))
        energies = batch.energy.flatten().detach().cpu().tolist()
        energy_per_atom_series.append([e / atoms_per_graph for e in energies])
        if device.type == "cuda":
            # Printed every block so a crash mid-run still leaves a memory trace in
            # the .out file. allocated/reserved show the residual baseline (a leak
            # climbs block over block); peak_reserved shows this block's transient
            # high-water mark (e.g. an anomalously large neighbor list).
            print(
                f"[memory] block {block_index + 1}/{n_blocks}: "
                f"allocated={torch.cuda.memory_allocated(device) / 1024**3:.3f} GB "
                f"reserved={torch.cuda.memory_reserved(device) / 1024**3:.3f} GB "
                f"peak_reserved_this_block={torch.cuda.max_memory_reserved(device) / 1024**3:.3f} GB",
                flush=True,
            )
            torch.cuda.reset_peak_memory_stats(device)

    hybrid.run(batch, n_blocks=n_blocks, on_block=record)
    return batch, pt_fraction_series, energy_per_atom_series


def batch_means_se(series: list[float], n_batches: int = 5) -> float:
    """SE of the mean of ``series`` from n_batches contiguous batch means (tolerates autocorrelation
    that the gate's per-block standard error ignores)."""
    n = len(series) // n_batches
    if n < 2:
        return float("nan")
    means = [statistics.fmean(series[i * n : (i + 1) * n]) for i in range(n_batches)]
    return statistics.stdev(means) / n_batches**0.5


class NvalchemiTraceEngine:
    """boundary_tracer engine: runs ``replicas`` independent walkers per phase -- all at the same
    (T, dmu) in ONE batch of width 2 * replicas -- saves their final states, and reports per phase
    the replica-mean x and E (per atom), their SEs, the composition drift between the last two
    gate windows, and the gate verdict. Replicas of one phase start from the same state and
    decorrelate through independent MC draws; their mean halves the variance fed to eq. (29) at
    replicas=2, and a replica that disagrees with its partner beyond noise (``replica_split``)
    makes the tracer reject the step. A phase's state is the list of its replicas' checkpoint
    run_ids in ``store`` (or, for the starting walkers, one .pt path used by every replica)."""

    def __init__(
        self,
        model,
        template,
        device,
        store,
        md_steps_per_block,
        mc_step_fraction,
        n_blocks,
        label,
        log_path,
        replicas=1,
        z=2.576,
        min_jump=0.03,
        mc_energy_only=MC_ENERGY_ONLY,
    ):
        self.model, self.template, self.device, self.store = (
            model,
            template,
            device,
            store,
        )
        self.md_steps, self.mc_fraction, self.n_blocks, self.label = (
            md_steps_per_block,
            mc_step_fraction,
            n_blocks,
            label,
        )
        self.log_path, self.replicas, self.z, self.min_jump = (
            log_path,
            replicas,
            z,
            min_jump,
        )
        self.mc_energy_only = mc_energy_only

    def _load(self, state: str) -> AtomicData:
        path = Path(state)
        if path.suffix == ".pt" and path.is_file():
            payload = torch.load(path, map_location=self.device, weights_only=True)
            return AtomicData.model_validate(
                payload["state"] if "state" in payload else payload
            ).to(self.device)
        return self.store.load(state, device=self.device)

    def _states(self, state) -> list[str]:
        states = [state] if isinstance(state, str) else list(state)
        if len(states) == 1:
            states = states * self.replicas
        if len(states) != self.replicas:
            raise ValueError(
                f"expected {self.replicas} replica states, got {len(states)} (was --trace-replicas changed "
                "mid-trace? keep it fixed for one trace.json)"
            )
        return states

    def _observe(self, xs: list[float], es: list[float]) -> dict:
        gx = equilibration_gate(xs, EQUILIBRATION_WINDOW_BLOCKS)
        ge = equilibration_gate(es, EQUILIBRATION_WINDOW_BLOCKS)
        tail_x, tail_e = (
            xs[-2 * EQUILIBRATION_WINDOW_BLOCKS :],
            es[-2 * EQUILIBRATION_WINDOW_BLOCKS :],
        )
        return dict(
            x=gx.get(
                "mean_last_window", statistics.fmean(xs[-EQUILIBRATION_WINDOW_BLOCKS:])
            ),
            x_se=max(
                batch_means_se(tail_x),
                (gx.get("combined_standard_error") or 0) / 2**0.5,
            ),
            E=ge.get(
                "mean_last_window", statistics.fmean(es[-EQUILIBRATION_WINDOW_BLOCKS:])
            ),
            E_se=max(
                batch_means_se(tail_e),
                (ge.get("combined_standard_error") or 0) / 2**0.5,
            ),
            drift=gx.get("difference", 0.0),
            resolved=bool(gx.get("resolved")) and bool(ge.get("resolved")),
        )

    def _combine(self, reps: list[dict]) -> dict:
        n = len(reps)
        xs = [r["x"] for r in reps]
        within = math.sqrt(sum(r["x_se"] ** 2 for r in reps)) / n
        between = statistics.stdev(xs) / math.sqrt(n) if n > 1 else 0.0
        e_within = math.sqrt(sum(r["E_se"] ** 2 for r in reps)) / n
        e_between = (
            statistics.stdev([r["E"] for r in reps]) / math.sqrt(n) if n > 1 else 0.0
        )
        spread = max(xs) - min(xs)
        split_tol = max(
            self.z * math.sqrt(2) * max(r["x_se"] for r in reps), self.min_jump
        )
        return dict(
            x=statistics.fmean(xs),
            x_se=max(within, between),
            E=statistics.fmean(r["E"] for r in reps),
            E_se=max(e_within, e_between),
            drift=statistics.fmean(r["drift"] for r in reps),
            resolved=all(r["resolved"] for r in reps),
            x_replicas=[round(v, 5) for v in xs],
            replica_split=bool(n > 1 and spread > split_tol),
        )

    def run(self, T, mu, state_a, state_g, tag=""):
        """Equilibrate both phases at ``(T, mu)`` and return their observables and final states."""
        tag = f"{self.label}.{tag}".replace("+", "")
        obs, new_states = self._execute(
            T, mu, tag, {"alpha": state_a, "gamma": state_g}
        )
        with self.log_path.open("a") as fh:
            fh.write(
                json.dumps(
                    dict(tag=tag, T=T, mu=mu, alpha=obs["alpha"], gamma=obs["gamma"])
                )
                + "\n"
            )
        a, g = obs["alpha"], obs["gamma"]
        print(
            f"[trace-run] {tag}: T={T:g} mu={mu:.5f} x_a={a['x']:.4f}{a['x_replicas']} x_g={g['x']:.4f}{g['x_replicas']} "
            f"E_a={a['E']:.4f} E_g={g['E']:.4f} acc={a['acceptance']:.3f} width={a['width']} {a['wall_seconds']:.0f}s",
            flush=True,
        )
        return a, g, new_states["alpha"], new_states["gamma"]

    def run_one(self, T, mu, state, phase, tag=""):
        """One phase's replicas only, for the tracer's Fig. 6 recentering sweeps (half the batch)."""
        tag = f"{self.label}.{tag}".replace("+", "")
        name = "alpha" if phase == "a" else "gamma"
        obs, new_states = self._execute(T, mu, tag, {name: state})
        o = obs[name]
        with self.log_path.open("a") as fh:
            fh.write(
                json.dumps(dict(tag=tag, T=T, mu=mu, recentering=name, **{name: o}))
                + "\n"
            )
        print(
            f"[trace-run] {tag}: T={T:g} mu={mu:.5f} {name} x={o['x']:.4f}{o['x_replicas']} (recentering) "
            f"acc={o['acceptance']:.3f} width={o['width']} {o['wall_seconds']:.0f}s",
            flush=True,
        )
        return o, new_states[name]

    def _execute(self, T, mu, tag, phase_states):
        """Run ``replicas`` walkers per phase in ``phase_states`` at ``(T, mu)`` in one batch."""
        phases = tuple(phase_states)
        runs = tuple(
            RunSpec(
                run_id=f"{tag}.{phase}.r{k}",
                temperature_k=float(T),
                pressure_ev_per_a3=PRESSURE_EV_PER_A3,
                chemical_potentials_ev={SPECIES[0]: 0.0, SPECIES[1]: float(mu)},
                species=SPECIES,
                batch_group="trace",
                metadata={"phase": phase, "replica": k},
            )
            for phase in phases
            for k in range(self.replicas)
        )
        parents = tuple(
            self._load(st) for p in phases for st in self._states(phase_states[p])
        )
        width = len(runs)
        hybrid, batch = make_workload(
            self.model,
            self.template,
            runs,
            parents,
            self.device,
            self.md_steps,
            self.mc_fraction,
            self.mc_energy_only,
        )
        start = time.perf_counter()
        result, x_series, e_series = run_hybrid_with_observables(
            hybrid, batch, self.n_blocks, width, SPECIES[1]
        )
        elapsed = time.perf_counter() - start
        acceptance = hybrid.mc.stats.acceptance
        per_run = []
        for i, (run, final_state) in enumerate(zip(runs, result.to_data_list())):
            self.store.save(run.run_id, final_state)
            per_run.append(
                self._observe([b[i] for b in x_series], [b[i] for b in e_series])
            )
        obs, new_states = {}, {}
        for j, phase in enumerate(phases):
            sl = slice(j * self.replicas, (j + 1) * self.replicas)
            obs[phase] = self._combine(per_run[sl]) | dict(
                acceptance=acceptance, wall_seconds=elapsed, width=width
            )
            new_states[phase] = [r.run_id for r in runs[sl]]
        # Each workload enters a fresh CUDA stream (BaseDynamics.__enter__), stranding the previous
        # stream's cached blocks; release them so reserved memory stays at one run's footprint.
        del hybrid, batch, result
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        return obs, new_states
