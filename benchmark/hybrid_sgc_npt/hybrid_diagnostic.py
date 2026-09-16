"""Standalone diagnostic: the REAL interleaved hybrid MC-MD workflow, run
block by block with per-step trajectory dumps, per-step wall time, and
true per-phase peak-memory accounting.

Unlike debug_npt_then_sgc.py (which runs NPT and SGC as two independent,
non-interleaved phases), this drives the actual production alternation:
n_mc_steps SGC trials, then a force refresh, then n_md_steps NPT steps,
then mc.synchronize(batch) -- repeated for n_blocks. This is exactly
HybridMCMD.run()'s own sequence (nvalchemi/hybrid/scheduler.py), just
unrolled into individual .step() calls (instead of bulk .run(n_steps=...)
calls) so every single MC trial and every single MD step can be saved,
timed, and checked individually. run_campaign.py's own
_run_hybrid_with_observables does the same unrolling, for the same reason
(per-block memory diagnostics); this goes one level finer, to per-step.

Critically, mc.synchronize(batch) is called both before the block loop
starts (after the initial npt.compute(batch)) and after every block's MD
sub-phase -- matching HybridMCMD.run() exactly. debug_npt_then_sgc.py's
bare SGC phase skipped this, so its first-ever SGC step silently paid for
an extra, redundant model forward+backward (self._energy was None,
triggering BaseMonteCarlo._initialize_energy() on top of the step's own
compute) -- visible as a one-time jump in reserved memory at the phase
boundary. Baking synchronize() in here removes that artifact.

Saves the FULL configuration after every single MC trial and every single
MD step (not once per block) as a normal FinalStateStore checkpoint, runs
the same minimum-image overlap check on each one, and prints per-step wall
time plus per-step instantaneous GPU memory. Because this alternates real
MC and MD blocks (unlike the two decoupled phases in debug_npt_then_sgc.py),
this is the first test that can actually reproduce a defect specific to
the interleaving itself, if one exists.

Per-phase memory accounting: torch.cuda.reset_peak_memory_stats(device) is
called at the start of each MC sub-phase and each MD sub-phase within every
block, and torch.cuda.max_memory_allocated(device) is read right after that
sub-phase finishes. This is a true per-phase peak (the actual high-water
mark of live tensors during that sub-phase, transient activations
included) -- unlike memory_reserved(), which is a floor that only grows
across the whole process and carries forward from one phase into the next
(see the debug_npt_then_sgc.py conversation this diagnostic followed from).

Run on Quest:
    source hpc/quest/env.sh
    "$QUEST_ENV/bin/python" benchmark/hybrid_sgc_npt/hybrid_diagnostic.py \\
        --out-dir "$RUN_ROOT/hybrid_sgc_npt/hybrid_diagnostic"

--n-blocks defaults to a small diagnostic-scale count (10), not
production's 100-200 (N_BLOCKS_SCAN_STEP / N_BLOCKS_REFERENCE) -- at
mc_steps=round(0.2*n_atoms) + md_steps=50 per block, saving every single
step, 10 blocks on a 500-atom system already means ~1500 checkpoint files.
Raise --n-blocks once this comes back clean at the default scale.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_campaign import (  # noqa: E402
    BAROSTAT_TIME_FS,
    CHECKPOINT,
    CONVENTIONAL_CELL,
    CRYSTAL_STRUCTURE,
    DT_FS,
    INFERENCE_SETTINGS,
    KB_EV,
    LATTICE_A_ANG,
    MC_STEP_FRACTION,
    MD_STEPS_PER_BLOCK,
    PRESSURE_EV_PER_A3,
    SEED,
    SIZE_REPEATS,
    SPECIES,
    TASK,
    TEMPLATE_SYMBOL,
    THERMOSTAT_TIME_FS,
    _refresh_masses_after_transmutation,
    build_ase_structure,
)
from export_structures import _check_min_distance, _to_atoms  # noqa: E402

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.integrators.npt import NPT
from nvalchemi.hybrid import HybridMCMD
from nvalchemi.mc import SGC
from nvalchemi.models.uma import UMAWrapper
from nvalchemi.scheduling.campaign import FinalStateStore


def _reset_phase_memory(device: torch.device) -> None:
    """Start a fresh per-phase peak-memory window."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _report_phase_memory(device: torch.device, label: str) -> None:
    """Report the true peak live-tensor footprint since the last reset."""
    if device.type != "cuda":
        return
    print(
        f"[phase-memory] {label}: peak_allocated={torch.cuda.max_memory_allocated(device) / 1024**3:.3f} GB "
        f"peak_reserved={torch.cuda.max_memory_reserved(device) / 1024**3:.3f} GB",
        flush=True,
    )


def _log_step(run_id: str, wall_s: float, device: torch.device) -> None:
    """Per-step wall time plus instantaneous (not peak) GPU memory."""
    if device.type == "cuda":
        print(
            f"[step] {run_id}: wall={wall_s:.4f}s "
            f"allocated={torch.cuda.memory_allocated(device) / 1024**3:.3f} GB "
            f"reserved={torch.cuda.memory_reserved(device) / 1024**3:.3f} GB",
            flush=True,
        )
    else:
        print(f"[step] {run_id}: wall={wall_s:.4f}s", flush=True)


def _build_initial_state(
    template, temperature_k: float, pt_fraction: float, seed: int, device: torch.device
) -> AtomicData:
    """Fresh alloy state at the requested composition -- same construction as
    run_campaign.py's _walker(), minus the RunSpec/continuation machinery
    this standalone test doesn't need."""
    data = AtomicData.from_atoms(template, device=device)
    n_atoms = data.num_nodes
    generator = torch.Generator(device=device).manual_seed(seed)
    numbers = torch.full_like(data.atomic_numbers, SPECIES[0])
    pt_count = round(pt_fraction * n_atoms)
    numbers[torch.randperm(n_atoms, device=device, generator=generator)[:pt_count]] = SPECIES[1]
    data.atomic_numbers = numbers
    data.atomic_masses = None
    data.use_default_masses()
    velocity_std = torch.sqrt(
        torch.as_tensor(KB_EV * temperature_k, device=device) / data.atomic_masses
    )
    data.velocities = (
        torch.randn((n_atoms, 3), device=device, generator=generator) * velocity_std[:, None]
    )
    data.velocities -= data.velocities.mean(dim=0, keepdim=True)
    data.forces = torch.zeros_like(data.positions)
    data.energy = torch.zeros(1, 1, device=device)
    data.stress = torch.zeros(1, 3, 3, device=device)
    return data


def _save_and_check(
    batch: Batch, store: FinalStateStore, run_id: str, pt_atomic_number: int
) -> AtomicData:
    """Persist this step's state as a normal checkpoint and run the overlap
    check on it immediately, so a bad frame is flagged the moment it appears."""
    state = batch.get_data(0)
    store.save(run_id, state)
    atoms = _to_atoms(state.model_dump(exclude_none=True), run_id, pt_atomic_number)
    _check_min_distance(atoms, run_id)
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True, help="Root for checkpoints/ under this run")
    parser.add_argument("--n-atoms", type=int, default=500)
    parser.add_argument("--pt-fraction", type=float, default=0.50, help="Initial composition (\"50/50\")")
    parser.add_argument("--temperature-k", type=float, default=1400.0)
    parser.add_argument(
        "--n-blocks", type=int, default=10,
        help="Hybrid MC/MD blocks to run (production uses 100-200; kept small here "
        "since every single step is saved -- see module docstring)",
    )
    parser.add_argument(
        "--mc-steps-per-block", type=int, default=None,
        help="Default: round(MC_STEP_FRACTION * n_atoms), matching run_campaign.py",
    )
    parser.add_argument(
        "--md-steps-per-block", type=int, default=MD_STEPS_PER_BLOCK,
        help=f"Default: MD_STEPS_PER_BLOCK ({MD_STEPS_PER_BLOCK}), matching run_campaign.py",
    )
    parser.add_argument("--delta-mu-ref-ev", type=float, default=-2.9662178325653072)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    n_mc_steps = (
        args.mc_steps_per_block
        if args.mc_steps_per_block is not None
        else max(1, round(MC_STEP_FRACTION * args.n_atoms))
    )
    n_md_steps = args.md_steps_per_block
    pt_atomic_number = SPECIES[1]

    checkpoint_dir = args.out_dir / "checkpoints"
    store = FinalStateStore(checkpoint_dir)

    template = build_ase_structure(
        TEMPLATE_SYMBOL, CRYSTAL_STRUCTURE, LATTICE_A_ANG, SIZE_REPEATS[args.n_atoms], cubic=CONVENTIONAL_CELL
    )
    if len(template) != args.n_atoms:
        raise ValueError(f"expected {args.n_atoms} atoms, built {len(template)}")

    print(
        f"[setup] n_atoms={args.n_atoms} pt_fraction={args.pt_fraction} T={args.temperature_k} K "
        f"n_blocks={args.n_blocks} mc_steps_per_block={n_mc_steps} md_steps_per_block={n_md_steps} "
        f"delta_mu_ref_eV={args.delta_mu_ref_ev} device={device}",
        flush=True,
    )

    model = UMAWrapper.from_checkpoint(
        CHECKPOINT, task_name=TASK, device=str(device), inference_settings=INFERENCE_SETTINGS
    )

    data = _build_initial_state(template, args.temperature_k, args.pt_fraction, args.seed, device)
    batch = Batch.from_data_list([data])
    store.save("initial", batch.get_data(0))

    temperatures = torch.tensor([args.temperature_k], device=device)
    pressures = torch.tensor([PRESSURE_EV_PER_A3], device=device)
    sgc = SGC(
        model=model,
        temperature=temperatures,
        species=list(SPECIES),
        chemical_potentials={
            SPECIES[0]: torch.tensor([0.0], device=device),
            SPECIES[1]: torch.tensor([args.delta_mu_ref_ev], device=device),
        },
        random_seed=args.seed,
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
    # Validates mc.model is md.model, same as production; driven manually
    # below (not via hybrid.run()) so every individual step can be saved,
    # timed, and memory-checked -- same reason run_campaign.py's own
    # _run_hybrid_with_observables doesn't call hybrid.run() either.
    hybrid = HybridMCMD(mc=sgc, md=npt, mc_steps=n_mc_steps, md_steps=n_md_steps)

    total_wall_s = 0.0
    with hybrid.md:
        t0 = time.perf_counter()
        hybrid.md.compute(batch)
        hybrid.mc.synchronize(batch)
        print(f"[setup] initial compute+synchronize: wall={time.perf_counter() - t0:.4f}s", flush=True)
        _save_and_check(batch, store, "block0000_setup", pt_atomic_number)

        for block_index in range(1, args.n_blocks + 1):
            # --- MC sub-phase: n_mc_steps individual SGC trials ---
            _reset_phase_memory(device)
            for mc_step in range(1, n_mc_steps + 1):
                t0 = time.perf_counter()
                hybrid.mc.step(batch)
                step_wall_s = time.perf_counter() - t0
                total_wall_s += step_wall_s
                _refresh_masses_after_transmutation(batch)
                run_id = f"block{block_index:04d}_mc{mc_step:04d}"
                _save_and_check(batch, store, run_id, pt_atomic_number)
                _log_step(run_id, step_wall_s, device)
            _report_phase_memory(device, f"block {block_index}/{args.n_blocks} MC phase ({n_mc_steps} steps)")

            # --- force/energy refresh, exactly matching HybridMCMD.run():
            # prevents a rejected MC trial's candidate-state derivatives
            # from being used by the MD integrator. ---
            _reset_phase_memory(device)
            t0 = time.perf_counter()
            hybrid.md.compute(batch)
            refresh_wall_s = time.perf_counter() - t0
            total_wall_s += refresh_wall_s
            print(f"[step] block{block_index:04d}_md_refresh: wall={refresh_wall_s:.4f}s", flush=True)

            # --- MD sub-phase: n_md_steps individual NPT steps ---
            for md_step in range(1, n_md_steps + 1):
                t0 = time.perf_counter()
                hybrid.md.step(batch)
                step_wall_s = time.perf_counter() - t0
                total_wall_s += step_wall_s
                run_id = f"block{block_index:04d}_md{md_step:04d}"
                _save_and_check(batch, store, run_id, pt_atomic_number)
                _log_step(run_id, step_wall_s, device)
            _report_phase_memory(device, f"block {block_index}/{args.n_blocks} MD phase ({n_md_steps} steps)")

            # --- adopt the just-refreshed energy as MC's new baseline,
            # exactly matching HybridMCMD.run() ---
            hybrid.mc.synchronize(batch)

    n_total_steps = args.n_blocks * (n_mc_steps + n_md_steps)
    print(
        f"[done] {args.n_blocks} blocks ({n_mc_steps} MC + {n_md_steps} MD each, "
        f"{n_total_steps} physics steps total) saved to {checkpoint_dir} "
        f"(mc_acceptance={hybrid.mc.stats.acceptance:.4f}); "
        f"total physics wall time={total_wall_s:.2f}s "
        f"(mean {total_wall_s / n_total_steps:.4f}s/step, excludes per-step checkpoint/overlap-check I/O)"
    )


if __name__ == "__main__":
    main()
