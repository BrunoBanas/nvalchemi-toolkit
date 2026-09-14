"""Standalone diagnostic: NPT-then-SGC full-trajectory dump.

Runs 300 raw NPT steps (equilibration, no MC at all) on a fresh 50/50 Au-Pt
500-atom FCC alloy at 1400 K, then switches to 0.2*n_atoms = 100 raw SGC
trial moves at a fixed delta_mu_ref_eV -- two sequential, NON-interleaved
phases (unlike the real campaign's block-interleaved HybridMCMD), built to
isolate whether the close-contact defect found in the delta-mu scan comes
from NPT alone, from SGC alone, or only shows up once the two are combined.

Saves the FULL configuration after every single step of BOTH phases (not
just once per block, the campaign's own granularity) as a normal
FinalStateStore checkpoint, so the whole trajectory can be walked afterwards
frame by frame. Also runs the same minimum-image overlap check and prints
GPU memory every step, live, so a crash mid-run still leaves a usable trace.

Reuses run_campaign.py's own constants and _refresh_masses_after_transmutation
(the mass-desync fix) plus export_structures.py's _to_atoms/_check_min_distance,
via plain sibling-script imports -- run this from the same directory as those
two files (matches how run_campaign.py itself is invoked).

Run on Quest:
    source hpc/quest/env.sh
    "$QUEST_ENV/bin/python" benchmark/hybrid_sgc_npt/debug_npt_then_sgc.py \\
        --out-dir "$RUN_ROOT/hybrid_sgc_npt/debug_npt_then_sgc"

Then convert/inspect the resulting checkpoints/*.pt the same way as any
other campaign checkpoint directory, e.g. with export_structures.py against
--checkpoint-root pointed at this run's checkpoints/ folder (its run_ids
won't match that script's <run_id> regex, which is fine -- this script
writes its own two-phase .extxyz trajectories directly, below).
"""

from __future__ import annotations

import argparse
import sys
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
    PRESSURE_EV_PER_A3,
    SEED,
    SIZE_REPEATS,
    SPECIES,
    TASK,
    TEMPLATE_SYMBOL,
    _refresh_masses_after_transmutation,
    build_ase_structure,
)
from export_structures import _check_min_distance, _to_atoms  # noqa: E402

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.integrators.npt import NPT
from nvalchemi.mc import SGC
from nvalchemi.models.uma import UMAWrapper
from nvalchemi.scheduling.campaign import FinalStateStore


def _print_memory(device: torch.device, label: str) -> None:
    if device.type != "cuda":
        return
    print(
        f"[memory] {label}: allocated={torch.cuda.memory_allocated(device) / 1024**3:.3f} GB "
        f"reserved={torch.cuda.memory_reserved(device) / 1024**3:.3f} GB",
        flush=True,
    )


def _build_initial_state(
    template, temperature_k: float, pt_fraction: float, seed: int, device: torch.device
) -> AtomicData:
    """Fresh 500-atom state at the requested composition -- same construction
    as run_campaign.py's _walker(), minus the RunSpec/continuation machinery
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
    parser.add_argument("--n-npt-steps", type=int, default=300)
    parser.add_argument(
        "--n-sgc-steps", type=int, default=None,
        help="Default: round(0.2 * n_atoms), matching MC_STEP_FRACTION in run_campaign.py",
    )
    parser.add_argument("--delta-mu-ref-ev", type=float, default=-2.9662178325653072)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    n_sgc_steps = args.n_sgc_steps if args.n_sgc_steps is not None else round(0.2 * args.n_atoms)
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
        f"n_npt_steps={args.n_npt_steps} n_sgc_steps={n_sgc_steps} "
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
    npt = NPT(
        model=model,
        dt=DT_FS,
        temperature=temperatures,
        pressure=pressures,
        thermostat_time=THERMOSTAT_TIME_FS,
        barostat_time=BAROSTAT_TIME_FS,
        pressure_coupling="isotropic",
    )
    chemical_potentials = {
        SPECIES[0]: torch.tensor([0.0], device=device),
        SPECIES[1]: torch.tensor([args.delta_mu_ref_ev], device=device),
    }
    sgc = SGC(
        model=model,
        temperature=temperatures,
        species=list(SPECIES),
        chemical_potentials=chemical_potentials,
        random_seed=args.seed,
    )

    # Phase 1: pure NPT equilibration -- no MC at all.
    print("[phase] starting NPT equilibration", flush=True)
    with npt:
        npt.compute(batch)
        for step in range(1, args.n_npt_steps + 1):
            npt.step(batch)
            run_id = f"npt_eq_step{step:04d}"
            _save_and_check(batch, store, run_id, pt_atomic_number)
            _print_memory(device, f"npt step {step}/{args.n_npt_steps}")

    # Phase 2: pure SGC on the NPT-equilibrated structure -- no MD at all.
    # Matches production: SGC is always called bare, never inside its own
    # `with sgc:` block (see HybridMCMD.run -- only `with self.md:` wraps it).
    print("[phase] starting SGC trials", flush=True)
    for step in range(1, n_sgc_steps + 1):
        sgc.step(batch)
        _refresh_masses_after_transmutation(batch)
        run_id = f"sgc_trial{step:04d}"
        _save_and_check(batch, store, run_id, pt_atomic_number)
        _print_memory(device, f"sgc trial {step}/{n_sgc_steps}")

    print(
        f"[done] {args.n_npt_steps} NPT steps + {n_sgc_steps} SGC trials saved to {checkpoint_dir} "
        f"(mc_acceptance={sgc.stats.acceptance:.4f})"
    )


if __name__ == "__main__":
    main()
