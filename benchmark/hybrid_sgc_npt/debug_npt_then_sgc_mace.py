"""Standalone diagnostic: NPT-then-SGC full-trajectory dump, MACE energy engine.

MACE counterpart of ``debug_npt_then_sgc.py``. Runs the exact same protocol
against a MACE-MP foundation model instead of UMA: 300 raw NPT steps
(equilibration, no MC at all) on a fresh 50/50 Au-Pt 500-atom FCC alloy at
1400 K, then 0.2*n_atoms = 100 raw SGC trial moves at a fixed
delta_mu_ref_eV -- two sequential, NON-interleaved phases (unlike the real
campaign's block-interleaved HybridMCMD).

Only three things differ from the UMA original:

1. Model loading: :class:`~nvalchemi.models.mace.MACEWrapper` in place of
   :class:`~nvalchemi.models.uma.UMAWrapper` (``--mace-checkpoint`` /
   ``--dtype`` / ``--enable-cueq`` / ``--compile-model`` replace
   ``CHECKPOINT`` / ``TASK`` / ``INFERENCE_SETTINGS``).
2. Neighbor list wiring: UMA builds its own neighbor graph internally
   (``UMAWrapper.model_config.neighbor_config`` is ``None``), so the UMA
   script never registers a hook. MACE expects the framework to maintain
   its COO neighbor list (see ``nvalchemi/models/mace.py``'s module
   docstring), so this script explicitly registers a
   :class:`~nvalchemi.hooks.NeighborListHook` on BOTH the NPT integrator
   (rebuilds every MD step as positions/cell move) and the SGC sampler
   (rebuilds before every trial's energy evaluation -- cheap insurance;
   positions are frozen during the pure-SGC phase so the list computed at
   the end of NPT would in practice stay valid, but rebuilding on the
   sampler too removes any dependence on integrator step ordering).
3. ``--delta-mu-ref-ev`` has NO default here (it does for the UMA script,
   hardcoded from a prior UMA calibration). MACE's raw per-element energy
   convention is a different model with a different offset -- reusing the
   UMA number would silently encode the wrong chemical-potential origin
   (see ``nvalchemi.mc.SGC``'s docstring and
   ``reference_energy_calibration_mace.py``'s module docstring for why).
   Run that calibration script first and pass its
   ``reference[T]["delta_mu_ref_eV"]`` here.

Everything else -- structure construction, the two-phase NPT-then-SGC
protocol, the mass-desync fix, the min-image overlap check, the
per-step FinalStateStore checkpointing -- is identical and reused from
run_campaign.py / export_structures.py exactly as the UMA script does,
which is the point: nvalchemi's BaseModelMixin interface makes the dynamics
code model-agnostic (see docs/userguide/models.md).

Run on Quest:
    source hpc/quest/env.sh
    "$QUEST_ENV_MACE/bin/python" benchmark/hybrid_sgc_npt/debug_npt_then_sgc_mace.py \\
        --out-dir "$RUN_ROOT/hybrid_sgc_npt/debug_npt_then_sgc_mace" \\
        --delta-mu-ref-ev <value from reference_energy_calibration_mace.py at T=1400 K>

Then convert/inspect the resulting checkpoints/*.pt the same way as any
other campaign checkpoint directory -- see debug_npt_then_sgc.py's own
docstring for details (identical here).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_campaign import (  # noqa: E402
    BAROSTAT_TIME_FS,
    CONVENTIONAL_CELL,
    CRYSTAL_STRUCTURE,
    DT_FS,
    KB_EV,
    LATTICE_A_ANG,
    PRESSURE_EV_PER_A3,
    SEED,
    SIZE_REPEATS,
    SPECIES,
    TEMPLATE_SYMBOL,
    THERMOSTAT_TIME_FS,
    _refresh_masses_after_transmutation,
    build_ase_structure,
)
from export_structures import _check_min_distance, _to_atoms  # noqa: E402

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.integrators.npt import NPT
from nvalchemi.hooks import NeighborListHook
from nvalchemi.mc import SGC
from nvalchemi.models.mace import MACEWrapper
from nvalchemi.scheduling.campaign import FinalStateStore

# MACE-MP checkpoint download cache: honor XDG_CACHE_HOME (see mace.tools.utils
# .get_cache_dir) the same way run_campaign.py's UMA path honors HF_HOME --
# both point at $RUN_ROOT/cache via hpc/quest/env.sh so neither fills $HOME's
# default quota on Quest. No-op off Quest.

_DTYPES = {"float32": torch.float32, "float64": torch.float64}


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
    this standalone test doesn't need. Identical to the UMA script's version."""
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
    parser.add_argument(
        "--delta-mu-ref-ev", type=float, required=True,
        help="mu(Pt) - mu(Au) for THIS MACE checkpoint, from "
        "reference_energy_calibration_mace.py's reference[T]['delta_mu_ref_eV'] at "
        "--temperature-k. No default -- see module docstring point 3.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--mace-checkpoint", type=str, default="medium-mpa-0",
        help="Named MACE-MP foundation checkpoint (auto-downloaded and cached under "
        "XDG_CACHE_HOME/mace) or a local .model/.pt path. Default 'medium-mpa-0' -- "
        "mace-torch's own current default (MPtrj + Alexandria), 89-element coverage "
        "including Au and Pt.",
    )
    parser.add_argument(
        "--dtype", type=str, default="float32", choices=sorted(_DTYPES),
        help="Cast MACE weights to this dtype. float32 matches the toolkit docs' "
        "GPU-throughput recommendation; float64 trades speed for the precision some "
        "MACE-MP checkpoints were originally released at.",
    )
    parser.add_argument(
        "--enable-cueq", action="store_true",
        help="Convert to cuEquivariance format for GPU speedup (requires the "
        "cuequivariance-torch + cuequivariance-ops-torch-cuXX packages, i.e. the "
        "toolkit's 'mace' extra installed alongside 'cu12' or 'cu13'). Off by default "
        "for the first correctness pass.",
    )
    parser.add_argument(
        "--compile-model", action="store_true",
        help="torch.compile the model (inference-only afterward). Off by default: for "
        "a short 400-step debug run, compilation overhead can dominate wall time.",
    )
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
        f"delta_mu_ref_eV={args.delta_mu_ref_ev} device={device} "
        f"mace_checkpoint={args.mace_checkpoint} dtype={args.dtype} "
        f"enable_cueq={args.enable_cueq} compile_model={args.compile_model}",
        flush=True,
    )

    model = MACEWrapper.from_checkpoint(
        args.mace_checkpoint,
        device=device,
        dtype=_DTYPES[args.dtype],
        enable_cueq=args.enable_cueq,
        compile_model=args.compile_model,
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

    # MACE needs the framework to maintain its COO neighbor list (unlike UMA,
    # which builds its own internally -- see module docstring point 2). Two
    # separate hook instances, one per dynamics engine that calls the model,
    # both firing at BEFORE_COMPUTE.
    npt.register_hook(
        NeighborListHook(model.model_config.neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE)
    )
    sgc.register_hook(
        NeighborListHook(model.model_config.neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE)
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
