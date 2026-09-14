"""Export equilibrated final-state checkpoints from a delta-mu scan campaign
to viewable structure trajectories.

Reads every ``<run_id>.pt`` checkpoint written by ``run_campaign.py`` /
``FinalStateStore`` under ``--checkpoint-root`` and writes one extended-XYZ
trajectory per hysteresis branch (temperature x A-rich/B-rich), ordered along
the delta_mu ladder (seed first, then dmu1, dmu2, ...), plus a CSV summary of
composition and energy per structure.

Deliberately depends on ``torch`` + ``ase`` only, not on ``nvalchemi``: a
checkpoint's payload is ``{"run_id": ..., "state": AtomicData.model_dump(),
"runtime_state": {...}}`` (see ``FinalStateStore.save``), and
``model_dump()`` leaves every tensor field as a plain ``torch.Tensor`` inside
plain dicts -- so the raw positions/atomic_numbers/cell/pbc/energy fields can
be read with ``torch.load(..., weights_only=True)`` without importing the
AtomicData class itself. That keeps this script runnable anywhere the
campaign's own environment (``$QUEST_ENV``) is available, with no extra
install.

Usage (from a Quest shell, after ``source hpc/quest/env.sh``):

    "$QUEST_ENV/bin/python" benchmark/hybrid_sgc_npt/export_structures.py \\
        --checkpoint-root "$RUN_ROOT/hybrid_sgc_npt/checkpoints/scan_1200_1400_hybrid/atoms500"

Then copy the resulting ``structures/`` folder back alongside the checkpoints
(the same way you already copy Results) and open the ``.extxyz`` files in
OVITO, VMD, or ``ase gui`` -- each file is one branch's structures in scan
order, so stepping through frames shows that branch's equilibrated
composition evolving along the delta_mu ladder.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch
from ase import Atoms
from ase.io import write

# AtomicData.model_dump() includes a plain ``set`` in one of its fields, which
# some torch versions don't allow-list under weights_only=True by default
# (PyTorch's own suggested fix for the resulting UnpicklingError). Safe here:
# these are your own trusted checkpoint files, and this only widens the
# allow-list by one builtin container type, not to arbitrary classes.
torch.serialization.add_safe_globals([set])

_RUN_ID_RE = re.compile(
    r"^atoms(?P<n_atoms>\d+)\.T(?P<temperature>\d+)\.(?P<branch>Arich|Brich)"
    r"(?:\.dmu(?P<dmu_index>\d+)\.mu(?P<mu>-?[0-9.]+))?$"
)


def _load_state(path: Path) -> dict:
    """Load one checkpoint's raw AtomicData payload (tensors only, no class import)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return payload["state"]


def _to_atoms(state: dict, run_id: str, pt_atomic_number: int) -> Atoms:
    """Rebuild an ASE ``Atoms`` from one AtomicData's raw field dict."""
    atomic_numbers = state["atomic_numbers"].numpy()
    positions = state["positions"].numpy()
    cell = state.get("cell")
    pbc = state.get("pbc")
    atoms = Atoms(
        numbers=atomic_numbers,
        positions=positions,
        cell=cell.numpy() if cell is not None else None,
        pbc=pbc.numpy() if pbc is not None else False,
    )
    pt_fraction = float((atomic_numbers == pt_atomic_number).mean())
    atoms.info["run_id"] = run_id
    atoms.info["pt_fraction"] = pt_fraction
    energy = state.get("energy")
    if energy is not None:
        energy_ev = float(energy.reshape(-1)[0])
        atoms.info["energy_eV"] = energy_ev
        atoms.info["energy_eV_per_atom"] = energy_ev / len(atoms)
    return atoms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-root", type=Path, required=True, help="Directory of <run_id>.pt checkpoints")
    parser.add_argument("--out", type=Path, default=None, help="Output directory (default: <checkpoint-root>/structures)")
    parser.add_argument("--pt-atomic-number", type=int, default=78, help="Atomic number used for the Pt-fraction column")
    args = parser.parse_args()

    out_dir = args.out or (args.checkpoint_root / "structures")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    branches: dict[tuple[str, str], list[tuple[int, Atoms]]] = {}
    checkpoint_paths = sorted(args.checkpoint_root.glob("*.pt"))
    if not checkpoint_paths:
        raise SystemExit(f"no .pt checkpoints found under {args.checkpoint_root}")

    for path in checkpoint_paths:
        run_id = path.stem
        match = _RUN_ID_RE.match(run_id)
        if match is None:
            print(f"[skip] unrecognized run_id format: {run_id}")
            continue
        atoms = _to_atoms(_load_state(path), run_id, args.pt_atomic_number)
        temperature = match["temperature"]
        branch = match["branch"]
        dmu_index = int(match["dmu_index"]) if match["dmu_index"] is not None else 0  # seed sorts first
        branches.setdefault((temperature, branch), []).append((dmu_index, atoms))
        rows.append(
            {
                "run_id": run_id,
                "temperature_K": temperature,
                "branch": branch,
                "dmu_index": dmu_index,
                "mu_eV": match["mu"] if match["mu"] is not None else "",
                "n_atoms": len(atoms),
                "pt_fraction": atoms.info["pt_fraction"],
                "energy_eV_per_atom": atoms.info.get("energy_eV_per_atom", ""),
            }
        )

    for (temperature, branch), frames in sorted(branches.items()):
        frames.sort(key=lambda pair: pair[0])
        traj_path = out_dir / f"T{temperature}_{branch}.extxyz"
        write(traj_path, [atoms for _, atoms in frames], format="extxyz")
        print(f"wrote {len(frames)} frames -> {traj_path}")

    summary_path = out_dir / "structures_summary.csv"
    fieldnames = ["run_id", "temperature_K", "branch", "dmu_index", "mu_eV", "n_atoms", "pt_fraction", "energy_eV_per_atom"]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: (r["temperature_K"], r["branch"], r["dmu_index"])))
    print(f"wrote summary -> {summary_path} ({len(rows)} structures)")


if __name__ == "__main__":
    main()
