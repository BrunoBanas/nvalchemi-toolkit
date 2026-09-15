#!/usr/bin/env python3
"""Convert a single hybrid MC-MD checkpoint (.pt) to an OVITO-readable file.

Standalone, local-machine companion to export_structures.py: takes ONE
<run_id>.pt checkpoint -- as written by
nvalchemi.scheduling.campaign.FinalStateStore.save (the
{"run_id": ..., "state": {...}, "runtime_state": {...}} wrapper), or a bare
AtomicData.model_dump() payload -- and writes one Extended XYZ (.extxyz)
frame. OVITO's XYZ reader auto-detects the extended-XYZ header
(Lattice=... Properties=...) and imports cell, PBC, species, and positions
directly, with no manual column mapping.

Deliberately depends on torch + ase only, not on nvalchemi itself (same
reasoning as export_structures.py): `pip install torch ase` on a laptop is
enough -- no CUDA, no HPC environment, no need to be on Quest. scp/rsync a
single checkpoint down from $RUN_ROOT and convert it here instead of
waiting on the Quest queue for an export job.

Usage:
    python local_export_structure.py atoms500.T1400.SGCtrial0042.pt
    python local_export_structure.py atoms500.T1400.SGCtrial0042.pt -o frame.extxyz

Then in OVITO: File > Load File (or drag-and-drop) the .extxyz output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ase import Atoms
from ase.io import write

# Same allow-list as export_structures.py: AtomicData.model_dump() includes a
# plain `set` in one of its fields, which some torch versions don't
# allow-list under weights_only=True by default (PyTorch's own suggested
# fix for the resulting UnpicklingError). Safe here: this only widens the
# allow-list by one builtin container type, not to arbitrary classes, and
# you're loading your own trusted checkpoint file.
torch.serialization.add_safe_globals([set])


def _load_state(path: Path) -> dict:
    """Load one checkpoint's raw AtomicData field dict.

    Tolerates either the FinalStateStore wrapper
    ({"run_id": ..., "state": {...}, "runtime_state": {...}}) or a bare
    AtomicData.model_dump() payload, so this also works on a checkpoint you
    hand-saved yourself with plain torch.save(state.model_dump(...)).
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "state" in payload and "atomic_numbers" not in payload:
        return payload["state"]
    return payload


def _to_atoms(state: dict, pt_atomic_number: int | None) -> Atoms:
    """Rebuild an ASE ``Atoms`` from one AtomicData's raw field dict.

    Tensors are normally already on CPU here (``torch.load`` above uses
    ``map_location="cpu"``), but ``.detach().cpu()`` before every
    ``.numpy()`` costs nothing when that's already true and protects
    against a state dict built from a still-on-device tensor (see
    export_structures.py's own ``_to_atoms`` for the bug this avoids).
    """

    def _np(tensor):
        return tensor.detach().cpu().numpy()

    atomic_numbers = _np(state["atomic_numbers"])
    positions = _np(state["positions"])
    cell = state.get("cell")
    pbc = state.get("pbc")
    # cell/pbc carry a leading graph-count dimension even for a single
    # structure -- [1, 3, 3] / [1, 3] -- squeeze it off before handing
    # shapes to ASE (matches AtomicData.from_atoms's convention).
    atoms = Atoms(
        numbers=atomic_numbers,
        positions=positions,
        cell=_np(cell).reshape(3, 3) if cell is not None else None,
        pbc=_np(pbc).reshape(3) if pbc is not None else False,
    )
    if cell is not None and atoms.pbc.any():
        # NPT/MC moves don't keep positions folded into the primary cell --
        # wrap so the frame shows one contiguous cell's worth of atoms.
        atoms.wrap()
    if pt_atomic_number is not None:
        atoms.info["pt_fraction"] = float((atomic_numbers == pt_atomic_number).mean())
    energy = state.get("energy")
    if energy is not None:
        energy_ev = float(_np(energy).reshape(-1)[0])
        atoms.info["energy_eV"] = energy_ev
        atoms.info["energy_eV_per_atom"] = energy_ev / len(atoms)
    return atoms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path, help="Path to a single <run_id>.pt checkpoint")
    parser.add_argument(
        "-o", "--out", type=Path, default=None,
        help="Output path (default: <checkpoint's name>.extxyz next to the input). "
        "A .xyz extension is also written as extended XYZ (plain XYZ can't carry "
        "cell/PBC, which OVITO wants); any other extension (.cfg, .lammpstrj, ...) "
        "is inferred and dispatched by ase.io.write normally.",
    )
    parser.add_argument(
        "--pt-atomic-number", type=int, default=78,
        help="Atomic number used for the pt_fraction info field (this project's "
        "Au-Pt default: Pt=78). Irrelevant to structure correctness -- pass "
        "--no-pt-fraction for other element systems.",
    )
    parser.add_argument(
        "--no-pt-fraction", action="store_true",
        help="Skip the pt_fraction info field entirely (for non-Au-Pt checkpoints).",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise SystemExit(f"no such file: {args.checkpoint}")

    out_path = args.out or args.checkpoint.with_suffix(".extxyz")
    pt_atomic_number = None if args.no_pt_fraction else args.pt_atomic_number

    state = _load_state(args.checkpoint)
    atoms = _to_atoms(state, pt_atomic_number)

    fmt = "extxyz" if out_path.suffix in (".extxyz", ".xyz") else None
    write(out_path, atoms, format=fmt)

    print(f"{args.checkpoint} -> {out_path} ({len(atoms)} atoms)")
    if "energy_eV_per_atom" in atoms.info:
        print(f"energy: {atoms.info['energy_eV_per_atom']:.6f} eV/atom")
    if "pt_fraction" in atoms.info:
        print(f"pt_fraction: {atoms.info['pt_fraction']:.4f}")


if __name__ == "__main__":
    main()
