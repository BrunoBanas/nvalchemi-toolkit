<!-- Title: feat(uma): select fairchem_cpp segment_mm MoLE layers via moe_layer_type -->
<!-- markdownlint-disable MD013-->
# ALCHEMI Toolkit Pull Request

## Description

Lets `UMAWrapper` run UMA's unmerged mixture-of-linear-experts (MoLE) layers through fairchem's `fairchem_cpp` `segment_mm` CUDA extension instead of fairchem's per-system Python loop.

Unmerged layers (`merge_mole=False`) are required whenever composition changes, as in SGC / VC-SGC Monte Carlo, or when systems of different composition share a batch. In those cases the loop is the bottleneck. On an A100-SXM4-80GB with `uma-s-1p2` and 500-atom Au-Pt systems, the extension gives:

| Workload | 1 system | 4 systems |
| --- | --- | --- |
| MC energy call | 1.57x faster | 1.95x faster |
| MD call (energy, forces, stress) | 1.01x | 1.17x |
| SGC-NPT hybrid MC-MD, time per block | 1.28x faster | 1.53x faster |

This costs about 15% more peak GPU memory. Results agree with the loop to ~1e-5 eV/atom and ~5e-4 eV/Å (measured below on an A100: 3.8e-6 eV/atom and 1.5e-4 eV/Å on a mixed Au-Pt batch).

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [x] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [x] Performance improvement
- [x] Documentation update
- [ ] Refactoring (no functional changes)
- [ ] CI/CD or infrastructure change

## Related Issues

Stacked on NVIDIA/nvalchemi-toolkit#217 (which contains #216): the resolution reads #217's `key=value` `inference_settings` spec, and the change is applied to #217's `uma.py`. Until #217 merges, this diff also shows #216 and #217. Review only the last commit. It is independent of the Monte Carlo and hybrid PRs.

## Changes Made

- `UMAWrapper.from_checkpoint(moe_layer_type="auto" | "pytorch" | "dgl")`; `"dgl"` is fairchem's name for the extension.
  - `"auto"` (the default) selects the extension only where it is known to work: a registered `uma-*` checkpoint with `merge_mole=False`, `compile=False` and `activation_checkpointing=False`. If `fairchem_cpp` is not installed, it warns and falls back to the loop.
  - An explicit `"dgl"` raises `ImportError` when `fairchem_cpp` is missing.
  - The `NVALCHEMI_UMA_MOE_LAYER_TYPE` environment variable sets the `"auto"` behaviour per job: `=dgl` turns a missing extension into an error, `=pytorch` opts out.
  - A `backbone.moe_layer_type` in `overrides` counts as an explicit request: it goes through the same checks as the argument, conflicting with an explicit value raises `ValueError`, and the resolved value replaces it in a copy of `overrides` (the caller's dict is never mutated).
- Guards for cases where `MOLEDGL` does not work:
  - `merge_mole=True`: fairchem's merge step needs `MOLE.merged_linear_layer`, which `MOLEDGL` lacks.
  - `activation_checkpointing=True`: edges are processed in 131,072-edge chunks, and `segment_mm` raises "Segment index out of bound" past the first chunk (above about 2,300 bulk atoms).

  Both always use the loop, with a warning when `"dgl"` was requested explicitly. `"dgl"` under `compile=True` warns as untested: `segment_mm` has no fake-tensor kernel, so dynamo may graph-break at every MoLE layer.
- Availability is read from `fairchem.core.models.uma.nn.mole.fairchem_cpp_found`, which a real `import fairchem_cpp` sets. The same-named flag in `mole_utils` is unconditionally `True` in fairchem 2.22–2.24, so fairchem's own `"dgl"` guard never fires.
- `UMAWrapper.moe_layer_type` and `UMAWrapper.mole_layer_counts` report the layers that were built. A resolved `"dgl"` that builds no `MOLEDGL` layer raises `RuntimeError`, for example when the override did not reach the backbone.
- User guide (`docs/userguide/models.md`): when to use the extension, the measurements above, and how to build `fairchem_cpp`, which is not on PyPI and lives on fairchem's `fairchem_cpp` branch.
- CHANGELOG entry under Unreleased / Added.

## Testing

- [x] Unit tests pass on CUDA: `test/models/test_uma.py --slow` passes in full on a GPU node with the UMA checkpoint, including the real-checkpoint tests (`TestOMolEquivalence`, `TestChargedInputs`) and the CUDA-only `TestTurboCompile`. The two UMA multi-GPU tests pass on 2 GPUs.
- [x] Unit tests pass on CPU (fairchem-core 2.24.0): 115 passed (#217's tests plus 22 new `moe_layer_type` cases). The resolution logic and the `from_checkpoint` plumbing (including overrides under `merge_mole=True`, activation checkpointing and a missing `fairchem_cpp`) are tested with fairchem's loader mocked.
- [x] CUDA: `TestMoLEDGLExtension` (`--slow`, skipped without CUDA, `fairchem_cpp` or the checkpoint) loads the checkpoint with each implementation, checks that real `MOLEDGL` layers were built, and compares energy, forces and stress on a batch of three Au-Pt systems of different composition (4, 16 and 28 Pt out of 32 atoms). Both tests pass on an A100-SXM4-80GB (fairchem-core 2.22.0, torch 2.13.0+cu126, `fairchem_cpp` from fairchem's `fairchem_cpp` branch, `uma-s-1p2`):
  - Layers: the `dgl` model builds 24 `MOLEDGL` and 3 `MOLE` layers (the 3 are fairchem's per-dataset output head, which always uses the loop); the `pytorch` model builds 27 `MOLE`.
  - Agreement with the loop, max |dgl − pytorch|: energy 3.8e-6 eV/atom (test tolerance 1e-4), forces 1.5e-4 eV/Å with the largest force 1.29 eV/Å (tolerance 2e-3), stress 2.5e-6 eV/Å³ with the largest stress 7.6e-2 (tolerance 1e-4).
- [x] Linting passes (pre-commit: ruff, ruff format, interrogate, markdownlint, license)
- [x] New tests added for new functionality meets coverage expectations?

The speedups above come from earlier A100 runs; they were not re-run for this PR.

## Checklist

- [x] I have read and understand the [Contributing Guidelines](../CONTRIBUTING.md)
- [x] I have updated the [CHANGELOG.md](../CHANGELOG.md)
- [x] I have performed a self-review of my code
- [x] I have added docstrings to new functions/classes
- [x] I have updated the documentation (if applicable)

## Additional Notes

For a single small system (~100 atoms) the MD call is CPU-bound and runs about 20% slower with the extension. Pass `moe_layer_type="pytorch"` there.

fairchem's per-dataset output head always uses the loop (3 small layers on `uma-s-1p2`). `uv sync` removes `fairchem_cpp`, since it is not in the lockfile, so it must be rebuilt after every sync.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01HBtAWFoh4ViZnY6avLBFhg
