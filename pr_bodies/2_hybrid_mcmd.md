<!-- Title: feat(hybrid): add HybridMCMD and nvalchemi.scheduling for hybrid MC-MD campaigns -->
<!-- markdownlint-disable MD013-->
# ALCHEMI Toolkit Pull Request

## Description

Adds `nvalchemi.hybrid.HybridMCMD`, which alternates MC and MD blocks on one batch (for example semi-grand-canonical NPT). It also adds `nvalchemi.scheduling`, for sizing and ordering many such runs: profiled batch widths, per-GPU waves, and restartable, dependency-ordered campaigns of state points.

**Stacked on the Monte Carlo PR** (NVIDIA/nvalchemi-toolkit#<MC PR>), which itself sits on NVIDIA/nvalchemi-toolkit#217. Review only the last two commits.

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [x] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Performance improvement
- [x] Documentation update
- [ ] Refactoring (no functional changes)
- [ ] CI/CD or infrastructure change

## Related Issues

Depends on the Monte Carlo PR (NVIDIA/nvalchemi-toolkit#<MC PR>) and NVIDIA/nvalchemi-toolkit#217.

## Changes Made

- `HybridMCMD(mc, md, mc_steps, md_steps, ...)`:
  - Runs MC blocks, then refreshes forces and stress, then runs MD from the accepted state.
  - `mc_energy_only=True` narrows `active_outputs` to `{"energy"}` for MC blocks (#217 then skips the backward pass) and restores the full outputs for MD.
  - MC and MD may use separate models of one potential, for example an unmerged UMA for SGC and a merged one for MD. Each MC block then re-evaluates its baseline with the MC model, so acceptance never mixes evaluation paths.
  - `before_md_block` and `after_md_block` hooks; `run(on_block=...)` for per-block observables and early stopping; `md_steps=0` for MC-only blocks (for example a cell relaxation between rigid-lattice SGC blocks).
  - Public `run_mc_block` and `run_md_block` for custom loops.
- `SimulationBatchPlanner`:
  - `profile` measures CUDA reserved memory and throughput at candidate widths.
  - `recommend_width` picks the smallest width near peak throughput that keeps memory headroom.
  - `infer_memory_model` and `estimate_width` extrapolate to other GPUs.
  - `assign_runs` packs runs into per-GPU serial waves.
- `RunSpec`, `CampaignSpec` (`cooling_from_reference`, `delta_mu_scan_*`), `CampaignScheduler` and `FinalStateStore`:
  - `parent_id` continuation edges and `depends_on` barriers.
  - Atomic final-state writes; completion is rebuilt from disk on restart.
  - Ready runs batch together across temperature, pressure and chemical potential.
- `examples/advanced/17_uma_hybrid_sgc_npt.py`: Au-Pt SGC-NPT with UMA, with a batch-width pre-run and a small campaign.
- Docs:
  - "Hybrid MC-MD" and "Batch planning and campaigns" sections in `docs/userguide/dynamics_simulations.md`.
  - API pages `docs/modules/hybrid.rst` and `docs/modules/scheduling.rst`.
  - CHANGELOG entries.
- Agent skill: routing row, version check, and hybrid and scheduling sections in `references/dynamics-optimization.md`.

## Testing

- [x] Unit tests pass on CUDA: `test/hybrid test/scheduling test/mc --slow` all pass on a GPU node, including the CUDA-only sync checks.
- [x] Unit tests pass on CPU (`DemoModelWrapper`): the same suites give 79 passed; the 4 CUDA-only sync checks skip.
- [x] Linting passes (pre-commit)
- [x] New tests added for new functionality meets coverage expectations?

- [x] `examples/advanced/17_uma_hybrid_sgc_npt.py` runs to completion on a GPU with UMA.

Example 17 needs a CUDA GPU and access to the gated UMA checkpoint. Like `09_uma_nve.py`, it exits early with a setup hint when the checkpoint is unavailable.

## Checklist

- [x] I have read and understand the [Contributing Guidelines](../CONTRIBUTING.md)
- [x] I have updated the [CHANGELOG.md](../CHANGELOG.md)
- [x] I have performed a self-review of my code
- [x] I have added docstrings to new functions/classes
- [x] I have updated the documentation (if applicable)

## Additional Notes

`FusedStage` (`mc + md`) is not a substitute for `HybridMCMD`. It runs MC and MD on different graphs, whereas alternating on the same graph needs a candidate-energy evaluation and an accepted-state force evaluation at different points in each block.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01HBtAWFoh4ViZnY6avLBFhg
