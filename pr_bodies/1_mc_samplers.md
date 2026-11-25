<!-- Title: feat(mc): add batched GPU-resident Monte Carlo samplers -->
<!-- markdownlint-disable MD013-->
# ALCHEMI Toolkit Pull Request

## Description

Adds `nvalchemi.mc`, a batched Monte Carlo package built on `BaseDynamics`. Each step makes one proposal per active graph, evaluates every trial in one model call, and accepts or rejects each graph on its own. Proposals, acceptance and rejection stay on the device, with no device-to-host synchronization per step.

**Stacked on NVIDIA/nvalchemi-toolkit#217** (which contains NVIDIA/nvalchemi-toolkit#216). Energy-only MC uses #217's derivative gating, and the UMA guidance uses its `key=value` settings spec. Until #217 merges, this diff also shows #216 and #217. Review only the commits after #217, from `feat(mc): add batched GPU-resident Monte Carlo samplers` on.

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [x] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Performance improvement
- [x] Documentation update
- [ ] Refactoring (no functional changes)
- [ ] CI/CD or infrastructure change

## Related Issues

Depends on NVIDIA/nvalchemi-toolkit#217. NVIDIA/nvalchemi-toolkit#216 is reached through #217.

## Changes Made

- `BaseMonteCarlo` (`nvalchemi/mc/base.py`): the trial/accept/restore cycle, Metropolis acceptance per graph, and mass updates on accepted moves from a per-graph species-to-mass lookup (custom masses are kept and never cross between graphs). `state_dict` / `load_state_dict` carry the generator state, the per-graph mass lookup and the acceptance counters, so a restored run continues exactly as an uninterrupted one. Temperatures, species and chemical potentials are cached on the device, and there is shared sync-free selection and pair-swap machinery.
- `SGC`: semi-grand-canonical single-site transmutations. Chemical potentials can be scalars or one value per graph.
- `VCSGC`: binary variance-constrained SGC (Sadigh et al., PRB 85, 184203, 2012).
  - Parametrised by `phi` or by `target_concentration` with an intensive `kappa`.
  - Returns `mu_B - mu_A` for each graph.
  - `reference_exchange_potential` centres the constraint for ML potentials.
- `Kawasaki`: nearest-neighbour swaps at fixed composition on a short-range proposal graph, rebuilt by both `synchronize` and `refresh_energy` after positions change.
  - By default only unlike-species pairs are drawn, with the Hastings factor `n(x)/n(x')`.
- `NonlocalSwap`: fixed-composition swaps of two unlike atoms at any distance, so one accepted move carries a solute across the system, e.g. between a nanoparticle's core and surface.
  - Each step draws, per system, one atom uniformly from all its atoms and a second uniformly from the atoms of another species. No neighbour graph is built or kept between steps.
  - The pair probability depends only on the species counts, which a swap preserves, so plain Metropolis is exact. Single-species systems are skipped.
- `FusedStage` interop (`nvalchemi/dynamics/base.py`): a sub-stage can declare `_stale_on_exit`. A graph that leaves MC after a rejected trial is then re-primed before its first MD update, with no extra model call. After an accepted trial, its forces carry over.
- Docs:
  - Monte Carlo section in `docs/userguide/dynamics_simulations.md`, including UMA settings for MC.
  - API page `docs/modules/mc.rst`.
  - CHANGELOG entry under Unreleased / Added.
- Agent skill:
  - Routing row and version check in `skills/nvalchemi-toolkit/SKILL.md`.
  - Monte Carlo section in `references/dynamics-optimization.md` (the note stays under 500 lines).
  - UMA inference-settings bullet in `references/models.md`.

## Testing

- [x] Unit tests pass on CUDA: `test/mc test/dynamics test/hooks --slow` all pass on a GPU node, including the CUDA-only checks that MC steps make no device-to-host synchronization, and the `FusedStage` stream tests.
- [x] Unit tests pass on CPU: `test/mc --slow` (68 passed; the 4 CUDA-only sync checks skip), including regression tests for per-graph masses, restarts into fresh and advanced samplers (same species, masses and `stats` as an uninterrupted run, and a `save_checkpoint`/`load_checkpoint` round trip), and `Kawasaki.refresh_energy`.
- [x] Linting passes (pre-commit: ruff, ruff format, interrogate, markdownlint, license)
- [x] New tests added for new functionality meets coverage expectations?

## Checklist

- [x] I have read and understand the [Contributing Guidelines](../CONTRIBUTING.md)
- [x] I have updated the [CHANGELOG.md](../CHANGELOG.md)
- [x] I have performed a self-review of my code
- [x] I have added docstrings to new functions/classes
- [x] I have updated the documentation (if applicable)

## Additional Notes

- The skill edits change `skills/nvalchemi-toolkit/`, so `skill.oms.sig` needs re-signing on your side. New links point at `main` instead of `v0.3.0`, because the pages do not exist at that tag.
- A follow-up PR, hybrid MC-MD with scheduling, builds on this one.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01HBtAWFoh4ViZnY6avLBFhg
