---
name: nvalchemi-uma-submission
description: >-
  How to choose FairChem UMA inference settings, batch width and GPU memory for
  nvalchemi Monte Carlo (Kawasaki, SGC, VC-SGC), MD (NPT, NVT) and hybrid MC-MD
  runs, and write the Slurm job. Use when setting up, sizing or speeding up a
  UMA simulation, asking how many walkers fit on a GPU, which inference settings
  or batch width to use, how much memory or wall time a run needs, or writing an
  sbatch script for MC, MD or HybridMCMD.
---

# UMA run planning and submission

## Overview

UMA exposes four inference knobs (`compile`, `merge_mole`, `tf32`,
`activation_checkpointing`); nvalchemi adds energy-only evaluation for Monte Carlo
and batching of independent walkers into one model call. Which combination is
fastest depends on whether the run changes composition, changes geometry, runs one
system or many, and fits the card. This skill turns a physics request into a plan
and a job script using coefficients measured on real runs:

- `benchmark/uma_efficiency/plan_run.py` ranks every valid settings class and
  returns memory, time per block, the recommended batch width, the number of GPU
  jobs and a time limit, for any system size, elements or structure.
- `benchmark/uma_efficiency/calibration.json` holds the measured coefficients;
  `benchmark/uma_efficiency/README.md` has the data, the model and how to refit it.
- `benchmark/uma_efficiency/driver_template.py` and
  `benchmark/uma_efficiency/job_template.sbatch` are the starting files.

Background: "UMA settings for SGC and SGC-NPT" and "UMA settings for
fixed-composition runs" in `docs/userguide/dynamics_simulations.md`. For the
dynamics and MC classes themselves see `nvalchemi-dynamics-api`; for the model
wrapper see `nvalchemi-model-wrapping`.

## Workflow

1. Classify the run (next section): sampler/ensemble, structure or size, state
   points and replicas, blocks and steps per block, and the purpose -- production
   / physics, or an efficiency comparison. Purpose changes the GPU request.
2. Decide the regime: throughput (many independent walkers to batch) or single
   system (one walker, or too large to batch). It changes which settings win.
3. Run the planner with the structure file when there is one.
4. Pick the entry point: an existing campaign script if one fits, else the driver
   template.
5. Fill `job_template.sbatch`, putting the planner summary in its header.
6. Check the pre-submit list, smoke-test with 2 blocks, then submit.

## Classify the run

| `--kind` | What moves | Composition | Geometry | Valid settings classes |
|---|---|---|---|---|
| `kawasaki` | species swaps | fixed | fixed | turbo, merged, unmerged |
| `sgc`, `vcsgc` | transmutations | changes | fixed | unmerged |
| `npt`, `nvt` | positions (+ cell) | fixed | changes | merged, unmerged, checkpointed variants |
| `kawasaki-npt`, `kawasaki-nvt` | swaps + MD | fixed | changes | merged, unmerged, checkpointed variants |
| `sgc-npt`, `vcsgc-npt`, `-nvt` | transmutations + MD | changes | changes | unmerged, checkpointed unmerged |

Every kind with MC runs its MC energy-only; MD always needs forces (and stress for
NPT). "Merged" classes drop out whenever batched walkers differ in composition.

## Why these settings

Each knob pays under exactly one condition.

- `compile` only when both geometry and composition are fixed (MC-only
  Kawasaki). torch.compile specializes on graph shape; MD changes the edge count
  almost every step, which caused 32 recompiles (~20 min) and then a fallback to
  uncompiled speed. In a Kawasaki-NPT hybrid, turbo lost to merge-without-compile
  at every width.
- `merge_mole` only when composition is fixed. Merging folds UMA's 32 experts
  into one plain model for one composition, and fairchem asserts that every graph
  in a batch has that reduced composition. Batch with merge only when all walkers
  share a composition (replicas of one structure). Turn merge off when different
  systems share a batch or a model -- other compositions, shapes or sizes, or
  several systems run through one model in turn -- or load a new `UMAWrapper`
  per system. Never for SGC / VC-SGC: fairchem 2.21 asserts, 2.22 silently falls
  back to the slower unmerged path. When valid it is the largest single win:
  pure NPT 1.8x (width 1) to 2.2x (width 4) faster than unmerged.
- `activation_checkpointing` only when nothing faster fits: about 2x slower per
  MD step, 2-3x less memory per walker. Energy-only MC has no backward pass, so it
  gains nothing there. With merge available, merge + checkpointing is the better
  fallback (pure NPT 2x faster than the `"batch"` preset at 0.6x its memory).
- `tf32` on: identical accept/reject sequences, ueV/atom energy differences.
- Energy-only MC skips the forces/stress backward pass: SGC 1.3-1.44x, Kawasaki
  (turbo) 2.1x per MC step. MC-only runs set it on the model; hybrids pass
  `mc_energy_only=True` and keep full outputs for MD.

Settings classes and the spec strings `UMAWrapper.from_checkpoint` accepts:

| Class | `inference_settings` |
|---|---|
| `compiled_merged` | `"turbo"` |
| `eager_merged` | `"compile=false,merge_mole=true,tf32=true,activation_checkpointing=false"` |
| `eager_unmerged` | `"compile=false,merge_mole=false,tf32=true,activation_checkpointing=false"` |
| `checkpointed_merged` | `"compile=false,merge_mole=true,tf32=true,activation_checkpointing=true"` |
| `checkpointed_unmerged` | `"compile=false,merge_mole=false,tf32=true,activation_checkpointing=true"` |

```python
from nvalchemi.hybrid import HybridMCMD
from nvalchemi.models.uma import UMAWrapper

spec = "compile=false,merge_mole=false,tf32=true,activation_checkpointing=false"
model = UMAWrapper.from_checkpoint(
    "uma-s-1p2", task_name="omat", device="cuda", inference_settings=spec
)
model.model_config.active_outputs = {"energy"}  # MC-only runs
# Hybrids instead: HybridMCMD(mc=mc, md=npt, mc_steps=100, md_steps=50,
#                             mc_energy_only=True)
```

MC and MD normally share one model, so its settings must be valid for both phases:
compile is out with MD, merge is out if either phase changes composition.
`HybridMCMD` also accepts two models of the same checkpoint and task that differ
only in settings; each MC block then re-evaluates its energy with the MC model,
and `before_md_block(batch)` runs before each MD block's first force call.

## Throughput vs single system

- Throughput regime: many small, independent walkers (a T / Δμ grid, replicas, a
  composition scan) and the goal is the most walker-blocks per GPU-hour. Settings
  are ranked by walker-blocks/s at each setting's own best width, so a setting
  with a large per-call overhead that batching amortizes can win even if it is
  not the fastest for one walker.
- Single-system regime: one system studied, or one too large for two walkers to
  fit. Width 1, ranked by time per block. Merge is valid whenever composition is
  fixed. Many such systems means more GPUs in parallel, not a wider batch.

`--mode auto` (default) picks single for `--walkers 1` and throughput otherwise,
and reports the regime it ended in, including "too large to batch on this card".
Measured (500-atom Au-Pt, A100-SXM4-80GB):

| Run | Throughput regime | Single system |
|---|---|---|
| Kawasaki MC | turbo (0.85 vs 0.69 walker-blocks/s for merge, no compile) | turbo ≈ merge, no compile (2.57 vs 2.63 s/block) |
| Kawasaki-NPT, same composition | merge, no compile, width 4-6 | merge, no compile |
| Kawasaki-NPT, mixed compositions | no merge, no compile | merge, no compile (model per system) |
| SGC, SGC-NPT | no merge, no compile, width 2-5 (1.1-1.2x); wider for smaller cells | no merge, no compile; checkpointing if it does not fit |
| Width 2 does not fit | - | fastest class that fits at width 1 |

## Planning

```bash
python benchmark/uma_efficiency/plan_run.py --kind sgc-npt \
    --structure start.xyz --walkers 20 --n-blocks 200
python benchmark/uma_efficiency/plan_run.py --kind kawasaki-npt \
    --n-atoms 500 --walkers 8 --mixed-compositions --gpu a100-pcie-40gb
```

Flags: `--n-atoms` (without a structure file, a dense fcc-like metal is
assumed), `--density` or `--mean-neighbors` for sparse or dense systems,
`--walkers` and `--mode`, `--gpu` (`any` [default], `a100-sxm4-80gb`,
`a100-pcie-40gb`, `h100-80gb`), `--mc-steps` (default 0.2 N per block),
`--md-steps` (default 50), `--mixed-compositions`, `--settings` (force a class;
the planner warns when it is invalid), `--json`.

How it estimates. UMA's cost follows graph edges, not element identity, so a
system enters through `N_eff = N * (neighbours per atom within 6 A) / 54` (54 for
the fcc Au-Pt calibration cell). A nanoparticle in vacuum costs less per atom, a
dense oxide more. Time per batched call is `c1 * (N0 + width * N_eff)`, where
`N0` is the per-call overhead batching amortizes (300-680 atom-equivalents for
merged paths, 60-140 unmerged); memory is `base + width * per_walker *
N_eff / 500`, limited to 85% of the card, with a measured 1.45x ramp between 500
and 1372 atoms.

How it chooses. Every valid class that fits is compared; the highest
walker-blocks/s wins in the throughput regime, the lowest s/block at width 1 in
the single regime, with classes within 3% treated as tied. The recommended width
is the smallest reaching 97% of the best feasible throughput: SGC-type runs gain
only 10-20% from batching at 500 atoms, and a looser rule would discard it. The
`gain` column is throughput relative to width 1; if it stays near 1.0x, spend
GPUs, not width.

Walkers, width and jobs. `--walkers` is the total number of chains (state points
x replicas); one job runs `width` of them; `waves` is the number of jobs. Walkers
in one batch may have different T and Δμ (the driver takes one value per walker)
but share the structure.

Card choice. A job that may land on any card is sized for the smallest (40 GB)
and timed for the slowest (A100-PCIe, ~1.10x slower than SXM; H100 is 2.1-2.4x
faster). If the fast settings do not fit 40 GB at width 1, the planner switches to
checkpointing; an 80 GB card would keep the faster settings.

Trust. Time is within 4% at 500 atoms for widths 1-4 (18 measured cells). Widths
above 4, sizes far from 500 atoms (N < 200, N > 1250, or a neighbour ratio off by
more than 50%) and SGC-NPT memory above width 4 are extrapolated (about ±30%): say
so, and propose a short width profile (`run_efficiency_matrix.sh`) before a long
campaign. NVT is assumed to cost the same as NPT.

## Entry points

- Au-Pt SGC-NPT phase-diagram campaigns (Δμ scans, boundary tracing, reference
  energies): `benchmark/hybrid_sgc_npt/run_campaign.py`. Its defaults are the
  measured best (unmerged, energy-only MC) and it profiles the width itself;
  write a job around it rather than a new driver. Analysis of its output:
  `nvalchemi-sgc-phase-boundary`.
- Efficiency questions: `benchmark/uma_efficiency/run_efficiency_matrix.sh`
  (kernels such as `kawasaki`, `sgc`, `hybrid`, `kawasaki_npt`, `npt`, the
  `*_wide` width sweeps and `*_2048` size checks) or the individual
  `benchmark/uma_efficiency/benchmark_*.py` scripts with `--inference-settings`,
  `--n-walkers` / `--batch-width` and `--mc-energy-only` / `--energy-only`.
- Anything else (other elements, nanoparticles, VC-SGC, custom observables): copy
  `benchmark/uma_efficiency/driver_template.py` and adapt its marked sections. It
  already sets masses from species, per-walker seeds / T / Δμ, the initial cell
  wrap and per-step `WrapPeriodicHook` for MD, unlike-pair Kawasaki, energy-only
  MC, a VC-SGC reference-potential guard, per-walker final structures and a
  `metrics.json` with peak memory and the card name.

```bash
python my_driver.py --kind kawasaki-npt --structure start.xyz --width 4 \
    --inference-settings \
    "compile=false,merge_mole=true,tf32=true,activation_checkpointing=false" \
    --temperature-k 800 900 1000 1100 --n-blocks 2 --output smoke/metrics.json
```

## Writing the job

Fill `benchmark/uma_efficiency/job_template.sbatch`: `__JOB_NAME__`,
`__CAMPAIGN__`, `__DRIVER__` and `__DRIVER_ARGS__` (with `--width` and
`--inference-settings` from the plan, and a distinct `--seed` per job),
`__TIME_LIMIT__` (the planner's suggested time) and the plan summary. Add the
site's `--account` / `--partition`. Keep the GPU guard, memory monitor,
`/usr/bin/time` record, USR1 forwarding and status file: they are what make
out-of-memory and time-limit failures diagnosable. For several jobs, submit the
template once per slice of state points.

GPU request:

- Production and physics runs: any GPU (`--gres=gpu:1`) for the shortest queue,
  with width sized for the smallest card and time for the slowest.
- Speed, memory, batching or settings comparisons: one card model only. Request
  it by type, exclude look-alikes (a generic "a100" type can match both 40 GB
  PCIe and 80 GB SXM cards) and set `EXPECT_GPU` so a misplaced job exits
  instead of producing a number that is silently not comparable.

## Pre-submit checklist

Physics:

- MD in a periodic cell has `WrapPeriodicHook(frequency=1,
  stage=DynamicsStage.AFTER_POST_UPDATE)` and an initial wrap before the first
  force call; otherwise diffusing atoms lose their minimum image and collapse.
- Isolated nanoparticles: MC or NVT in a vacuum box, with at least the 6 A cutoff
  plus a margin of vacuum. NPT on a vacuum-padded box rescales mostly vacuum.
  Give the planner the structure file so the lower neighbour count is used.
- Kawasaki keeps `unlike_pairs_only=True` (default); its Metropolis-Hastings
  correction for the changing pair count is built in.
- VC-SGC gets a calibrated `reference_exchange_potential`; UMA's per-element
  energies differ by eV and otherwise drive walkers to an end member.
- Custom hybrid loops call `hybrid.run_mc_block(batch)` (never `mc.run`) and, with
  a separate MD model, `hybrid.prepare_md_block(batch)` before `md.compute`.
  Samplers built on `BaseMonteCarlo` keep masses in step with species.
- Seeds differ per walker and per job.

Performance:

- Settings match the class table (no compile with MD, no merge when composition
  changes or batched walkers differ).
- Width x memory fits the smallest card the job can land on.
- The time limit is the planner's (1.5x the estimate on the slowest card, rounded
  up to 15 min); long campaigns get a 2-block smoke test first.
- One shared `HF_HOME` for all runs, so multi-GB checkpoints are not duplicated.

## Reporting the plan

State the regime; give a table of kind, N, N_eff, settings spec, width, GiB per
card, s/block, number of jobs and time limit; say which numbers are extrapolated;
list the files written and the submit command.

## Key files

- `benchmark/uma_efficiency/plan_run.py` -- planner.
- `benchmark/uma_efficiency/calibration.json` -- measured coefficients.
- `benchmark/uma_efficiency/README.md` -- measurements, model, recalibration.
- `benchmark/uma_efficiency/driver_template.py` -- adaptable driver.
- `benchmark/uma_efficiency/job_template.sbatch` -- Slurm job template.
- `benchmark/uma_efficiency/run_efficiency_matrix.sh` -- benchmark kernels.
- `benchmark/hybrid_sgc_npt/run_campaign.py` -- SGC-NPT phase-diagram campaign.
- `nvalchemi/models/uma.py` -- `UMAWrapper`, settings specs, derivative gating.
- `nvalchemi/hybrid/scheduler.py` -- `HybridMCMD`.
- `docs/userguide/dynamics_simulations.md` -- UMA settings sections.
