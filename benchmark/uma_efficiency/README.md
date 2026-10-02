# UMA efficiency: planner, calibration and benchmarks

Tools for choosing UMA inference settings and batch widths for nvalchemi Monte
Carlo, MD and hybrid MC-MD runs, and the benchmarks that calibrate them. The
`nvalchemi-uma-submission` agent skill (`.claude/skills/`) drives these files.

| File | Role |
| --- | --- |
| `plan_run.py` | Planner: valid settings, memory and time per batch width, recommended width, job count, time limit. |
| `calibration.json` | Measured cost and memory coefficients the planner reads (derivation below). |
| `driver_template.py` | Adaptable driver for Kawasaki / SGC / VC-SGC, NPT / NVT and their hybrids. |
| `job_template.sbatch` | Slurm job template: GPU guard, memory monitor, time record, USR1 forwarding. |
| `run_efficiency_matrix.sh` | Runs benchmark kernels (settings x widths) back to back in one GPU allocation. |
| `benchmark_batched_pure_kawasaki.py` | Batched Kawasaki MC only. |
| `benchmark_batched_pure_sgc.py` | Batched SGC MC only. |
| `benchmark_hybrid_sgc_npt_single_point.py` | SGC-NPT / Kawasaki-NPT hybrid, or `--md-only` NPT, with per-phase timing. |
| `benchmark_npt_md_only_single_point.py` | Plain NPT; also the seeded-composition helper the MC benchmarks share. |
| `memory_profile_matrix.py` | Peak memory vs batch width across kernels and system sizes. |
| `profile_mole_overhead.py` | Where unmerged mixture-of-experts (MoLE) time goes, with cached-piece ablations. |
| `check_mole_layer_type.py` | Which MoLE implementation (Python loop or fused) a checkpoint uses. |
| `_uma_inference.py`, `_run_diagnostics.py` | Shared helpers: settings specs; crash-tolerant progress files. |

The benchmarks build their Au-Pt cells with `../hybrid_sgc_npt/run_campaign.py`
(sizes 108, 256, 500, 1372, 2048, 4000 atoms) and write `metrics.json` plus
`progress.jsonl` / `partial_metrics.json`, which survive a time-limit kill.

## Quick start

```bash
python benchmark/uma_efficiency/plan_run.py --kind sgc-npt --n-atoms 500 \
    --walkers 8 --gpu a100-sxm4-80gb
KERNELS="kawasaki sgc hybrid" EXPECT_GPU=A100-SXM4-80GB \
    bash benchmark/uma_efficiency/run_efficiency_matrix.sh
```

## Reference system

All coefficients come from `uma-s-1p2`, task `omat`, fairchem-core 2.22, on an
Au-Pt fcc cell (a = 4.00 A, 500 atoms, 54 neighbours per atom within UMA's 6 A
cutoff) at 1200 K. One A100-SXM4-80GB allocation per kernel, so node-to-node
variation (~5%) stays out of each comparison. Kawasaki MC-only: 100 blocks x
100 steps; SGC MC-only and hybrids: 20 blocks of 100 MC + 50 NPT steps
(dt 3 fs). Memory is peak `torch.cuda.max_memory_reserved`.

## Measurements

Each cell: s/block, peak GiB reserved, wall s for 4 walkers ((4 / width)
sequential runs), speedup over the first row at width 1.

Kawasaki MC-only:

| Settings | Width 1 | Width 2 | Width 4 |
| --- | --- | --- | --- |
| turbo, species-blind, full forces | 4.27 / 2.9 / 1992 / 1.00 | 6.84 / 5.7 / 1506 / 1.32 | 11.81 / 11.2 / 1251 / 1.59 |
| turbo, unlike pairs, energy-only | 2.57 / 2.3 / 1304 / 1.53 | 3.68 / 4.6 / 863 / 2.31 | 5.83 / 9.0 / 648 / 3.07 |
| merge, no compile, energy-only | 2.63 / 3.7 / 1286 / 1.55 | 4.08 / 7.2 / 924 / 2.16 | 6.78 / 14.4 / 737 / 2.70 |

SGC MC-only:

| Settings | Width 1 | Width 2 | Width 4 |
| --- | --- | --- | --- |
| `batch` preset | 18.08 / 3.2 / 1681 / 1.00 | 33.85 / 5.2 / 1474 / 1.14 | 67.25 / 9.2 / 1404 / 1.20 |
| no merge/compile/ckpt, energy-only | 7.35 / 2.2 / 883 / 1.90 | 13.08 / 3.2 / 568 / 2.96 | 24.49 / 5.0 / 513 / 3.27 |

SGC-NPT hybrid:

| Settings | Width 1 | Width 2 | Width 4 |
| --- | --- | --- | --- |
| `batch`, full outputs | 27.26 / 3.9 / 2299 / 1.00 | 50.91 / 6.8 / 2162 / 1.06 | 101.1 / 17.6 / 2044 / 1.12 |
| no merge/compile/ckpt, energy-only MC | 12.61 / 9.0 / 1254 / 1.83 | 22.38 / 17.8 / 990 / 2.32 | 42.72 / 48.7 / 913 / 2.52 |

Kawasaki-NPT hybrid:

| Settings | Width 1 | Width 2 | Width 4 |
| --- | --- | --- | --- |
| turbo, species-blind, full outputs | 79.28 / 4.0 / 6633 / 1.00 | 84.75 / 7.9 / 3520 / 1.88 | 95.68 / 15.6 / 1988 / 3.34 |
| merge, no compile, energy-only MC | 5.95 / 4.1 / 708 / 9.37 | 8.67 / 8.0 / 390 / 17.0 | 14.57 / 16.0 / 314 / 21.1 |
| no merge, no compile, energy-only MC | 12.35 / 6.9 / 1222 / 5.43 | 21.72 / 12.4 / 984 / 6.74 | 41.30 / 23.5 / 887 / 7.48 |
| turbo, energy-only MC | 72.08 / 4.0 / 6042 / 1.10 | 75.41 / 7.9 / 3149 / 2.11 | 79.21 / 20.1 / 1648 / 4.02 |

Pure NPT (`--md-only`, median s/block):

| Settings | Width 1 | Width 2 | Width 4 |
| --- | --- | --- | --- |
| `batch` preset | 8.66 / 3.2 / 993 / 1.00 | 16.12 / 5.2 / 803 / 1.24 | 31.91 / 9.2 / 725 / 1.37 |
| merge, no compile | 2.65 / 4.1 / 502 / 1.98 | 4.30 / 8.0 / 253 / 3.93 | 7.55 / 16.0 / 186 / 5.35 |
| no merge, no compile | 4.74 / 6.9 / 668 / 1.49 | 8.49 / 12.4 / 424 / 2.35 | 16.82 / 23.5 / 409 / 2.43 |
| merge, no compile, checkpointing | 5.32 / 2.0 / 714 / 1.39 | 9.53 / 4.1 / 500 / 1.99 | 17.82 / 7.5 / 434 / 2.29 |
| turbo | 2.65 / 4.1 / 5370 / 0.19 | 4.28 / 7.8 / 2776 / 0.36 | 7.57 / 15.5 / 1465 / 0.68 |

Under MD, turbo recompiles on every change of the graph's edge count (32
recompiles, 1146-1181 s in the first block) and then runs at merge-no-compile
speed, so compile never pays when MD is involved.

## Model

Per batched model call on the reference card:

```text
time   t(w) = c1 * (N0 + w * N_eff)                   [ms]
memory M(w) = base + w * per_walker * (N_eff / 500)   [GiB]
N_eff  = N * (neighbours per atom within the cutoff) / 54
```

`c1` and `N0` come from the width-1 and width-4 medians of each phase, per
settings class and output set (`mc_energy_only`, `mc_full_outputs`,
`md_full_outputs`). `N0`, the fixed per-call overhead in atom-equivalents, is
what batching amortizes: large for merged / compiled paths (300-680), small for
unmerged ones (60-140). Hybrid blocks add one MD evaluation (the energy
refresh at each MC block) and each job ~75 s of model loading. UMA's cost
follows the edge count, not element identity, which is why the system enters
through `N_eff`. SGC-NPT memory grows faster than linear up to width 4
(9.0 / 17.8 / 48.7 GiB), so its whole curve is scaled with `N_eff`.

Validation: the model reproduces all 18 measured 500-atom cells within 4% in
time (width 2 was not fitted). Memory is exact at widths 1 and 4 and within 2%
at width 2, except SGC-NPT (+25%, conservative). SGC-NPT at 2048 atoms with
checkpointing on an A100-PCIe-40GB: 18.0 GiB predicted, 18.84 GiB measured.

Size scaling (`memory_profile_matrix.py`, width 1, 500-2048 atoms):
checkpointed runs grow linearly (1.03 GB + 4.28 MB/atom); compiled full-force
runs rise from 5.6 MB/atom at 500 atoms to 7.9-8.0 MB/atom at 1372-2048, with
~1.4x higher per-atom time. The planner applies this as a ramp to 1.45x from 500
to 1372 atoms. Nothing above 2048 atoms is measured.

GPU factors: A100-PCIe-40GB ~1.10x slower than SXM; H100 ~2.1-2.4x faster on
every workload measured (factor 0.45, from a few paired runs).

## Findings behind the rules

- compile: valid only when neither geometry nor composition changes (MC-only
  Kawasaki). Any MD changes the edge count and forces recompiles.
- merge_mole: folds the 32 experts into one plain model for one composition;
  fairchem asserts that every graph in a batch shares it. Valid for Kawasaki
  and MD with identical walker compositions; never for SGC / VC-SGC (fairchem
  2.21 asserts, 2.22 falls back to unmerged after the first change).
- activation checkpointing: ~2x slower MD, ~2-3x less memory per walker;
  only when nothing faster fits. Energy-only MC has no backward to checkpoint.
- tf32: identical accept/reject sequences, ueV/atom energy differences.
- energy-only MC: SGC 1.3-1.44x, Kawasaki (turbo) 2.1x per MC step.
- Kawasaki unlike-pair proposals: removes the ~50% of draws that swapped
  identical species; acceptance per real swap unchanged (~0.7 at 1200 K).
- Hybrid MC-MD blocks cost within ~3% of the same MC and MD run separately
  (width 1 Kawasaki-NPT: ~9%).
- Unmerged MoLE costs ~2.5x merged per atom in MD and ~4x in energy-only MC,
  although with fixed coefficients it is the same linear algebra;
  `profile_mole_overhead.py` measures which per-call piece is responsible.
  `HybridMCMD` accepts separate MC and MD models plus a `before_md_block`
  callback, the hook for a per-block re-merged MD model.

Known gaps: widths above 4 and sizes other than 500 atoms are extrapolated
(run the `*_wide`, `sgc_npt_108/256` and `*_2048` kernels); NVT is assumed to cost
the same as NPT; VC-SGC the same as SGC; checkpointed energy-only MC the same as
unmerged; only `uma-s-1p2` / `omat` is calibrated.

## Recalibrating

1. Run the kernels you need on one pinned card, for example
   `KERNELS="kawasaki sgc hybrid kawasaki_npt npt" EXPECT_GPU=A100-SXM4-80GB`
   with `run_efficiency_matrix.sh`.
2. From each case's `run/metrics.json` take the MC and MD block times (median
   `phase_timing` for hybrids, run wall / blocks for MC-only) and
   `peak_gpu_memory_reserved_GB`.
3. Per settings class, with t in ms per step and N = 500:
   `c1 = (t4 - t1) / (3 N)`, `N0 = t1 / c1 - N`; `per_walker = (M4 - M1) / 3`,
   `base = M1 - per_walker`.
4. Update `calibration.json` (keep a `source` string per entry) and replay the
   measured cells with `plan_run.py --settings <class> --max-width 4`.
