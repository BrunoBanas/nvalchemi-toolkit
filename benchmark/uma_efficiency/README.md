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
| `fit_calibration.py` | Fits `calibration.json` to efficiency-matrix results (time, memory, per class). |
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

### Follow-up: wider batches, other sizes, checkpointing, compile

Same card and kernels (`run_efficiency_matrix.sh`, `*_wide`, `sgc_npt_108/256`,
`*_2048`, `sgc_npt_ckpt`, `sgc_compile`). Seconds per block per walker (block /
width) and total peak GiB:

| Workload, settings | Width 1 | Width 4 | Width 8 | Width 12 (16) |
| --- | --- | --- | --- | --- |
| Kawasaki MC, turbo + energy-only | 2.57 / 2.3 | 1.49 / 9.0 | 1.27 / 17.9 | 1.21 / 26.8 |
| SGC MC, unmerged + energy-only | 7.35 / 2.2 | 6.07 / 5.0 | 5.90 / 8.9 | 5.85 / 12.7 |
| NPT, merge w/o compile | 2.85 / 4.1 | 2.07 / 16.0 | 1.87 / 31.6 | 1.81 / 47.4 |
| Kawasaki-NPT, merge w/o compile | 5.95 / 4.1 | 3.64 / 16.0 | 3.31 / 31.6 | 3.23 / 47.4 |
| SGC-NPT 108 atoms, unmerged | 4.12 / 3.0 | 1.79 / 8.0 | 1.67 / 16.4 | (1.83 / 50.1) |
| SGC-NPT 256 atoms, unmerged | 6.12 / 5.3 | 4.34 / 21.2 | 4.35 / 52.3 | - |
| SGC-NPT 500 atoms, unmerged + checkpointing | 14.2 / 3.9 | 12.0 / 17.6 | - | - |

2048 atoms, width 1 (Kawasaki MC also width 2), s/block and GiB:

| Workload | Fastest | Alternatives |
| --- | --- | --- |
| Kawasaki MC (410 trials) | merge w/o compile 28.2 / 14.6 | turbo 32.4 / 9.2; width 2 per walker: turbo 24.9, merge 25.4 |
| NPT (50 steps) | merge w/o compile 8.36 / 16.2 | no merge 15.7 / 24.7; merge + checkpointing 18.9 / 7.7 |
| Kawasaki-NPT | merge w/o compile 35.7 / 16.2 | merge + checkpointing 112 / 7.7 (MC phase 3.4x slower) |
| SGC-NPT | unmerged 117 / 35.8 | unmerged + checkpointing 122 / 13.1 |

SGC MC with compile (no merge, energy-only) is slower than without at both widths
(width 1: 9.72 vs 7.17 s/block, 2.9 vs 2.2 GiB).

## Model

Per batched model call on the reference card:

```text
time    t(w)  = c1 * (N0 + w * (N_eff * sf + Nw))              [ms]
memory  M(w)  = base + a * A + b * A * (w - 1),  A = w * N_eff / 500   [GiB]
N_eff   = N * (neighbours per atom within the cutoff) / 54
```

`N0` is the per-call overhead batching amortizes (in atom-equivalents), `Nw` a
per-walker overhead it does not (unmerged MoLE runs one small matmul per walker),
`sf` the measured large-graph slowdown of compiled runs (1.0 at 500 atoms to 1.45
at 1372 and above). Memory is linear in atom count; the cross term `b` is
non-zero only for SGC-NPT, whose memory grows faster than linear in the number of
walkers (2048 atoms x 1 walker: 35.8 GiB; 256 x 8: 52.3 GiB). Hybrid blocks add
one MD evaluation per block and each job ~75 s of model loading. UMA's cost
follows the edge count, not element identity, which is why the system enters
through `N_eff`. All coefficients are least-squares fits (relative error) by
`fit_calibration.py` over the 62 measured cells.

Validation, replaying all 62 cells (108-2048 atoms, widths 1-16): median error
1.6% in time and 0.5% in memory. Cells beyond 10%: SGC-NPT at 108-256 atoms
(time -20% to +10%, memory -14% to +19%; fixed per-call costs dominate small
cells) and SGC-NPT 500 atoms x 4 walkers (memory -21%, the jump from 17.8 to
48.7 GiB between widths 2 and 4). SGC-NPT therefore plans against 70% of the card
(`budget_scale`) instead of 85%.

GPU factors: A100-PCIe-40GB ~1.10x slower than SXM; H100 ~2.1-2.4x faster on
every workload measured (factor 0.45, from a few paired runs).

## Findings behind the rules

- compile: valid only when neither geometry nor composition changes (MC-only
  Kawasaki). Any MD changes the edge count and forces recompiles; for SGC MC it
  is slower than no compile.
- merge_mole: folds the 32 experts into one plain model for one composition;
  fairchem asserts that every graph in a batch shares it. Valid for Kawasaki
  and MD with identical walker compositions; never for SGC / VC-SGC (fairchem
  2.21 asserts, 2.22 falls back to unmerged after the first change). For a
  single 2048-atom Kawasaki system it beats turbo (28.2 vs 32.4 s/block).
- activation checkpointing: ~2x slower MD, ~2-3x less memory per walker; only
  when nothing faster fits. Unmerged energy-only MC is unaffected (SGC-NPT 2048
  atoms with checkpointing is only 4% slower), but with merge_mole it slows MC
  ~3.4x, so merge + checkpointing is a poor fallback for Kawasaki-NPT.
- tf32: identical accept/reject sequences, ueV/atom energy differences.
- energy-only MC: SGC 1.3-1.44x, Kawasaki (turbo) 2.1x per MC step.
- Kawasaki unlike-pair proposals: removes the ~50% of draws that swapped
  identical species; acceptance per real swap unchanged (~0.7 at 1200 K).
- Hybrid MC-MD blocks cost within ~3% of the same MC and MD run separately
  (width 1 Kawasaki-NPT: ~9%).
- Unmerged UMA is 2.7x (MD) to 3.5x (MC, width 4) slower than merged. The MoLE
  bookkeeping (expert-weight mixing, edge counts, coefficients) is under 1.5% of
  it and caching it gains nothing (`profile_mole_overhead.py`); the time is in
  the matmuls, which run as plain-FP32 kernels (`gemmSN`, `ampere_sgemm`) instead
  of the TF32 tensor-core GEMMs the merged model's plain linear layers get.
  Re-merging an MD model per block costs ~3.7 s per merge, more than it saves.
  Mixed walker compositions cost the same as identical ones.

Known gaps: SGC MC above 500 atoms, Kawasaki MC with unmerged settings, NVT
(assumed equal to NPT), VC-SGC (assumed equal to SGC) and H100 beyond a few
paired runs are not measured; widths beyond 12 at 500 atoms are extrapolated;
only `uma-s-1p2` / `omat` is calibrated.

## Recalibrating

1. Run the kernels you need on one pinned card, for example
   `KERNELS="kawasaki sgc hybrid kawasaki_npt npt" EXPECT_GPU=A100-SXM4-80GB`
   with `run_efficiency_matrix.sh`.
2. Fit and review: `python benchmark/uma_efficiency/fit_calibration.py
   <OUTPUT_ROOT>` prints every coefficient with its cell count and worst error.
3. Write them: add `--write` to update `calibration.json` (hand-set entries such
   as checkpointed merged MC and the SGC-NPT `budget_scale` are kept).
4. Replay the measured cells with `plan_run.py --settings <class>
   --max-width <width>` and check the recommendations.
