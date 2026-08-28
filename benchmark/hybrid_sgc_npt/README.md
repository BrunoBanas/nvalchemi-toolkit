# Hybrid SGC-NPT efficiency campaign

Throughput benchmark for `HybridMCMD` (batched `SGC` + `NPT`) across a coarse
Au-Pt temperature / chemical-potential grid, at three system sizes, on a
single 40 GB A100 GPU per size.

| File | Role |
| --- | --- |
| `run_campaign.py` | The app: builds the grid, profiles a batch width, runs the campaign, logs throughput. |
| `submit_campaign.slurm` | The input script: a 3-task SLURM array job, one task per size. |

## Grid

* Temperature: 3000 K -> 1600 K in 200 K steps (8 points).
* `delta_mu = mu(Pt) - mu(Au)`: -1.0 -> 1.0 eV in 0.2 eV steps (11 points).
* Sizes: 500 / 1372 / 2048 atoms — conventional-cubic Au fcc supercells
  (5x5x5 / 7x7x7 / 8x8x8 conventional cells, 4 atoms/cell).

8 x 11 = 88 state points per size, 264 total. Each size runs as an
independent process/GPU/checkpoint directory — they never share a batch.

## Continuation

Each `delta_mu` column is one `CampaignSpec.cooling_from_reference` chain: a
3000 K reference run seeds seven cooling children down to 1600 K via
`RunSpec.parent_id`. 200 K is a coarse step, so a cooled child is not already
equilibrated at its new set point, but it inherits a plausible composition and
a lattice constant close to the target — enough to shorten re-equilibration
relative to a random start. Reference runs get the full block budget
(`N_BLOCKS_REFERENCE = 200`); continuation children get a reduced budget
(`N_BLOCKS_CONTINUATION = 100`) that keeps the same ~50-block equilibration
window (`EQUILIBRATION_BLOCKS`). Set `USE_CONTINUATION = False` in
`run_campaign.py` for a flat, independent grid instead (every state point
random-started, `N_BLOCKS_REFERENCE` each — a clean per-state-point
comparison at roughly double the total block count).

## Hybrid block

50 MD steps at `dt=3 fs`, then `round(0.2 * n_atoms)` MC trials (one
attempted transmutation per graph per MC step): 100 MC trials/block at 500
atoms, 274 at 1372, 410 at 2048.

## Batch width

`SimulationBatchPlanner` profiles the reference-row workload at candidate
widths on the actual GPU and recommends the smallest width within 95% of peak
throughput while reserving <= 85% of device memory
(`BATCH_MEMORY_FRACTION` / `BATCH_THROUGHPUT_FRACTION`). A 500-atom walker is
expected to reserve about 3.5 GB, so on a 40 GB A100 the recommended width for
that size should land around 10. Before trusting the recommendation,
`_verify_memory_floor` fits a linear memory model from the profile
(`SimulationBatchPlanner.infer_memory_model`) and raises if the fitted
per-walker cost falls below 75% of that (atom-count-scaled) 3.5 GB
expectation — a profiler that under-counts memory would otherwise pick an
unsafely large width for a multi-day unattended run. Pass `--batch-width` to
skip profiling and set a value by hand (e.g. a previously benchmarked width).

## Running

Interactively, one size at a time:

```bash
UV_PROJECT_ENVIRONMENT=.venv-uma uv sync --extra uma --extra ase
huggingface-cli login   # once, for the gated UMA checkpoint

uv run python benchmark/hybrid_sgc_npt/run_campaign.py \
    --n-atoms 500 \
    --checkpoint-root benchmark/hybrid_sgc_npt/checkpoints \
    --device cuda
```

On a cluster:

```bash
sbatch benchmark/hybrid_sgc_npt/submit_campaign.slurm
```

Edit the `#SBATCH` directives (partition, account, walltime) and the
"cluster-specific setup" block in `submit_campaign.slurm` for your site
first. The array has 3 tasks (`--array=0-2`), mapping to the 500 / 1372 /
2048-atom sizes; each task requests one GPU.

### Resuming

`CampaignScheduler` reconstructs completed run IDs from the
`FinalStateStore` checkpoint directory (`checkpoints/atoms<N>/*.pt`) on
startup, so re-running the same command — after `--requeue`, a walltime
timeout, or a manual resubmission — skips finished state points and resumes
from the last completed run in each chain. A run that was killed mid-block is
simply redone from its parent's checkpoint; there is no partial-run recovery
below the granularity of one campaign node.

## Output

Each size writes `checkpoints/atoms<N>/<run_id>.pt` (final atomic state per
completed run) and appends to `checkpoints/atoms<N>_throughput.csv`
(`run_id, n_atoms, batch_width, n_blocks, wall_seconds,
walker_blocks_per_second, mc_acceptance, continuation`) — the efficiency
record for this benchmark.
