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
* `DELTA_MU_EV`: -1.0 -> 1.0 eV in 0.2 eV steps (11 points). See
  "Chemical-potential calibration" below for what this actually means.
* Sizes: 500 / 1372 / 2048 atoms — conventional-cubic Au fcc supercells
  (5x5x5 / 7x7x7 / 8x8x8 conventional cells, 4 atoms/cell).

8 x 11 = 88 state points per size, 264 total. Each size runs as an
independent process/GPU/checkpoint directory — they never share a batch.

## Chemical-potential calibration

`nvalchemi.mc.SGC`'s `chemical_potentials` is a literal, absolute per-atom
energy, not a relative bias — `{Au: 0.0, Pt: delta_mu}` does not mean "Au
and Pt are equally favorable," it means "whatever this checkpoint's own raw
energy convention already encodes, uncorrected." That raw offset is 1-3
eV/atom on this UMA checkpoint, large enough to swamp the +-1.0 eV
`DELTA_MU_EV` sweep and drive every run to one pure phase regardless of
`delta_mu` (the sibling repo's `scout_sgc_temperature_composition_drift.py`
job 5484379 is a documented example of exactly this failure, at
`delta_mu=0.0`).

Pass `--reference-energies-json <reference_energy_calibration.py output>`
(recommended; run that script first — see the companion
`nvalchemi-toolkit-quest-deploy` repo's `phase_diagram_guide.md` section 3
and `hpc/quest/tests/reference_energy_calibration.py`) to fix this: every
run's `chemical_potentials_ev` is rebuilt as `{Au: 0.0, Pt:
reference[T]["delta_mu_ref_eV"] + delta_mu_excess}` at that run's own
temperature, with `DELTA_MU_EV`'s per-column value reinterpreted as
`delta_mu_excess`. `REFERENCE_ENERGIES_JSON` (env var) plumbs this through
`submit_campaign.slurm`; `ALLOW_UNRESOLVED_REFERENCE=1` bypasses an
unresolved `equilibration_gate` (not recommended). Omitting it keeps the
old literal, uncalibrated behavior (a warning is printed at startup) — fine
for a pure throughput/memory benchmark, not for drawing conclusions about
the real Au-Pt phase boundary.

## Continuation

Each `delta_mu` column is one `CampaignSpec.cooling_from_reference` chain: a
3000 K reference run seeds seven cooling children down to 1600 K via
`RunSpec.parent_id`. 200 K is a coarse step, so a cooled child is not already
equilibrated at its new set point, but it inherits a plausible composition and
a lattice constant close to the target — enough to shorten re-equilibration
relative to a random start. Reference runs get the full block budget
(`N_BLOCKS_REFERENCE = 200`); continuation children get a reduced budget
(`N_BLOCKS_CONTINUATION = 100`). Set `USE_CONTINUATION = False` in
`run_campaign.py` for a flat, independent grid instead (every state point
random-started, `N_BLOCKS_REFERENCE` each — a clean per-state-point
comparison at roughly double the total block count).

## Hybrid block

50 MD steps at `dt=3 fs`, then `round(0.2 * n_atoms)` MC trials (one
attempted transmutation per graph per MC step): 100 MC trials/block at 500
atoms, 274 at 1372, 410 at 2048.

## Equilibration gate

Equilibration used to be *assumed* within the first `EQUILIBRATION_BLOCKS`
(50) and never checked. Every block's per-graph Pt fraction and
energy/atom are now recorded during the run, and afterward
`_equilibration_gate` (PHASE_DIAGRAM_MANUAL.md section 7, the same
one-shot pattern `reference_energy_calibration.py` already uses) compares
the last two `EQUILIBRATION_WINDOW_BLOCKS`-block (25) windows of both
series against twice their combined standard error. This is a single
end-of-run check, not the manual's full three-consecutive-checks
promotion protocol — a run that fails it is logged as unresolved, not
retried or extended automatically. Recording adds one small GPU->CPU
transfer per block; see `_run_hybrid_with_observables`'s docstring for why
that's the cheaper option relative to the alternative of calling
`hybrid.run(batch, n_blocks=1)` in a loop.

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
hf auth login   # once, for the gated UMA checkpoint

uv run python benchmark/hybrid_sgc_npt/run_campaign.py \
    --n-atoms 500 \
    --checkpoint-root benchmark/hybrid_sgc_npt/checkpoints \
    --device cuda \
    --reference-energies-json \
        /path/to/reference_energy_calibration/<job_id>/run/reference_energies.json
```

Drop the `--reference-energies-json` line for the old literal, uncalibrated
behavior (a startup warning is printed) -- fine for a pure throughput/memory
run, not for real phase-diagram production.

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
completed run), `checkpoints/atoms<N>/<run_id>.equilibration.json` (both
gates' window means, difference, and standard error — see "Equilibration
gate" above), and appends to `checkpoints/atoms<N>_throughput.csv`
(`run_id, n_atoms, batch_width, n_blocks, wall_seconds,
walker_blocks_per_second, mc_acceptance, continuation,
composition_gate_resolved, energy_gate_resolved, resolved`) — the
efficiency record for this benchmark. A `checkpoints/atoms<N>_throughput.csv`
left over from before this fix has the old 8-column header; new rows
appended to it will have 11 columns instead (no migration -- per the
companion repo's `phase_diagram_guide.md` status table, this campaign has
never actually been run yet) -- delete it and let the header be rewritten
if you hit this.
