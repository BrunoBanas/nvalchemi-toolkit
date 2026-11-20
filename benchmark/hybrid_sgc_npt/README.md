# Hybrid SGC-NPT campaigns and phase-diagram drivers

Production drivers for Au-Pt phase diagrams with UMA: batched `SGC` / `VCSGC`
Monte Carlo alternated with `NPT` through `HybridMCMD`, scheduled with
`nvalchemi.scheduling`. Shared building blocks (constants, cell builders,
`make_workload`, observables and the equilibration gate, reference calibration,
the tracer engine) live in `campaign_lib.py`; every driver imports from there.
The `nvalchemi-sgc-phase-boundary` and `nvalchemi-uma-submission` agent skills
(`.claude/skills/`) explain when to use which.

| File | Role |
| --- | --- |
| `campaign_lib.py` | Shared library: system/protocol constants, `make_workload`, `run_hybrid_with_observables`, `equilibration_gate`, `compute_reference_energies`, `NvalchemiTraceEngine`. |
| `run_campaign.py` | CLI: coarse T x Δμ grid, Δμ hysteresis scans (`--mode delta-mu-scan`), boundary tracing (`--mode trace-boundary`). |
| `submit_campaign.slurm` | 3-task Slurm array for `run_campaign.py`, one task per size. |
| `run_mu_ladder.py` | Gated Δμ ladders warm-started from checkpoints (also rigid-lattice SGC with `md_steps_per_block: 0`). |
| `run_vcsgc_scan.py` | VC-SGC-NPT walkers at fixed target compositions inside a miscibility gap. |
| `run_phase_campaign.py` | One-GPU pool mixing ladders, fan-out points, VC-SGC walkers and boundary traces. |
| `boundary_tracer.py` | Engine-agnostic eq. 29 coexistence-line tracer (self-test in `../phase_boundary/`). |
| `vcsgc_analysis.py` | Δμ(c), g(c) and common tangent from VC-SGC series (numpy only). |
| `pure_free_energy.py` / `pure_free_energy_analysis.py` | Pure-element Gibbs free energies (lattice, Hessian, NPT ladder; then numpy-only analysis). |
| `uma_melting_coexistence.py` / `uma_melting_bisect.py` / `uma_melting_analysis.py` | Solid-liquid coexistence melting points (one run, bisection driver, analysis). |
| `export_structures.py` | Checkpoints (a directory or one `.pt`) to extended XYZ for OVITO / VMD / `ase gui`. |

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
`delta_mu` (a scouting run at `delta_mu=0.0` showed exactly this
failure).

Pass `--reference-energies-json <reference_energies.json>` (recommended; `--mode
delta-mu-scan` calibrates the reference in-process and writes
`auto_reference_energies.json`, see `compute_reference_energies` in
`campaign_lib.py`) to fix this: every run's `chemical_potentials_ev` is rebuilt
as `{Au: 0.0, Pt: reference[T]["delta_mu_ref_eV"] + delta_mu_excess}` at that
run's own temperature, with `DELTA_MU_EV`'s per-column value reinterpreted as
`delta_mu_excess`. `REFERENCE_ENERGIES_JSON` (env var) plumbs this through
`submit_campaign.slurm`; `ALLOW_UNRESOLVED_REFERENCE=1` bypasses an unresolved
`equilibration_gate` (not recommended). Omitting it keeps the old literal,
uncalibrated behavior (a warning is printed at startup) — fine for a pure
throughput/memory benchmark, not for drawing conclusions about the real Au-Pt
phase boundary.

In `--mode delta-mu-scan` the file is optional: temperatures it lacks are
calibrated in-process (pure Au and pure Pt NPT on the already-loaded model),
written to `<checkpoint-root>/atoms<N>/auto_reference_energies.json` for reuse,
and held to the same equilibration gate as file-supplied entries.

## Modes

* `--mode cooling` (default): the grid above, one cooling chain per Δμ column
  (see "Continuation").
* `--mode delta-mu-scan`: A-rich and B-rich endpoints at every
  `--scan-temperatures-k` (highest first), each seeding a Δμ_excess ladder that
  marches toward 0.0 (`CampaignSpec.delta_mu_scan_runs`). Each lower
  temperature's two endpoints continue from the previous temperature's
  endpoints (not from its whole ladder), so a lower temperature starts as soon
  as the higher one's seeds finish; `--scan-independent-temperatures` starts
  every temperature fresh instead. All temperatures go into one
  `CampaignSpec`, which must be a self-contained dependency graph, so run them
  in one invocation rather than chaining invocations by run id.
* `--mode trace-boundary`: integrate the coexistence line from one known point
  with `boundary_tracer.py` and `NvalchemiTraceEngine`.

## Δμ ladder spacing

Each branch's ladder is non-uniform: every new point sits
`--delta-mu-excess-refine-ratio` (default 0.5) of the way from the previous
point to 0.0 until the step would drop below `--delta-mu-excess-min-step-ev`
(default 0.01 eV), then lands on 0.0 (bracket 0.2: -0.2, -0.1, -0.05, -0.025,
-0.0125, 0.0). Coarse deep inside a phase, fine near the transition.
`--delta-mu-excess-overshoot-ev` continues each ladder past 0.0 (steps doubling)
so the two branches share a Δμ window; without that overlap no walker is ever
metastable where the other has data, so hysteresis, and with it a first-order
boundary, cannot be resolved.

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

Every block's per-graph Pt fraction and energy/atom are recorded through
`HybridMCMD.run(..., on_block=...)` (one small GPU->CPU transfer per block).
After the run, `_equilibration_gate` compares the last two
`EQUILIBRATION_WINDOW_BLOCKS`-block (25) windows of both series against twice
their combined standard error. It is a single end-of-run check: a run that
fails it is logged as unresolved, not retried or extended automatically
(`run_mu_ladder.py` repeats the gate until it passes).

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
        /path/to/auto_reference_energies.json
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
efficiency record for this benchmark.
