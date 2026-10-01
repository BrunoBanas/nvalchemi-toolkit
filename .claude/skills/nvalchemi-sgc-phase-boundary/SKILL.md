---
name: nvalchemi-sgc-phase-boundary
description: >-
  How to turn semi-grand-canonical (SGC, VC-SGC, hybrid MC-MD) Monte Carlo
  isotherms x(Δμ, T) into phase boundaries with the van de Walle & Asta
  thermodynamic-integration method, and trace a whole T-x coexistence line
  from one known point. Use when an SGC scan should yield a phase diagram,
  miscibility gap, binodal, solvus, coexistence Δμ, tie-line compositions or
  critical temperature, or when judging whether a scan can resolve a boundary.
---

# SGC phase-boundary analysis

## Overview

Semi-grand-canonical Monte Carlo holds temperature and the chemical-potential
difference Δμ = μ_B − μ_A fixed and lets composition x = x_B respond. A first-order
phase boundary shows up as a jump in x(Δμ) with hysteresis between a low-x branch
sweeping Δμ up and a high-x branch sweeping down; the coexistence Δμ lies inside
that hysteresis window. This skill applies A. van de Walle & M. Asta, Modelling
Simul. Mater. Sci. Eng. 10, 521 (2002):

- `benchmark/phase_boundary/sgc_phase_boundary.py` integrates φ = −∫x dΔμ per
  phase, finds φ_α = φ_γ, verifies jumps statistically, checks boundaries between
  temperatures with the Clausius-Clapeyron relation (eq. 29), and reports whether
  the data can support a boundary at all.
- `benchmark/hybrid_sgc_npt/boundary_tracer.py` integrates eq. 29 from one
  coexistence point, running two walkers (one per phase) per temperature step.
  `benchmark/hybrid_sgc_npt/run_campaign.py --mode trace-boundary` drives it with
  nvalchemi.

Equations, anchoring options, status labels and validation are in
`benchmark/phase_boundary/README.md`. Producing the SGC data is covered by
`nvalchemi-uma-submission` (settings, batch width, jobs) and
`nvalchemi-dynamics-api`.

## Understand the data first

Find out, from the files or by asking:

- Species and convention: which species is B? The analysis uses x = x_B and
  Δμ = μ_B − μ_A.
- What each run is: its T, Δμ, which branch (sweep) it belongs to and its order
  along that sweep. A boundary scan normally has a low-x branch sweeping Δμ upward
  and a high-x branch sweeping downward, each warm-started from its previous point.
- Error bars and equilibration flags: the x standard error and any drift or
  equilibration flag.
- The ensemble: rigid lattice, or relaxed / hybrid MD+MC. This decides which φ
  anchors are legitimate.
- Whether the simulations are physically sane. The analysis trusts ⟨x⟩ and cannot
  tell a broken simulation from real thermodynamics. Check that each branch sits
  near its own end at the extreme Δμ, that trends agree with sibling scans, that
  the closest atom pair in final structures is not far below a bond length
  (< ~1.5 A means collapse), that volume per atom stays near the end members, and
  that acceptance has not collapsed. Warm-started ladders pass corruption forward,
  so check late steps too. If the data fail, report that instead of a diagram.

## Prepare the input

A tidy CSV, one row per SGC run:

| Column | Required | Meaning |
|---|---|---|
| `T` | yes | temperature (K) |
| `mu` | yes | Δμ = μ_B − μ_A (eV) |
| `x` | yes | <x_B> |
| `branch` | recommended | sweep label; branches are ranked by mean x |
| `order` | recommended | step along the branch (default: Δμ order in the sweep direction) |
| `x_se` | recommended | standard error of ⟨x⟩ (default 0.005) |
| `x_drift` | optional | late-minus-early window mean; inflates unresolved runs' error bars |
| `E`, `E_se` | optional | ⟨E⟩ per atom (eV); needed for eq. 29 and `--anchor energy` |
| `resolved` | optional | equilibration gate passed |

nvalchemi campaigns write `*.equilibration.json` files (for example from
`run_campaign.py --mode delta-mu-scan`); pass their directory directly with
`--species A B`. For other sources (LAMMPS logs, icet / mchammer containers, ATAT
output) write a short converter to the CSV and keep it next to the results.

## Choose how φ is anchored

If the data are one connected phase the anchor is irrelevant. If they contain two
phases the relative φ constant must come from somewhere, and it can move the
boundary, so check with the user when it matters:

- `--pure-free-energies pure.json` (`{"T": {"A": F_A, "B": F_B}}`) is correct for
  any ensemble when end-member free energies are known.
- `--anchor energy` sets F_pure ≈ E_pure. It is valid only for rigid-lattice SGC;
  relaxed and hybrid runs carry vibrational entropy that differs between species.
- Tail anchors need sweeps that reach near-pure compositions; the script warns.
- With no anchor, opposite crossings of the two walkers still bracket Δμ_coex
  (`boundary-bracketed`).

## Run the analysis

```bash
python benchmark/phase_boundary/sgc_phase_boundary.py <data.csv | json_dir> \
    --species A B --out <results_dir> \
    [--pure-free-energies pure.json | --anchor energy] \
    [--temperatures 1200 1400] [--drop-unresolved] [--title "..."]
```

Other options: `--min-jump` (smallest x discontinuity counted as first order,
default 0.03), `--alpha` (false-positive rate of the jump test), `--bootstrap N`
(error bars, default 300) and `--no-drift-inflation`. Outputs in `--out`:

- `phase_boundary.png`: isotherms, φ(Δμ), the T-x diagram, G(x), stability
  ∂Δμ/∂x against the ideal kT/[x(1−x)], and branch hysteresis.
- `report.md` (status table and diagnostics), `summary.json`, and
  `phi_G_T<T>.csv` per temperature.

Put results next to the input data unless told otherwise.

## Interpret the result

Read `report.md` and the figure; for each temperature:

- `boundary-confirmed`: report Δμ_coex, x_α and x_γ with bootstrap spreads and the
  eq. 29 residual to neighbouring temperatures. A residual much larger than the
  Δμ_coex spread points to anchoring or equilibration problems.
- `boundary-bracketed`: Δμ_coex from the two walkers' opposite crossings, with no
  anchor needed; report the bracket and both compositions. If anchored φ curves
  exist but do not cross, the anchors are the suspect part.
- `boundary-tentative`: say why (no hysteresis, or an extrapolated crossing) and
  what data would confirm it.
- `single-phase`: say so, and how close the system is to instability (how far
  ∂Δμ/∂x falls below ideal). Say if the scan design could hide a gap.
- `inconclusive`: one sweep contains a jump candidate only the reverse sweep can
  settle; recommend running it.
- `no-crossing` / `unanchored`: explain what is missing.

Scan-design checks to raise even when not asked:

- Do the branches overlap in Δμ? Ladders that each stop at a shared centre cannot
  show hysteresis; each branch must run past the other's start.
- Is the steep part of x(Δμ) sampled by both branches, or only one?
- Are many runs near the transition unresolved or drifting? Drift toward the other
  phase suggests slow nucleation; those runs need more blocks.
- Is it a rigid lattice? Then the boundary belongs to the lattice model and
  ignores size-mismatch relaxation.

When the data cannot resolve a boundary, give a rerun plan: a new Δμ centre, a
bracket that lets each branch overshoot the other's start, a step size and the
blocks per step.

Stop and ask before: using a rigid-lattice anchor for relaxed or hybrid data;
dropping unresolved runs (`--drop-unresolved` can remove exactly the transition
points); fitting a solution model (Redlich-Kister and similar) and extrapolating
to a gap the data do not show -- that result is a model, not a measurement; and
relabelling branches or phases by hand.

## Trace the full T-x boundary

A boundary at one temperature (`boundary-confirmed` or `boundary-bracketed`)
starts the whole coexistence line. The tracer integrates eq. 29,

```text
dΔμ_coex/dβ = (E_γ − E_α) / [β (x_γ − x_α)] − Δμ_coex / β
```

from (T₀, Δμ₀). Each step runs two walkers, one per phase, at the same (T, Δμ),
warm-started from the previous step -- far cheaper than a Δμ ladder per
temperature, and no free-energy anchor is needed after the start. The tracer owns
the predictor-corrector, the in-phase checks, step halving and growth, critical
point and stopping logic and a resumable JSON trace; an engine supplies only
`run(T, mu, state_a, state_g, tag) -> (obs_a, obs_g, new_state_a, new_state_g)`
with observation keys `x, x_se, E, E_se, drift, resolved`.

- nvalchemi: `run_campaign.py --mode trace-boundary` (engine
  `NvalchemiTraceEngine`). Start from two checkpoints of the Δμ scan: the last
  point on each branch before it transformed, inside the hysteresis window.
  `--trace-replicas 2` runs two walkers per phase and rejects a step when one
  replica switches phase alone.
- Other engines: copy the tracer with
  `python benchmark/hybrid_sgc_npt/boundary_tracer.py vendor <dest.py> --header
  <license header>` and implement `run`.
- E must be the energy the SGC acceptance uses, per atom, with the same zero as
  Δμ: potential energy for rigid-lattice SGC, plus PV for NPT hybrids.

Direction matters (eq. 31):

- Downward in T is stable: the gap widens and errors shrink (a 4 meV start error
  decays to 1.3 meV over 350 K in the self-test).
- Upward toward T_c is not: expect ~1 meV Δμ drift, compositions good to
  0.02-0.04 within ~50 K of T_c, step rejections as walkers start switching, and a
  "gap closed" stop. The report fits gap ∝ (T_c − T)^β_c to estimate T_c
  (`--beta-c` 0.326 for 3D Ising, 0.5 for mean-field models); quote it as an
  estimate.
- An upward relaxed-lattice trace can end at melting rather than T_c; check
  crystallinity along it.
- Pass the start uncertainty (for example the hysteresis half-width) as `mu0_se`
  (`--trace-mu0-se` in `run_campaign.py`).

```bash
python benchmark/hybrid_sgc_npt/boundary_tracer.py report down/trace.json \
    up/trace.json --out <dir> --species A B --title "..." [--beta-c 0.326]
```

This writes `traced_boundary.csv`, `traced_boundary.json` (stop reasons,
rejections, T_c estimate) and `traced_boundary.png`. Before reporting, check each
point's `alpha_resolved` / `gamma_resolved` and whether the stop reason is the
target temperature or a physical limit.

## Validating the tools

- `python benchmark/phase_boundary/make_synthetic.py <dir> --design overlap`
  (or `meet`) builds sub-regular-solution SGC data with an exact binodal
  (`truth.json`) and exact end-member free energies (`pure.json`). Analyse it and
  compare with `truth.json`; expected accuracy is in the README.
- `python benchmark/phase_boundary/test_boundary_tracer.py` runs the tracer end to
  end on a mean-field model whose walkers switch phase only past their spinodals:
  a downward trace against the exact binodal, an upward trace stopping just below
  the exact T_c, and a biased start that must not grow going down. It prints
  PASS/FAIL. Run it after any change to the tracer.

## Key files

- `benchmark/phase_boundary/sgc_phase_boundary.py` -- isotherm analysis.
- `benchmark/phase_boundary/make_synthetic.py` -- synthetic data with exact answers.
- `benchmark/phase_boundary/test_boundary_tracer.py` -- tracer self-test.
- `benchmark/phase_boundary/README.md` -- equations, anchors, labels, validation.
- `benchmark/hybrid_sgc_npt/boundary_tracer.py` -- eq. 29 boundary tracer.
- `benchmark/hybrid_sgc_npt/run_campaign.py` -- Δμ scans and `--mode
  trace-boundary` for nvalchemi.
