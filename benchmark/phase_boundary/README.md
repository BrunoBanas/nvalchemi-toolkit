# SGC phase-boundary analysis

Turns semi-grand-canonical (SGC / VC-SGC / hybrid MC-MD) averages
⟨x⟩(Δμ, T), and optionally ⟨E⟩, into phase boundaries with diagnostics that say
whether the data can support one, and traces a whole T-x coexistence line from one
known coexistence point. The method is A. van de Walle & M. Asta, "Self-driven
lattice-model Monte Carlo simulations of alloy thermodynamic properties and phase
diagrams", Modelling Simul. Mater. Sci. Eng. 10, 521 (2002). The
`nvalchemi-sgc-phase-boundary` agent skill (`.claude/skills/`) drives these files.

| File | Role |
| --- | --- |
| `sgc_phase_boundary.py` | Isotherm analysis: φ = −∫x dΔμ, φ_α = φ_γ crossings, jump verification, eq. 29 checks, figure and report. |
| `make_synthetic.py` | Sub-regular-solution SGC data with an exact binodal (`truth.json`) and exact F_pure (`pure.json`). |
| `test_boundary_tracer.py` | End-to-end self-test of the tracer on a mean-field model; prints PASS/FAIL. |
| `../hybrid_sgc_npt/boundary_tracer.py` | Engine-agnostic eq. 29 boundary tracer; `run_campaign.py --mode trace-boundary` drives it with nvalchemi. |

## Usage

```bash
python benchmark/phase_boundary/sgc_phase_boundary.py <data.csv | json_dir> \
    --species A B --out <results_dir> \
    [--pure-free-energies pure.json | --anchor energy] [--temperatures 1200 1400]
python benchmark/hybrid_sgc_npt/boundary_tracer.py report down/trace.json \
    up/trace.json --out <dir> --species A B [--beta-c 0.326]
python benchmark/phase_boundary/make_synthetic.py <dir> --design overlap
python benchmark/phase_boundary/test_boundary_tracer.py
```

Input CSV, one row per SGC run: `T`, `mu` (Δμ = μ_B − μ_A, eV) and `x` (<x_B>) are
required; `branch`, `order` and `x_se` are recommended; `x_drift`, `E`, `E_se` and
`resolved` are optional. A directory of nvalchemi campaign `*.equilibration.json`
files (for example from `run_campaign.py --mode delta-mu-scan`) can be passed
directly with `--species`.

## Conventions

| Paper | Here |
| --- | --- |
| μ = μ_A − μ_B, x = x_A | Δμ = μ_B − μ_A, x = x_B (the species listed second in `--species A B`) |
| φ(β, μ) = F − μx | φ = F − Δμ·x (μ_A is the energy zero; shifting it moves every phase equally) |

G(x) = φ + Δμ·x, so ∂G/∂x = Δμ and ∂²G/∂x² = ∂Δμ/∂x, the stability plotted in
panel (e) of the figure.

## Equations used

| Eq. | Content | Where |
| --- | --- | --- |
| (2)-(3) | d(βφ) = (E − μx)dβ − βx dμ; at fixed T, dφ = −x dΔμ (trapezoid rule) | `integrate()` |
| (6)-(7) | a boundary is where φ_α = φ_γ; coexisting x = −∂φ/∂Δμ of each phase | `analyse_isotherm()` |
| (15) | z_α = √2 erf⁻¹(1 − α), set by `--alpha` (default 0.01, z = 2.576) | `main()` |
| (17) | leave-one-out cross-validation picks the extrapolation order (0-2) | `jump_test()` |
| (21)-(26) | prediction variance; a jump needs \|Q − Q*\| ≥ z√(v + σ²) | `jump_test()` |
| Fig. 5 | a walker changed phase when it matches the other branch better than its own | `segment_branch()` |
| (29) | dΔμ/dβ = (E_γ − E_α)/[β(x_γ − x_α)] − Δμ/β, checked between temperatures | `clausius_clapeyron_check()` |
| (31) | the tracing error is stable when integrating from high to low T | trace downward in T |

## Fixing the integration constant

Eq. (3) fixes φ only up to a constant per connected path. The script applies, in
order:

1. Continuity. No verified jump and no significant hysteresis means one phase,
   integrated as a single path; the isotherm is single-phase.
2. Pure-end anchors (`--pure-free-energies`). The low-x phase is anchored to F_A
   and the high-x phase to F_B through a Langmuir dilute tail:
   φ_low(Δμ₁) = F_A + kT ln(1 − x₁) and φ_high(Δμ_N) = F_B − Δμ_N + kT ln x_N.
   The error is O(Ω x₁²), so sweeps must reach near-pure compositions. F_A and
   F_B must share the energy zero of E and Δμ.
3. Energy extrapolation (`--anchor energy`): F_pure ≈ E_pure from a quadratic
   fit. Exact only for rigid-lattice SGC. Wrong for hybrid MD+MC or relaxed
   runs, where F_pure = E_pure − T·S_vib and vibrational entropy differs between
   the species; those need an independent free energy.

A φ difference δ between phases moves Δμ_coex by about δ/(x_γ − x_α): a 1 meV/atom
anchor error across a 0.8-wide gap shifts Δμ_coex by about 1.25 meV.

## Status labels

- `single-phase`: one connected phase; no boundary in the sampled window.
- `boundary-confirmed`: φ_low = φ_high inside both phases' sampled ranges, with
  significant hysteresis there.
- `boundary-tentative`: a crossing that relies on extrapolated φ or x, or has no
  hysteresis evidence (for example ladders that only meet in the middle).
- `boundary-bracketed`: both walkers crossed in opposite directions, bracketing
  Δμ_coex without any free-energy anchor (paper Fig. 6); compositions come from
  each phase's home walker.
- `inconclusive`: one connected dataset with a candidate jump that no opposite
  sweep can confirm or rule out; run the reverse sweep.
- `no-crossing`: two phases, one stable across the whole window; extend the sweep.
- `unanchored`: two phases with no way to set the relative φ constant.

## Jump verification

A candidate jump (fails the eq. 26 test and moves x with the sweep) is confirmed
against the other phase's sequence, the opposite walker's own pre-jump segment,
iterated to self-consistency -- not against its raw data, which after its own
crossing sit in the same phase and would veto a real jump. Confirmation needs
distinct phases where both sequences have data (with no shared Δμ, the jump is
trusted only if the opposite walker also crossed) and a better prediction from
the other phase than from the walker's own trend. A run still drifting toward the
other phase is tested with its raw error bar, held out of the trend that judges
the next point, and classed as mid-transformation if a jump follows.

## Boundary tracing

Differentiating βφ_α = βφ_γ along the coexistence line gives eq. (29). Everything
on its right-hand side is measured at coexistence, so two walkers, one per phase,
give the slope, and no anchor is needed after the starting point.

- Steps are taken in β: Adams-Bashforth 2 predictor (Euler first), walkers run at
  the predicted Δμ, trapezoid corrector, rerun if the corrector moves Δμ by more
  than `tol_mu` (2 meV). The accepted Δμ is the one the walkers ran at.
- A step is rejected (walkers restored) if a walker jumps toward the other phase,
  keeps drifting toward it, or the gap falls below `min_gap`.
- Recentering (paper Fig. 6, tracer 1.2, `recenter=True` by default): if exactly
  one walker left its phase (a collapsed gap counts when that transformation
  explains it), the tracer re-measures Δμ_coex at the same T:
  1. sweep that walker's Δμ back until it returns to its phase (Δμ_back, state S);
  2. from S, sweep forward until it transforms again (Δμ_fwd);
  3. restart both walkers at the midpoint and continue integrating, with
     Δμ_se = half the bracket plus half the last sweep step.

  Sweeps start at 2.5 meV, double every 4 runs (up to 8×) and use the engine's
  `run_one` if it has one. Give the start compositions (`x_a0`, `x_g0`;
  `--trace-x-alpha0/--trace-x-gamma0`) to recenter a mis-centred start too. The
  midpoint is biased by half any asymmetry between the two metastability limits.
  If recentering fails, dT is halved instead.
  Tracing stops when dT would fall below `dt_min`, distinguishing "gap closed"
  (critical point) from "walkers keep transforming".
- The step grows 1.5x after a clean step, up to `dt_max`; while the gap shrinks it
  shrinks by at most 15% per step.
- Error stability (eq. 31): errors decay where the gap widens (usually downward in
  T) and grow toward T_c. T_c is estimated from gap = A(T_c − T)^β_c fitted to
  the last points (β_c = 0.326 for 3D Ising, 0.5 for mean-field models).

## Validation

Synthetic sub-regular solution (L0 = 0.20 eV, L1 = 0.03 eV, T_c ≈ 1160 K, noise
0.004 in x), `make_synthetic.py`:

| Design | Anchors | Result vs. exact binodal |
| --- | --- | --- |
| Overlapping branches | supplied F | confirmed at 800, 950 and 1050 K; x within ≈ 0.005, Δμ_coex within ≈ 1 meV; 1300 K single-phase |
| Overlapping branches | E-extrapolated | confirmed; noisier near T_c (1050 K x_γ off by ≈ 0.025) |
| Ladders meet in the centre | supplied F | tentative; x off by up to 0.02; 1300 K inconclusive |
| One sweep only | none | inconclusive at every T (correct) |
| Overlapping branches | none (bracket only) | bracketed; Δμ_coex exact at 950 / 1050 K |
| Any | any | eq. (29) residuals ≲ 3 meV between adjacent T |

Tracer self-test (`test_boundary_tracer.py`; L0 = 0.20, L1 = 0.03 eV, exact
T_c = 1214 K, noise 0.003 in x and 1.5 meV in E):

| Test | Result |
| --- | --- |
| Down 950 → 600 K, exact start | Δμ within 0.21 meV, x within 0.004 at all points |
| Up from 950 K | stops at 1195 K ("gap closed", after recentering near T_c); T_c estimate 1214.2 K (exact 1214.3); 1.1 stopped at 1161 K |
| Down with Δμ₀ off by +4 meV | error decays to +1.3 meV at 600 K |
| Start off by +10 meV, nucleation window ±6 meV | recentered start within 0.0 meV, worst 1.8 meV to 600 K; without recentering the start is rejected |
| Asymmetric window (+3 / −9 meV) | recentered start biased −2.5 meV (predicted −3.0) |
| Trace drifting out of a ±4 meV window | 2 recenterings, reaches 600 K within 3.5 meV; without: stops at 834 K |

Run both after any change to the analysis or the tracer.

## Lessons from real data (Au-Pt, UMA, 500 atoms)

- Ladders that march toward a common centre and stop there share one Δμ and
  cannot show hysteresis; each branch must run past the other's start.
- Centre the ladder on the steep part of x(Δμ); a Δμ reference calibrated in a
  different ensemble can be off by ~0.1 eV.
- Treat unresolved runs as suspect: Pt-rich runs that keep losing Pt look like a
  slowly nucleating transformation.
- Rigid-lattice SGC ignores size-mismatch relaxation; its boundary belongs to the
  lattice model.
