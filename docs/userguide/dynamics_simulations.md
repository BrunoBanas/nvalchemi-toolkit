<!-- markdownlint-disable MD014 -->

(dynamics_simulations_guide)=

# Optimization and Integrators

This page covers the concrete simulation types provided by the dynamics module.
All of them follow the [execution loop](dynamics_guide) described in the dynamics
overview --- they generally differ only in what `pre_update` and `post_update` do.

## Geometry optimization

Geometry optimization finds the nearest local energy minimum by iteratively moving
atoms downhill on the potential energy surface. The toolkit provides the **FIRE**
(Fast Inertial Relaxation Engine) algorithm in two variants.

### Fixed-cell optimization

{py:class}`~nvalchemi.dynamics.optimizers.fire.FIRE` optimizes atomic positions
while keeping the simulation cell fixed:

```python
from nvalchemi.dynamics import FIRE, ConvergenceHook

with FIRE(
    model=model,
    dt=0.1,           # initial timestep (femtoseconds)
    n_steps=500,
    hooks=[ConvergenceHook.from_fmax(0.05)],
) as opt:
    relaxed = opt.run(batch)
```

FIRE uses an adaptive timestep and velocity mixing: when the system is moving
downhill (forces aligned with velocities), the timestep grows and velocities are
biased toward the force direction. When the system overshoots, the timestep shrinks
and velocities are zeroed. This makes it robust across a wide range of systems
without manual tuning.

### Variable-cell optimization

{py:class}`~nvalchemi.dynamics.optimizers.fire.FIREVariableCell` extends FIRE to
simultaneously optimize both atomic positions and the simulation cell. This is
useful for finding equilibrium crystal structures where the lattice parameters are
not known a priori:

```python
from nvalchemi.dynamics.optimizers.fire import FIREVariableCell
from nvalchemi.dynamics import ConvergenceHook

with FIREVariableCell(
    model=model,
    dt=0.1,
    n_steps=500,
    hooks=[ConvergenceHook.from_fmax(0.05)],
) as opt:
    relaxed = opt.run(batch)
```

The cell degrees of freedom are propagated using an NPH-like scheme at zero target
pressure. The model must return tensile-positive `stress` in addition
to `forces`.

### Choosing between fixed and variable cell

Use fixed-cell FIRE when the cell is known (e.g. a bulk crystal at experimental
lattice parameters, or a molecule in vacuum where the cell is just a bounding box).
Use variable-cell FIRE when the equilibrium cell shape or volume is unknown, such as
when screening candidate crystal structures or computing equations of state.

## Molecular dynamics

Molecular dynamics (MD) propagates the equations of motion forward in time, sampling
the trajectory of a system at finite temperature. The toolkit provides integrators
for three standard ensembles.

### NVE: energy conservation

{py:class}`~nvalchemi.dynamics.integrators.nve.NVE` uses the Velocity Verlet
algorithm --- a symplectic integrator that conserves total energy in the
microcanonical ensemble:

```python
from nvalchemi.dynamics import NVE

with NVE(model=model, dt=1.0, n_steps=1000) as md:
    trajectory = md.run(batch)
```

NVE is the natural choice for verifying that a model's energy surface is smooth
enough for stable dynamics: if the total energy drifts significantly, the force
field is likely too noisy for the chosen timestep.

### NVT: constant temperature

{py:class}`~nvalchemi.dynamics.integrators.nvt_langevin.NVTLangevin` implements the
BAOAB Langevin splitting scheme, which samples the canonical (NVT) ensemble exactly
--- the thermostat does not introduce systematic bias:

```python
from nvalchemi.dynamics import NVTLangevin

with NVTLangevin(
    model=model,
    dt=1.0,              # femtoseconds
    temperature=300.0,    # Kelvin
    friction=0.01,        # collision frequency (1/fs)
    n_steps=10000,
) as md:
    trajectory = md.run(batch)
```

The `friction` parameter controls how strongly the thermostat couples to the
system. A low value gives longer correlation times (closer to NVE); a high value
thermalises quickly but damps real dynamics.

### NPT: constant pressure

{py:class}`~nvalchemi.dynamics.integrators.npt.NPT` uses the
Martyna--Tobias--Klein (MTK) barostat with Nose--Hoover chains to sample the
isothermal-isobaric ensemble. Both the atomic positions and the simulation cell
evolve:

```python
from nvalchemi.dynamics import NPT

with NPT(
    model=model,
    dt=1.0,
    temperature=300.0,
    pressure=1.0,            # target pressure (eV/Å^3; positive = compression)
    barostat_time=100.0,     # barostat coupling time (fs)
    thermostat_time=100.0,   # thermostat coupling time (fs)
    n_steps=10000,
) as md:
    trajectory = md.run(batch)
```

The model must return `stress` for NPT to propagate the cell degrees of freedom.

## Monte Carlo

The independent `nvalchemi.mc` package provides batched Monte Carlo samplers.
`SGC` proposes single-site species transmutations and samples the
semi-grand-canonical potential `E - sum(mu_i N_i)`:

```python
from nvalchemi.mc import SGC

with SGC(
  model=model,
  temperature=1000.0,
  species=[1, 2],
  chemical_potentials={1: 0.0, 2: 0.2},
  n_steps=10000,
) as mc:
  result = mc.run(batch)

```

One proposal is attempted independently for every active graph per step.
Accepted moves are available as the graph-level boolean `batch.mc_accepted`.
Composition-conserving Kawasaki sampling will be added as a separate MC style;
it is intentionally not exposed until its local proposal graph and detailed
balance tests are complete.

## Hybrid MC-MD blocks

`HybridMCMD` owns the alternation between an MC sampler and a Toolkit MD
integrator. Both stages must use the same model object. The scheduler keeps one
`Batch` on the active GPU, refreshes forces and stress after MC, then starts the
MD block from the accepted configuration:

```python
from nvalchemi.hybrid import HybridMCMD
from nvalchemi.mc import SGC

mc = SGC(
  model=model,
  temperature=1000.0,
  species=[1, 2],
  chemical_potentials={1: 0.0, 2: 0.2},
)
scheduler = HybridMCMD(mc=mc, md=npt, mc_steps=100, md_steps=20)
result = scheduler.run(batch, n_blocks=1000)
```

The supplied batch must contain a preallocated `forces` field, as required by
the selected MD integrator. Do not combine the MC stage with `FusedStage`:
alternating MC-MD needs a candidate-energy evaluation and an accepted-state
force evaluation at different points in each block.

### Simulation batch planning

`SimulationBatchPlanner` selects a batch width for any collection of independent
simulations: MC, MD, geometry optimization, hybrid MC-MD, or a user-defined
runner. Its `profile` method runs representative work at candidate widths,
records actual CUDA reserved memory and throughput, then `recommend_width`
selects the smallest width near peak throughput while retaining memory headroom.
Its `assign_runs` method packs a larger campaign into concurrent per-GPU batches
and serial queue waves. `HybridBatchPlanner` remains an alias for compatibility.

```python
import torch

from nvalchemi.scheduling import SimulationBatchPlanner

planner = SimulationBatchPlanner(memory_fraction=0.85, throughput_fraction=0.95)
measurements = planner.profile(
  workload_factory=make_workload,
  widths=[1, 2, 4, 8, 16],
  device="cuda:0",
  warmup_blocks=2,
  measured_blocks=4,
)
width = planner.recommend_width(
  measurements,
  total_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
)
assignments = planner.assign_runs(total_runs=100, batch_width=width, gpu_ids=[0, 1])
```

`make_workload(width, device)` must reuse the production model and
return a newly constructed `(runner, batch)` pair of that width. The runner only
needs a `run(batch, n_blocks=...)` method, so it can wrap an MC, MD, optimizer,
or hybrid protocol. Assignments
for the same GPU are deliberately serial: one model and one full batch occupy
the device at a time. Launch one process per requested GPU and feed it that
GPU's queue in wave order.

For a quick estimate on a second, memory-different GPU, fit the successful
profile points and use the fitted model with the target device memory:

```python
memory = planner.infer_memory_model(measurements)
estimated_width = planner.estimate_width(
  total_memory_bytes=target_gpu_total_memory,
  model_resident_bytes=memory.model_resident_bytes,
  bytes_per_walker=memory.bytes_per_walker,
)
```

This extrapolation is deliberately conservative and should only choose the
candidate widths for a short target-GPU profile; it should not be treated as a
production capacity claim after changing the model or physics configuration.

For heterogeneous system sizes, use `SizeAwareSampler` to pack each active GPU
batch subject to calibrated `max_atoms`, `max_edges`, and `max_batch_size`
budgets. Its default `estimated_bytes_per_atom=300` and
`model_memory_fraction=0.2` remain available as a conservative starting
heuristic. They are explicit parameters: validate them for the selected model
with a representative profile, then override them if needed. A completed
independent run may be replaced from the queue; an active run must retain its
own configuration and simulation state until it reaches its declared stopping
condition.

### Dependency-aware campaigns

`CampaignSpec` describes state points as explicit `RunSpec` nodes. A run may
have one `parent_id`, whose final atomic state seeds a serial continuation, and
additional `depends_on` edges that express a barrier without changing the
continuation state. `CampaignScheduler` only exposes runs whose dependencies
have durable final-state checkpoints. It batches compatible ready runs even
when their temperatures, pressures, chemical potentials, structures, or sizes
differ; the selected runner must support the corresponding per-graph tensors.

```python
from nvalchemi.scheduling import CampaignSpec, RunSpec

reference = [
    RunSpec(
        run_id="mu_minus",
        temperature_k=3000.0,
        pressure_ev_per_a3=6.324e-7,
        species=(79, 78),
        chemical_potentials_ev={79: 0.0, 78: -0.10},
    ),
    RunSpec(
        run_id="mu_plus",
        temperature_k=3000.0,
        pressure_ev_per_a3=6.324e-7,
        species=(79, 78),
        chemical_potentials_ev={79: 0.0, 78: +0.10},
    ),
]
campaign = CampaignSpec.cooling_from_reference(
    reference,
    temperatures_k=[3000.0, 2800.0, 2600.0],
)
```

When a run completes, pass its individual `AtomicData` final state to
`CampaignScheduler.complete`. The `FinalStateStore` writes it atomically and a
child subsequently receives it through `parent_state`. The record includes
positions, cell, atom types, velocities, and other atomic fields. Advanced
integrator/MC restart state may be supplied separately as a tensor-only
`runtime_state` payload.

`SGC` accepts scalar reservoirs or one chemical-potential value per graph for
each species. Thus different `delta_mu` values can be batched with different
temperatures, provided all graphs share the model, MC style, and reservoir
species. `NPT` already accepts per-graph temperatures and pressures. For a
mixed-size campaign, apply `SizeAwareSampler` or an equivalent atom/edge budget
when choosing `max_batch_size`; `CampaignScheduler` preserves the dependency
graph while the capacity layer decides how many ready runs fit together.

## Writing your own dynamics

All integrators and optimizers inherit from
{py:class}`~nvalchemi.dynamics.base.BaseDynamics`. To implement a custom one, you
subclass it and override `pre_update` and `post_update` --- the two methods that
define how the batch state evolves within a single step.

### The minimal contract

Your subclass must provide:

1. **`__needs_keys__`** --- a set of strings naming the model outputs your dynamics
   reads (e.g. `{"forces"}`, or `{"forces", "stress"}` for cell-aware schemes).
2. **`__provides_keys__`** --- a set of strings naming the batch keys your dynamics
   writes (e.g. `{"positions", "velocities"}`).
3. **`pre_update(batch)`** --- called *before* the model forward pass. Typically
   updates positions using current velocities and/or forces.
4. **`post_update(batch)`** --- called *after* the model forward pass. Typically
   completes the velocity update with the newly computed forces.

Both methods receive the {py:class}`~nvalchemi.data.Batch` and modify it
**in-place**. Return value is `None`.

### Example: a Velocity Verlet integrator

The `DemoDynamics` class in `nvalchemi.dynamics.demo` is a complete, minimal
Velocity Verlet implementation that is useful as a template:

```python
from nvalchemi.data import Batch
from nvalchemi.dynamics.base import BaseDynamics, ConvergenceHook

class MyVerlet(BaseDynamics):
    __needs_keys__ = {"forces"}
    __provides_keys__ = {"positions", "velocities"}

    def __init__(self, model, n_steps, dt=1.0, hooks=None, convergence_hook=None, **kwargs):
        super().__init__(
            model=model, hooks=hooks, convergence_hook=convergence_hook,
            n_steps=n_steps, **kwargs,
        )
        self.dt = dt
        self._prev_accelerations = None

    def pre_update(self, batch: Batch) -> None:
        """Position half-step: x(t+dt) = x(t) + v*dt + 0.5*a*dt^2."""
        import torch
        positions = batch.positions
        velocities = batch.velocities
        forces = batch.forces
        masses = batch.atomic_masses.unsqueeze(-1)

        with torch.no_grad():
            if forces is not None and not torch.all(forces == 0):
                acc = forces / masses
                self._prev_accelerations = acc.clone()
                positions.add_(velocities * self.dt + 0.5 * acc * self.dt**2)
            else:
                positions.add_(velocities * self.dt)

    def post_update(self, batch: Batch) -> None:
        """Velocity half-step: v(t+dt) = v(t) + 0.5*(a_old + a_new)*dt."""
        import torch
        velocities = batch.velocities
        forces = batch.forces
        masses = batch.atomic_masses.unsqueeze(-1)

        with torch.no_grad():
            new_acc = forces / masses
            if self._prev_accelerations is not None:
                velocities.add_(0.5 * (self._prev_accelerations + new_acc) * self.dt)
            else:
                velocities.add_(new_acc * self.dt)
```

```{important}
The demo ``DemoDynamics`` class is intended for debugging and pedagogy
only. Do not use this class for production runs, and instead, see the
{py:class}`~nvalchemi.dynamics.integrators.nve.NVE` class instead.
```

### Data flow through a step

Understanding what the batch contains at each point is key to writing correct
updates:

| Point in step | What just happened | What the batch contains |
|---------------|--------------------|-------------------------|
| `pre_update` entry | Hooks ran | Positions and velocities from the *previous* step; forces may be from the previous `compute` (or absent on step 0) |
| `pre_update` exit | You updated positions | New positions; velocities partially updated (or unchanged) |
| After `compute` | Model ran | Fresh `forces` (and `energy`, `stress`, etc.) for the new positions |
| `post_update` entry | Forces are fresh | Complete the velocity update with new forces |
| `post_update` exit | Step is done | Consistent positions, velocities, and forces for the current timestep |

### Gotchas and tips

- **Use `torch.no_grad()`**: Wrap in-place updates in `torch.no_grad()` to avoid
  conflicts with autograd. When `forces_via_autograd=True`, `compute()` sets
  `requires_grad_(True)` on positions to compute forces via backprop.
- **In-place operations**: Modify batch tensors in-place (`positions.add_(...)`)
  rather than reassigning. The batch's storage model expects tensors to be updated
  in place.
- **First-step fallback**: On the first call to `pre_update`, forces may be `None`
  or zero (no model evaluation has happened yet). Guard against this and fall back
  to an Euler step.
- **Per-system state**: If your integrator needs auxiliary state (e.g. thermostat
  variables, previous accelerations), store it as instance attributes. The
  `_prev_accelerations` pattern above is typical.
- **`__needs_keys__` matters**: `BaseDynamics` uses this set to verify the model
  produces the required outputs before the simulation starts. If your dynamics needs
  stress, declare `{"forces", "stress"}`.
- **FusedStage compatibility**: When your dynamics runs inside a
  {py:class}`~nvalchemi.dynamics.base.FusedStage`, a save-and-restore mask is
  applied around `pre_update` and `post_update` so that only systems belonging to
  your stage are modified. You do not need to handle masking yourself.
- **Running under domain decomposition**: A per-atom integrator works under
  {py:class}`~nvalchemi.distributed.DomainParallel` unchanged, but any *global*
  reduction (kinetic energy, temperature, a convergence dot-product) needs
  cross-rank handling. See {doc}`distributed` → *Distributed dynamics* for the
  contract and the `HookScope.GLOBAL` recipe.

## See also

- **Overview**: The [Dynamics overview](dynamics_guide) describes the shared execution
  loop and multi-stage pipelines.
- **Hooks**: The [Hooks guide](hooks_guide) covers convergence criteria,
  logging, and snapshots.
- **Examples**: ``basic/02_geometry_optimization.py`` demonstrates a complete relaxation
  workflow.
