# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Collective force-bias Monte Carlo (tfMC / uniform-acceptance FBMC)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics.base import BaseDynamics
from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.mc.base import MonteCarloStats

if TYPE_CHECKING:
    from nvalchemi.dynamics.base import ConvergenceHook
    from nvalchemi.hooks import Hook
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["ForceBiasMC", "log_sinhc", "sample_tilted_uniform"]


def log_sinhc(x: torch.Tensor) -> torch.Tensor:
    r"""Return :math:`\log(\sinh(x)/x)`, stable for small and large :math:`|x|`."""
    ax = x.abs()
    small = ax < 1e-4
    safe = torch.where(small, torch.ones_like(ax), ax)
    large = (
        safe + torch.log1p(-torch.exp(-2.0 * safe)) - math.log(2.0) - torch.log(safe)
    )
    return torch.where(small, ax * ax / 6.0, large)


def sample_tilted_uniform(
    tilt: torch.Tensor, half_width: torch.Tensor, uniform: torch.Tensor
) -> torch.Tensor:
    r"""Draw :math:`\xi\in[-\Delta,\Delta]` with density :math:`\propto e^{a\xi}` by inversion.

    Parameters
    ----------
    tilt
        :math:`a`, any shape.
    half_width
        :math:`\Delta > 0`, broadcastable to ``tilt``.
    uniform
        Uniform draws on :math:`[0, 1)`, shape of ``tilt``.

    Returns
    -------
    torch.Tensor
        One sample per element of ``tilt``.
    """
    b = tilt.abs()
    sign = torch.where(tilt < 0, -torch.ones_like(tilt), torch.ones_like(tilt))
    b_delta = b * half_width
    flat = b_delta < 1e-6
    safe_b = torch.where(flat, torch.ones_like(b), b)
    # Positive tilt b: xi = Delta + log(u + (1 - u) exp(-2 b Delta)) / b.
    tilted = (
        half_width
        + torch.log(uniform + (1.0 - uniform) * torch.exp(-2.0 * b_delta)) / safe_b
    )
    plain = half_width * (2.0 * uniform - 1.0)
    return sign * torch.where(flat, plain, tilted)


class ForceBiasMC(BaseDynamics):
    r"""Collective force-bias Monte Carlo: every atom moves every step, biased by its force.

    Each step draws, for every Cartesian component of every atom, a
    displacement :math:`\xi \in [-\Delta_i, \Delta_i]` with density
    :math:`\propto \exp(\gamma \beta F \xi)`, with
    :math:`\Delta_i = \Delta (m_\mathrm{min}/m_i)^{1/4}` (Mees, Pourtois, Neyts,
    Thijsse & Stesmans, PRB 85, 134301 (2012)). Like an MD step it costs one
    energy-and-force call for the whole system, so it is a drop-in
    replacement for the MD integrator of :class:`~nvalchemi.hybrid.HybridMCMD`.

    ``metropolize=False`` is time-stamped force-bias MC (tfMC, uniform
    acceptance): every move is accepted. It samples the canonical
    configurational distribution only in the small-:math:`\Delta` limit.

    ``metropolize=True`` adds the Metropolis--Hastings test of Pangali, Rao &
    Berne (CPL 55, 413 (1978)) with the forces at both ends,

    .. math::

        \log A = -\beta\Delta U + \sum_k \bigl[-a'_k\xi_k - \log C(a'_k)
                 - a_k\xi_k + \log C(a_k)\bigr],\quad a = \gamma\beta F,
        \quad C(a) = 2\Delta\,\frac{\sinh(a\Delta)}{a\Delta},

    which is exact for any :math:`\Delta`; rejected graphs get back their
    positions, energy and forces. Acceptance of a collective move decays with
    system size, so :math:`\Delta` must shrink as :math:`N` grows.

    Velocities are neither used nor changed. ``remove_drift`` subtracts each
    graph's mass-weighted mean displacement (the tfMC centre-of-mass fix for
    a free particle); it changes the proposal density, so it is only allowed
    without ``metropolize``.
    """

    __needs_keys__: set[str] = {"energy", "forces"}
    __provides_keys__: set[str] = {"positions"}

    def __init__(
        self,
        model: BaseModelMixin,
        temperature: float | torch.Tensor,
        max_displacement: float = 0.1,
        gamma: float = 0.5,
        metropolize: bool = False,
        mass_scaling: bool = True,
        remove_drift: bool = False,
        random_seed: int = 421,
        n_steps: int | None = None,
        hooks: list[Hook] | None = None,
        convergence_hook: ConvergenceHook | dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a force-bias MC stage.

        Parameters
        ----------
        model
            Model that returns energies and forces.
        temperature
            Positive scalar temperature in K, or one temperature per graph.
        max_displacement
            :math:`\\Delta` in Å for the lightest atom of the batch.
        gamma
            Force-bias strength :math:`\\gamma` (1/2 is the tfMC choice).
        metropolize
            Add the exact Metropolis--Hastings test (see class notes).
        mass_scaling
            Scale :math:`\\Delta_i` by :math:`(m_\\mathrm{min}/m_i)^{1/4}`.
        remove_drift
            Remove each graph's mass-weighted mean displacement (tfMC only).
        random_seed
            Seed of the device generator.
        n_steps
            Default number of steps for :meth:`run`.
        hooks
            Hooks dispatched at the standard dynamics stages.
        convergence_hook
            Optional Toolkit convergence hook.
        **kwargs
            Forwarded to :class:`~nvalchemi.dynamics.base.BaseDynamics`.
        """
        super().__init__(
            model=model,
            n_steps=n_steps,
            hooks=hooks,
            convergence_hook=convergence_hook,
            **kwargs,
        )
        initial = torch.as_tensor(temperature)
        if initial.ndim > 1 or torch.any(initial <= 0):
            raise ValueError(
                "temperature must be positive and scalar or one-dimensional"
            )
        if max_displacement <= 0 or gamma <= 0:
            raise ValueError("max_displacement and gamma must be positive")
        if metropolize and remove_drift:
            raise ValueError(
                "remove_drift changes the proposal density; it is tfMC-only "
                "(metropolize=False)"
            )
        self.temperature = temperature
        self.max_displacement = float(max_displacement)
        self.gamma = float(gamma)
        self.metropolize = metropolize
        self.mass_scaling = mass_scaling
        self.remove_drift = remove_drift
        self._random_seed = random_seed
        self._generator: torch.Generator | None = None
        self._saved: dict[str, torch.Tensor] | None = None
        self._attempted: torch.Tensor | None = None
        self._accepted: torch.Tensor | None = None

    def _ensure_generator(self, batch: Batch) -> torch.Generator:
        if self._generator is None or self._generator.device != batch.device:
            self._generator = torch.Generator(device=batch.device)
            self._generator.manual_seed(self._random_seed)
        return self._generator

    def _beta_per_atom(self, batch: Batch) -> torch.Tensor:
        """Return :math:`1/k_BT` for every atom, shape ``[V, 1]``."""
        temperature = torch.as_tensor(
            self.temperature, dtype=batch.positions.dtype, device=batch.device
        )
        if temperature.ndim == 0:
            temperature = temperature.expand(batch.num_graphs)
        elif temperature.shape != (batch.num_graphs,):
            raise ValueError("temperature tensor must have one value per graph")
        return (1.0 / (KB_EV * temperature))[batch.batch_idx.to(torch.long), None]

    def _half_width(self, batch: Batch) -> torch.Tensor:
        """Return :math:`\\Delta_i` for every atom, shape ``[V, 1]``."""
        positions = batch.positions
        if not self.mass_scaling:
            return torch.full(
                (positions.shape[0], 1),
                self.max_displacement,
                dtype=positions.dtype,
                device=positions.device,
            )
        masses = getattr(batch, "atomic_masses", None)
        if masses is None:
            raise ValueError("mass_scaling=True needs batch.atomic_masses")
        masses = masses.detach().reshape(-1, 1).to(positions.dtype)
        return self.max_displacement * (masses.min() / masses) ** 0.25

    def pre_update(self, batch: Batch) -> None:
        """Draw and apply one force-biased displacement of every atom."""
        forces = getattr(batch, "forces", None)
        if forces is None:
            raise ValueError("ForceBiasMC needs batch.forces from a prior compute()")
        positions = batch.positions
        beta = self._beta_per_atom(batch)
        half_width = self._half_width(batch)
        tilt = self.gamma * beta * forces.detach()
        uniform = torch.rand(
            positions.shape,
            device=positions.device,
            dtype=positions.dtype,
            generator=self._ensure_generator(batch),
        )
        xi = sample_tilted_uniform(tilt, half_width, uniform)
        if self.remove_drift:
            graph = batch.batch_idx.to(torch.long)
            masses = batch.atomic_masses.detach().reshape(-1, 1).to(positions.dtype)
            weighted = torch.zeros(
                batch.num_graphs, 3, dtype=positions.dtype, device=positions.device
            ).index_add_(0, graph, masses * xi)
            total = torch.zeros(
                batch.num_graphs, 1, dtype=positions.dtype, device=positions.device
            ).index_add_(0, graph, masses)
            xi = xi - (weighted / total)[graph]
        if self.metropolize:
            self._saved = {
                "positions": positions.detach().clone(),
                "energy": batch.energy.detach().clone(),
                "forces": forces.detach().clone(),
                "tilt": tilt,
                "xi": xi,
                "half_width": half_width,
                "beta": beta,
            }
        with torch.no_grad():
            positions.add_(xi)

    def post_update(self, batch: Batch) -> None:
        """Metropolis--Hastings test on the new forces (``metropolize`` only)."""
        if not self.metropolize:
            return
        saved = self._saved
        if saved is None:
            raise RuntimeError("ForceBiasMC acceptance attempted without a proposal")
        graph = batch.batch_idx.to(torch.long)
        xi, half_width, beta = saved["xi"], saved["half_width"], saved["beta"]
        tilt_old = saved["tilt"]
        tilt_new = self.gamma * beta * batch.forces.detach()
        log_c_old = log_sinhc(tilt_old * half_width)
        log_c_new = log_sinhc(tilt_new * half_width)
        # The log(2 Delta) parts of log C cancel: Delta is the same both ways.
        per_atom = (-tilt_new * xi - log_c_new - tilt_old * xi + log_c_old).sum(dim=1)
        proposal = torch.zeros(
            batch.num_graphs, dtype=per_atom.dtype, device=per_atom.device
        ).index_add_(0, graph, per_atom)
        beta_graph = torch.zeros(
            batch.num_graphs, dtype=beta.dtype, device=beta.device
        ).index_add_(0, graph, beta.reshape(-1)) / batch.num_nodes_per_graph.to(
            beta.dtype
        )
        delta_u = (batch.energy.detach() - saved["energy"]).reshape(-1)
        log_accept = (-beta_graph * delta_u + proposal).clamp(max=0.0)
        log_uniform = torch.log(
            torch.rand(
                batch.num_graphs,
                device=batch.device,
                dtype=log_accept.dtype,
                generator=self._ensure_generator(batch),
            )
        )
        accepted = log_uniform < log_accept
        with torch.no_grad():
            reject_atom = (~accepted)[graph, None]
            batch.positions.copy_(
                torch.where(reject_atom, saved["positions"], batch.positions)
            )
            batch.forces.copy_(torch.where(reject_atom, saved["forces"], batch.forces))
            batch.energy.copy_(
                torch.where(
                    accepted.reshape_as(batch.energy), batch.energy, saved["energy"]
                )
            )
            attempted = torch.as_tensor(batch.num_graphs, device=batch.device)
            if self._attempted is None:
                self._attempted = attempted.clone()
                self._accepted = accepted.sum().detach()
            else:
                self._attempted += attempted
                self._accepted += accepted.sum().detach()
        self._saved = None

    def reset_statistics(self) -> None:
        """Reset the attempted/accepted counters (``metropolize`` only)."""
        self._attempted = None
        self._accepted = None

    @property
    def stats(self) -> MonteCarloStats:
        """Host-side attempted/accepted counts; zero attempts without ``metropolize``."""
        attempted = 0 if self._attempted is None else int(self._attempted.item())
        accepted = 0 if self._accepted is None else int(self._accepted.item())
        return MonteCarloStats(attempted=attempted, accepted=accepted)
