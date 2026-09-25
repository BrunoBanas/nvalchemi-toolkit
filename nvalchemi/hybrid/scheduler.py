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
"""A block scheduler for alternating MC and MD on one Toolkit batch."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

from nvalchemi.data import Batch

if TYPE_CHECKING:
    from collections.abc import Iterator

    from nvalchemi.dynamics.base import BaseDynamics
    from nvalchemi.mc.base import BaseMonteCarlo

__all__ = ["HybridMCMD"]


class HybridMCMD:
    """Alternate batched MC and MD blocks without a host-side state handoff.

    The first implementation requires one shared model object for both stages.
    This makes every MC trial and MD force evaluation use the same potential.

    With ``mc_energy_only=True`` the shared model's ``active_outputs`` is
    narrowed to ``{"energy"}`` for each MC block and restored for MD, so a
    wrapper that honours ``active_outputs`` (``UMAWrapper`` does) skips the
    forces/stress backward on every MC trial. Each MC block then starts by
    re-evaluating its baseline energy under the same narrowed outputs
    (:meth:`~nvalchemi.mc.base.BaseMonteCarlo.refresh_energy`), rather than
    adopting MD's, so every acceptance ratio compares energies from one
    evaluation path -- one extra model call per block.
    """

    def __init__(
        self,
        mc: BaseMonteCarlo,
        md: BaseDynamics,
        mc_steps: int,
        md_steps: int,
        mc_energy_only: bool = False,
    ) -> None:
        """Initialize the hybrid scheduler.

        Parameters
        ----------
        mc
            A GPU-resident MC sampler.
        md
            A Toolkit molecular-dynamics integrator using the same model.
        mc_steps
            MC proposals per graph in each hybrid block.
        md_steps
            MD integration steps in each hybrid block.
        mc_energy_only
            Evaluate energies only (no forces/stress) during MC blocks.
        """
        if mc.model is not md.model:
            raise ValueError("MC and MD must share the same model object")
        if mc_steps < 0 or md_steps < 0:
            raise ValueError("mc_steps and md_steps must both be non-negative")
        if mc_energy_only and "energy" not in mc.model.model_config.outputs:
            raise ValueError("mc_energy_only requires a model that outputs energy")
        self.mc = mc
        self.md = md
        self.mc_steps = mc_steps
        self.md_steps = md_steps
        self.mc_energy_only = mc_energy_only

    @contextlib.contextmanager
    def _mc_outputs(self) -> Iterator[None]:
        """Narrow the shared model to energy-only for one MC block, then restore."""
        if not self.mc_energy_only:
            yield
            return
        config = self.mc.model.model_config
        full = set(config.active_outputs)
        config.active_outputs = {"energy"}
        try:
            yield
        finally:
            config.active_outputs = full

    def run_mc_block(self, batch: Batch) -> None:
        """Run one block of ``mc_steps`` MC proposals.

        Applies ``mc_energy_only`` (narrowed outputs plus the baseline
        re-evaluation). Call this, not ``self.mc.run``, from any loop that
        reimplements :meth:`run` -- e.g. to record observables per block -- or
        the energy-only setting is silently bypassed.

        Parameters
        ----------
        batch
            The shared hybrid batch, advanced in place.
        """
        if self.mc_steps == 0:
            return
        with self._mc_outputs():
            if self.mc_energy_only:
                self.mc.refresh_energy(batch)
            self.mc.run(batch, n_steps=self.mc_steps)

    def run(self, batch: Batch, n_blocks: int) -> Batch:
        """Run alternating MC and MD blocks on the supplied batch.

        A force/energy evaluation follows each MC block before MD begins. It
        prevents candidate-state derivatives from a rejected MC move being used
        by the MD integrator.
        """
        if n_blocks < 1:
            raise ValueError("n_blocks must be positive")
        if getattr(batch, "forces", None) is None:
            raise ValueError("hybrid MC-MD requires preallocated batch.forces")
        with self.md:
            self.md.compute(batch)
            self.mc.synchronize(batch)
            for _ in range(n_blocks):
                self.run_mc_block(batch)
                self.md.compute(batch)
                self.md.run(batch, n_steps=self.md_steps)
                self.mc.synchronize(batch)
        return batch
