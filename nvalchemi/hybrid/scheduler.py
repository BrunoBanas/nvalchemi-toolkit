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

from typing import TYPE_CHECKING

from nvalchemi.data import Batch

if TYPE_CHECKING:
    from nvalchemi.dynamics.base import BaseDynamics
    from nvalchemi.mc.base import BaseMonteCarlo

__all__ = ["HybridMCMD"]


class HybridMCMD:
    """Alternate batched MC and MD blocks without a host-side state handoff.

    The first implementation requires one shared model object for both stages.
    This makes every MC trial and MD force evaluation use the same potential.
    """

    def __init__(
        self,
        mc: BaseMonteCarlo,
        md: BaseDynamics,
        mc_steps: int,
        md_steps: int,
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
        """
        if mc.model is not md.model:
            raise ValueError("MC and MD must share the same model object")
        if mc_steps < 1 or md_steps < 1:
            raise ValueError("mc_steps and md_steps must both be positive")
        self.mc = mc
        self.md = md
        self.mc_steps = mc_steps
        self.md_steps = md_steps

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
                self.mc.run(batch, n_steps=self.mc_steps)
                self.md.compute(batch)
                self.md.run(batch, n_steps=self.md_steps)
                self.mc.synchronize(batch)
        return batch
