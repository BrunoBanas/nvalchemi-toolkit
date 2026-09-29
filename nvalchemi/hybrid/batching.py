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
"""Backward-compatible capacity-planning names for hybrid MC-MD workflows."""

from __future__ import annotations

from nvalchemi.scheduling import (
    BatchMeasurement,
    BatchMemoryEstimate,
    RunAssignment,
    SimulationBatchPlanner,
)

__all__ = [
    "BatchMemoryEstimate",
    "BatchMeasurement",
    "HybridBatchPlanner",
    "RunAssignment",
    "SimulationBatchPlanner",
    "WalkerAssignment",
]

# Hybrid terminology remains valid but the implementation is simulation-generic.
HybridBatchPlanner = SimulationBatchPlanner
WalkerAssignment = RunAssignment
