# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backward-compatible capacity-planning names for hybrid MC-MD workflows."""

from nvalchemi.scheduling import (
    BatchMemoryEstimate,
    BatchMeasurement,
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
