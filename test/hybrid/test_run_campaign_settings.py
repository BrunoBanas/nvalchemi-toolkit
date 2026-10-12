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
"""run_campaign.py's ``--inference-settings auto`` resolution (CPU only, no model)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "benchmark" / "hybrid_sgc_npt"
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

campaign = pytest.importorskip("run_campaign")


class TestResolveInferenceSettings:
    def test_auto_compiles_hybrid_runs_without_merge(self):
        spec = campaign.resolve_inference_settings("auto", md_steps_per_block=50)
        assert spec == campaign.INFERENCE_SETTINGS_COMPILED
        assert "compile=true" in spec and "merge_mole=false" in spec

    def test_auto_keeps_pure_sgc_uncompiled(self):
        spec = campaign.resolve_inference_settings("auto", md_steps_per_block=0)
        assert spec == campaign.INFERENCE_SETTINGS
        assert "compile=false" in spec and "merge_mole=false" in spec

    @pytest.mark.parametrize(
        "explicit", ["turbo", "compile=false,merge_mole=false,tf32=true"]
    )
    def test_explicit_spec_passes_through(self, explicit):
        assert (
            campaign.resolve_inference_settings(explicit, md_steps_per_block=50)
            == explicit
        )
