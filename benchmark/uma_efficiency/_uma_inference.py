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
"""UMA inference-path knobs the three fairchem presets don't expose.

Two independent levers, both aimed at SGC's per-step cost:

**Custom ``InferenceSettings``.** ``"default"``/``"turbo"``/``"batch"`` are the
only named presets, and for SGC only ``"batch"`` is known safe --
``merge_mole=True`` assumes a fixed composition, which SGC breaks every step.
(``compile=True`` has only ever been measured together with ``merge_mole`` for
SGC, in the ``default``/``turbo`` presets, so whether compile alone is safe and
fast for SGC is untested, not known-bad.) But ``"batch"`` also turns on
``activation_checkpointing``, which recomputes activations during the backward
pass, and there is nothing composition-related about that. :func:`resolve`
accepts a ``key=value`` spec so the checkpointing and tf32 flags can be set
independently of the compile/merge flags.

**Energy-only evaluation.** Monte Carlo needs energies, never forces, but UMA's
``MLP_EFS_Head`` computes forces (and, for periodic tasks, stress) by autograd
through the energy on every call -- a backward pass per MC step whose result the
sampler discards. The head gates those branches on the backbone's
``regress_config``, so :func:`restrict_to_energy_only` clears
``regress_config.forces`` / ``.stress``, drops the matching entries from BOTH
of the model's task tables (``_tasks`` for output post-processing and
``_dataset_to_tasks`` for the collate step around ``predict()`` -- either one
left alone asks for an output that is no longer produced), and narrows the
nvalchemi wrapper's ``active_outputs``.

Both reach into fairchem internals (verified against fairchem-core 2.21/2.22:
``fairchem/core/models/uma/escn_md.py``'s ``MLP_EFS_Head.forward`` and
``fairchem/core/units/mlip_unit/predict.py``'s ``_process_outputs``), so
:func:`restrict_to_energy_only` fails loudly rather than silently doing nothing
if a future version moves them. Do NOT use it for a run that needs forces: MD,
NPT, and any hybrid MC-MD block all do.

Call it after the model is built and before the first evaluation. fairchem
initializes the model lazily on that first call, so if a version were to
(re-)register forces/stress tasks there, post-processing would then ask for an
output the head no longer produces and raise ``KeyError`` on the first MC step
-- loud, not silent. If that happens, call this again after one evaluation.
"""

from __future__ import annotations

from typing import Any

_DERIVATIVE_PROPERTIES = frozenset({"forces", "stress", "hessian"})
_NAMED_PRESETS = frozenset({"default", "turbo", "batch"})


def _coerce(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "yes", "1"}:
        return True
    if lowered in {"false", "no", "0"}:
        return False
    try:
        return int(value)
    except ValueError:
        return value.strip()


def resolve(spec: str) -> Any:
    """Return a preset name unchanged, or build ``InferenceSettings`` from a spec.

    ``spec`` is either one of fairchem's preset names or a comma-separated
    ``key=value`` list naming ``InferenceSettings`` fields, e.g.
    ``"compile=false,merge_mole=false,tf32=true,activation_checkpointing=false"``
    -- SGC-safe (no compile, no MoLE merge) without paying for activation
    recomputation. Unknown field names raise rather than being ignored.
    """
    if "=" not in spec:
        if spec not in _NAMED_PRESETS:
            raise SystemExit(
                f"--inference-settings {spec!r} is neither a preset "
                f"({', '.join(sorted(_NAMED_PRESETS))}) nor a key=value spec"
            )
        return spec

    from fairchem.core.units.mlip_unit.api.inference import (
        InferenceSettings,  # noqa: PLC0415
    )

    fields = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        key, separator, value = item.partition("=")
        if not separator:
            raise SystemExit(f"--inference-settings: {item!r} is not key=value")
        fields[key.strip()] = _coerce(value)
    known = set(InferenceSettings.__dataclass_fields__)
    unknown = set(fields) - known
    if unknown:
        raise SystemExit(
            f"--inference-settings: unknown field(s) {sorted(unknown)}; "
            f"InferenceSettings accepts {sorted(known)}"
        )
    return InferenceSettings(**fields)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (set, frozenset, tuple, list)):
        return sorted(str(item) for item in value)
    return str(value)  # e.g. base_precision_dtype is a torch.dtype


def describe(settings: Any) -> dict[str, Any]:
    """Return a JSON-safe description of a preset name or settings object."""
    if isinstance(settings, str):
        return {"preset": settings}
    fields = getattr(settings, "__dataclass_fields__", {})
    return {
        "preset": None,
        **{name: _json_safe(getattr(settings, name)) for name in fields},
    }


def restrict_to_energy_only(model: Any) -> dict[str, Any]:
    """Stop UMA computing forces/stress, so an MC step pays no backward pass.

    Returns a report of what was switched off, for the run's metrics. Raises
    ``RuntimeError`` if the fairchem internals it needs are not where it expects
    them -- a silent no-op would look like "energy-only made no difference".
    """
    predict_unit = getattr(model, "predict_unit", None)
    inner = getattr(getattr(predict_unit, "model", None), "module", None)
    regress_config = getattr(getattr(inner, "backbone", None), "regress_config", None)
    if regress_config is None:
        raise RuntimeError(
            "energy-only: could not reach predict_unit.model.module.backbone.regress_config; "
            "fairchem's model layout changed -- re-check MLP_EFS_Head.forward before trusting this flag"
        )
    if getattr(regress_config, "direct_forces", False):
        raise RuntimeError(
            "energy-only: this checkpoint predicts forces directly, not by autograd, "
            "so there is no backward pass to skip"
        )

    before = {
        "forces": bool(getattr(regress_config, "forces", False)),
        "stress": bool(getattr(regress_config, "stress", False)),
        "hessian": bool(getattr(regress_config, "hessian", False)),
    }
    for field in ("forces", "stress", "hessian"):
        if hasattr(regress_config, field):
            setattr(regress_config, field, False)

    # Tasks for properties we no longer produce have to go, from BOTH of
    # fairchem's task tables: _process_outputs iterates `_tasks` (name -> task),
    # and the collate_predictions decorator around predict() iterates
    # `_dataset_to_tasks` (dataset -> [task]). Missing the second one raises
    # KeyError('<task>_forces') on the first evaluation.
    tasks = getattr(inner, "_tasks", None)
    dataset_to_tasks = getattr(inner, "_dataset_to_tasks", None)
    if not isinstance(tasks, dict) or not isinstance(dataset_to_tasks, dict):
        raise RuntimeError(
            "energy-only: expected dict task tables at predict_unit.model.module._tasks and "
            "._dataset_to_tasks; fairchem's task layout changed -- re-check predict.py's "
            "collate_predictions and _process_outputs before trusting this flag"
        )

    def derivative(task: Any) -> bool:
        return getattr(task, "property", None) in _DERIVATIVE_PROPERTIES

    dropped_tasks = [name for name, task in tasks.items() if derivative(task)]
    for name in dropped_tasks:
        tasks.pop(name)
    for task_list in dataset_to_tasks.values():
        task_list[:] = [
            task for task in task_list if not derivative(task)
        ]  # in place: keep aliases valid

    leftover = [
        task.name
        for task_list in dataset_to_tasks.values()
        for task in task_list
        if derivative(task)
    ]
    leftover += [name for name, task in tasks.items() if derivative(task)]
    if leftover:
        raise RuntimeError(
            f"energy-only: derivative tasks survived pruning: {leftover}"
        )

    # nvalchemi side: stop the wrapper expecting a forces/stress key back.
    active_before = set(model.model_config.active_outputs)
    model.model_config.active_outputs = {"energy"}

    report = {
        "applied": True,
        "regress_config_before": before,
        "dropped_tasks": dropped_tasks,
        "active_outputs_before": sorted(active_before),
        "active_outputs_after": ["energy"],
    }
    if not before["forces"] and not before["stress"]:
        report["note"] = (
            "forces/stress were already off; no backward pass was being paid"
        )
    return report
