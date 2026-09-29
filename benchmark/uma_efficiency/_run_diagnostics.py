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
"""Crash- and timeout-tolerant progress diagnostics for long GPU benchmarks.

A benchmark that only writes ``metrics.json`` at the very end leaves nothing
behind when Slurm kills it at the time limit. :class:`RunDiagnostics` keeps
three files current in ``out_dir`` for the whole run instead:

* ``progress.jsonl`` -- one JSON object per event (phase changes, per-width
  profile results, per-block records) plus a periodic ``heartbeat`` with GPU
  and host memory; every line is flushed as written, so a SIGKILL loses at
  most the line being written.
* ``partial_metrics.json`` -- the run's full current state, atomically
  rewritten on every heartbeat and event, so it is never more than one
  heartbeat stale even if the main thread is stuck inside a CUDA call.
* ``stacks.txt`` -- Python stack dumps of every thread: periodically (to show
  where a slow phase is spending its time) and on SIGUSR1/SIGTERM. These are
  written by ``faulthandler`` at C level, so they appear even while the main
  thread is blocked in native code.

SIGUSR1 (sent by the sbatch ahead of the time limit via ``--signal``) records
the signal and flushes the state; SIGTERM (Slurm's time-limit kill) does the
same, marks the run ``terminated`` and exits 143.
"""

from __future__ import annotations

import faulthandler
import json
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import torch

_GIB = 1024**3


def _host_rss_gib() -> float | None:
    try:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024 / _GIB
    except OSError:
        return None
    return None


class RunDiagnostics:
    """Incremental on-disk progress log, heartbeat, and stack dumps."""

    def __init__(
        self,
        out_dir: Path,
        device: torch.device,
        *,
        heartbeat_seconds: float = 60.0,
        stack_dump_seconds: float = 1800.0,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.heartbeat_seconds = heartbeat_seconds
        self.start = time.perf_counter()
        # Reentrant: the signal handlers run on the main thread, possibly while it holds the lock.
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._progress = open(self.out_dir / "progress.jsonl", "a", buffering=1)
        self._stacks = open(self.out_dir / "stacks.txt", "a", buffering=1)
        self.state: dict[str, Any] = {
            "status": "running",
            "phase": "startup",
            "phase_started_elapsed_s": 0.0,
            "signals_received": [],
            "run_info": {
                "hostname": socket.gethostname(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "argv": sys.argv,
                "torch_version": torch.__version__,
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_total_memory_GiB": torch.cuda.get_device_properties(
                    device
                ).total_memory
                / _GIB,
                "started_unix": time.time(),
            },
        }
        signal.signal(signal.SIGUSR1, self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)
        # Registered after signal.signal so chain=True also runs the Python handlers.
        for signum in (signal.SIGUSR1, signal.SIGTERM):
            faulthandler.register(
                signum, file=self._stacks, all_threads=True, chain=True
            )
        if stack_dump_seconds > 0:
            faulthandler.dump_traceback_later(
                stack_dump_seconds, repeat=True, file=self._stacks
            )
        self._thread = threading.Thread(
            target=self._heartbeat_loop, name="diagnostics-heartbeat", daemon=True
        )
        self._thread.start()
        self.event("start")

    # -- public API ---------------------------------------------------------

    def elapsed(self) -> float:
        return time.perf_counter() - self.start

    def set_phase(self, phase: str, **fields: Any) -> None:
        with self._lock:
            self.state["phase"] = phase
            self.state["phase_started_elapsed_s"] = self.elapsed()
        self.event("phase", phase=phase, **fields)

    def update(self, **fields: Any) -> None:
        with self._lock:
            self.state.update(fields)
        self._write_partial()

    def append(self, key: str, record: dict[str, Any]) -> None:
        """Append ``record`` to the list ``state[key]`` and log it as an event."""
        with self._lock:
            self.state.setdefault(key, []).append(record)
        self.event(key, **record)

    def event(self, kind: str, **fields: Any) -> None:
        line = {
            "event": kind,
            "elapsed_s": round(self.elapsed(), 3),
            "unix": round(time.time(), 3),
            **fields,
        }
        text = json.dumps(line, default=str)
        with self._lock:
            self._progress.write(text + "\n")
        print(f"[diag] {text}", flush=True)
        self._write_partial()

    def cuda_memory(self) -> dict[str, float]:
        return {
            "cuda_allocated_GiB": torch.cuda.memory_allocated(self.device) / _GIB,
            "cuda_reserved_GiB": torch.cuda.memory_reserved(self.device) / _GIB,
            "cuda_max_reserved_GiB": torch.cuda.max_memory_reserved(self.device) / _GIB,
        }

    def close(self, status: str) -> None:
        self.update(status=status)
        self.event("finish", status=status)
        self._stop.set()
        faulthandler.cancel_dump_traceback_later()

    # -- internals ----------------------------------------------------------

    def _write_partial(self) -> None:
        with self._lock:
            snapshot = dict(
                self.state, elapsed_s=self.elapsed(), written_unix=time.time()
            )
            target = self.out_dir / "partial_metrics.json"
            tmp = target.with_suffix(".json.tmp")
            try:
                tmp.write_text(json.dumps(snapshot, indent=2, default=str) + "\n")
                os.replace(tmp, target)
            except (
                OSError
            ) as error:  # a reentrant signal-handler write already replaced it
                print(f"[diag] partial_metrics write skipped: {error!r}", flush=True)

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            try:
                with self._lock:
                    phase = self.state["phase"]
                    in_phase = self.elapsed() - self.state["phase_started_elapsed_s"]
                self.event(
                    "heartbeat",
                    phase=phase,
                    seconds_in_phase=round(in_phase, 1),
                    host_rss_GiB=_host_rss_gib(),
                    **self.cuda_memory(),
                )
            except Exception as error:  # never let the monitor kill the run
                print(f"[diag] heartbeat failed: {error!r}", flush=True)

    def _on_signal(self, signum: int, _frame: Any) -> None:
        name = signal.Signals(signum).name
        with self._lock:
            self.state["signals_received"].append(
                {"signal": name, "elapsed_s": self.elapsed()}
            )
        if signum == signal.SIGTERM:
            self.update(status="terminated")
            self.event(
                "signal",
                signal=name,
                note="killed (time limit or scancel); state above is final",
            )
            self._progress.flush()
            self._stacks.flush()
            os._exit(143)
        self.event("signal", signal=name, note="time limit approaching; state flushed")


# -- batched-MC benchmark building blocks --------------------------------------
# Shared by benchmark_batched_pure_{kawasaki,sgc}.py: a per-block-logging
# runner, a one-width-at-a-time profiling sweep, and a budgeted production loop.


class MCBlockRunner:
    """Adapt ``sampler.run(batch, n_steps=...)`` to SimulationBatchPlanner's
    ``runner.run(batch, n_blocks=...)`` contract, one block at a time so each
    block is logged as it completes. Repeated run() calls on the same batch
    keep the sampler's energy and cumulative stats, so this is the same
    trajectory as a single run(n_steps=n_blocks * mc_steps_per_block).

    ``tracked_z`` is the atomic number whose per-replica fraction is logged
    (drifts under SGC; constant under Kawasaki, which makes it a check)."""

    def __init__(
        self,
        sampler: Any,
        mc_steps_per_block: int,
        diag: RunDiagnostics,
        record_key: str,
        tracked_z: int,
        call_labels: tuple[str, ...] = (),
        **labels: Any,
    ) -> None:
        self.sampler = sampler
        self.mc_steps_per_block = mc_steps_per_block
        self.diag = diag
        self.record_key = record_key
        self.tracked_z = tracked_z
        self.call_labels = call_labels
        self.labels = labels
        self._calls = 0
        self._last_stats = (0, 0)

    def run(self, batch: Any, n_blocks: int) -> Any:
        stage = (
            self.call_labels[self._calls]
            if self._calls < len(self.call_labels)
            else f"call{self._calls}"
        )
        self._calls += 1
        for block in range(n_blocks):
            batch = self.run_block(batch, stage=stage, block=block)
        return batch

    def run_block(self, batch: Any, **labels: Any) -> Any:
        torch.cuda.synchronize(batch.device)
        start = time.perf_counter()
        batch = self.sampler.run(batch, n_steps=self.mc_steps_per_block)
        torch.cuda.synchronize(batch.device)
        seconds = time.perf_counter() - start
        self.diag.append(
            self.record_key,
            {
                **self.labels,
                **labels,
                "seconds": seconds,
                "mc_steps_per_second": self.mc_steps_per_block / seconds,
                **self._block_observables(batch),
                **self.diag.cuda_memory(),
            },
        )
        return batch

    def _block_observables(self, batch: Any) -> dict[str, Any]:
        width = batch.num_graphs
        stats = self.sampler.stats
        attempted = stats.attempted - self._last_stats[0]
        accepted = stats.accepted - self._last_stats[1]
        self._last_stats = (stats.attempted, stats.accepted)
        fraction = (
            (batch.atomic_numbers.reshape(width, -1) == self.tracked_z)
            .double()
            .mean(dim=1)
        )
        energy = batch.energy.detach().reshape(width).double()
        finite = torch.isfinite(energy)
        e_per_atom = energy[finite] / (batch.num_nodes // width)
        return {
            "block_acceptance": accepted / attempted if attempted else None,
            "cumulative_acceptance": stats.acceptance,
            "tracked_z": self.tracked_z,
            "x_tracked_mean": fraction.mean().item(),
            "x_tracked_min": fraction.min().item(),
            "x_tracked_max": fraction.max().item(),
            "energy_eV_per_atom_mean": e_per_atom.mean().item()
            if finite.any()
            else None,
            "n_nonfinite_energies": int((~finite).sum().item()),
        }


def profile_widths_incrementally(
    planner: Any,
    workload_factory: Any,
    widths: Any,
    *,
    device: torch.device,
    warmup_blocks: int,
    measured_blocks: int,
    target_memory_fraction: float,
    diag: RunDiagnostics,
    profile_budget_seconds: float | None,
) -> tuple[list[Any], str | None]:
    """Profile ascending widths one at a time, logging each as it finishes.

    Stops at the first OOM, the first width over ``target_memory_fraction`` of
    device memory (wider cannot fit either), or once the sweep has used
    ``profile_budget_seconds``. Returns the measurements and why it stopped
    early (None if the whole ladder ran)."""
    memory_limit_bytes = int(
        torch.cuda.get_device_properties(device).total_memory * target_memory_fraction
    )
    sweep_start = time.perf_counter()
    measurements: list[Any] = []
    stop_reason = None
    for width in sorted(widths):
        if (
            profile_budget_seconds is not None
            and time.perf_counter() - sweep_start > profile_budget_seconds
        ):
            stop_reason = f"profile budget {profile_budget_seconds:.0f}s exhausted before width {width}"
            break
        diag.set_phase("profile", batch_width=width)
        width_start = time.perf_counter()
        (measurement,) = planner.profile(
            workload_factory,
            [width],
            device=device,
            warmup_blocks=warmup_blocks,
            measured_blocks=measured_blocks,
        )
        measurements.append(measurement)
        peak = measurement.peak_reserved_bytes
        diag.append(
            "profile_widths",
            {
                "batch_width": width,
                "status": measurement.status,
                "walker_blocks_per_second": measurement.walker_blocks_per_second,
                "peak_reserved_GiB": peak / _GIB if peak else None,
                "fits_memory_limit": peak is not None and peak <= memory_limit_bytes,
                "width_wall_seconds": time.perf_counter() - width_start,
                "error": (measurement.error or "")[:500] or None,
            },
        )
        if measurement.status != "ok":
            stop_reason = f"width {width} ran out of memory"
            break
        if peak is not None and peak > memory_limit_bytes:
            stop_reason = (
                f"width {width} exceeded {target_memory_fraction:.0%} of device memory"
            )
            break
    if stop_reason:
        diag.event("profile_stopped", reason=stop_reason)
    return measurements, stop_reason


def run_production_blocks(
    runner: MCBlockRunner,
    batch: Any,
    n_blocks: int,
    *,
    process_start: float,
    wall_budget_seconds: float | None,
    diag: RunDiagnostics,
) -> tuple[Any, int, str | None, float]:
    """Run up to ``n_blocks`` blocks, stopping at a block boundary once the
    process has run ``wall_budget_seconds``. Returns (batch, blocks_completed,
    truncated_reason, run_seconds)."""
    diag.set_phase("production", batch_width=batch.num_graphs, n_blocks=n_blocks)
    torch.cuda.reset_peak_memory_stats(batch.device)
    torch.cuda.synchronize(batch.device)
    run_start = time.perf_counter()
    blocks_completed = 0
    truncated_reason = None
    for block in range(n_blocks):
        if (
            wall_budget_seconds is not None
            and time.perf_counter() - process_start > wall_budget_seconds
        ):
            truncated_reason = (
                f"wall budget {wall_budget_seconds:.0f}s reached after {block} blocks"
            )
            diag.event("production_truncated", reason=truncated_reason)
            break
        batch = runner.run_block(batch, block=block)
        blocks_completed += 1
        elapsed = time.perf_counter() - run_start
        diag.update(
            production_blocks_completed=blocks_completed,
            production_eta_seconds=elapsed
            / blocks_completed
            * (n_blocks - blocks_completed),
        )
    torch.cuda.synchronize(batch.device)
    return batch, blocks_completed, truncated_reason, time.perf_counter() - run_start
