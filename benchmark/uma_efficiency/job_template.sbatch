#!/usr/bin/env bash
# __JOB_DESCRIPTION__
#
# Planned with benchmark/uma_efficiency/plan_run.py: __PLAN_SUMMARY__
#
# Template for one GPU job running an nvalchemi + UMA driver (for example a copy
# of benchmark/uma_efficiency/driver_template.py). Fill every __PLACEHOLDER__;
# add --account / --partition on the sbatch command line or as #SBATCH lines for
# your site. Keep the structure: the GPU guard, memory monitor, /usr/bin/time
# record, USR1 forwarding and status file are what make failed and timed-out
# jobs diagnosable.
#
# GPU request: production and physics runs can take any card (shortest queue);
# size the batch width for the smallest card and --time for the slowest.
# Speed / memory / batching comparisons must pin one card model: request it by
# type, exclude look-alikes (a generic "a100" type can match 40 GB PCIe and
# 80 GB SXM cards), and set EXPECT_GPU so a misplaced job exits instead of
# producing a number that is silently not comparable.
#
#SBATCH --job-name=__JOB_NAME__
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=__TIME_LIMIT__
# Ahead of the limit, bash gets USR1 (trapped below and forwarded to Python) so
# the run can flush state before Slurm's SIGTERM.
#SBATCH --signal=B:USR1@900

set -euo pipefail
: "${OUTPUT_ROOT:?persistent output directory}"
PYTHON="${PYTHON:-python}"
RUN_TAG="${RUN_TAG:-${SLURM_JOB_ID:-local}}"
EXPECT_GPU="${EXPECT_GPU:-}"   # e.g. A100-SXM4-80GB for comparisons; empty = any card
# One shared model cache: per-run caches duplicate multi-GB checkpoints.
export HF_HOME="${HF_HOME:-${OUTPUT_ROOT}/cache}"

case_root="${OUTPUT_ROOT}/__CAMPAIGN__/${RUN_TAG}"
mkdir -p "${case_root}/run"

gpu_name="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
printf '%s\nhost=%s\njob=%s\n' "${gpu_name}" "$(hostname)" "${SLURM_JOB_ID:-local}" > "${case_root}/gpu.txt"
if [[ -n "${EXPECT_GPU}" && "${gpu_name}" != *"${EXPECT_GPU}"* ]]; then
  echo "landed on '${gpu_name}' ($(hostname)), expected ${EXPECT_GPU}; refusing to run" >&2
  exit 3
fi

gpu_args=()
if [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  gpu_args=(-i "${CUDA_VISIBLE_DEVICES%%,*}")
fi
nvidia-smi "${gpu_args[@]}" --query-gpu=timestamp,memory.used,utilization.gpu,power.draw \
  --format=csv,nounits -l 5 > "${case_root}/gpu_monitor.csv" 2>&1 &
monitor_pid=$!

forward_usr1() {
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SIGUSR1: time limit approaching, forwarding to the run" >&2
  pkill -USR1 -P "${run_pid}" 2>/dev/null || true
}
trap forward_usr1 USR1

set +e
/usr/bin/time -f 'wall_seconds=%e\nmax_rss_kib=%M\nexit_status=%x' -o "${case_root}/time.txt" \
  env PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
    OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS=1 \
    "${PYTHON}" __DRIVER__ \
    __DRIVER_ARGS__ \
    --output "${case_root}/run/metrics.json" \
    > "${case_root}/stdout.log" 2> "${case_root}/stderr.log" &
run_pid=$!
# bash runs the USR1 trap only between commands; `wait` returns early when it
# fires, so keep waiting until the run itself exits.
while true; do
  wait "${run_pid}"
  status=$?
  kill -0 "${run_pid}" 2>/dev/null || break
done
set -e
kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true
printf 'exit_status=%s\n' "${status}" > "${case_root}/status.txt"
if [[ "${status}" != 0 ]]; then
  echo "__JOB_NAME__ failed (exit ${status}); see ${case_root}" >&2
  exit "${status}"
fi
echo "__JOB_NAME__ finished: ${case_root}/run/metrics.json"
