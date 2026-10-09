#!/usr/bin/env bash
# UMA efficiency matrix: time and peak memory of nvalchemi MC / MD / hybrid
# workloads (500-atom Au-Pt, 1200 K unless a kernel says otherwise) under
# several inference settings at forced batch widths, all in ONE allocation so
# node-to-node variation (~5%) cannot leak into the comparison. The results
# calibrate benchmark/uma_efficiency/calibration.json (see README.md there).
#
# Run inside a GPU allocation, from the repository root:
#
#   KERNELS="kawasaki sgc hybrid" bash benchmark/uma_efficiency/run_efficiency_matrix.sh
#   sbatch --gres=gpu:1 --time=06:00:00 --export=ALL,KERNELS="npt+kawasaki_npt" \
#       benchmark/uma_efficiency/run_efficiency_matrix.sh
#
# Slurm runs a copy of this file, so submit from the repository root or set
# NVALCHEMI_TOOLKIT_DIR. Several kernels in one job share one GPU reservation.
#
# Kernels (configs; widths 1 2 4 unless noted):
#   kawasaki        pure Kawasaki MC: baseline (turbo, species-blind, full forces),
#                   best (turbo, energy-only), merge_nocompile (energy-only)
#   sgc             pure SGC MC: baseline ("batch"), best (no merge/compile/ckpt,
#                   tf32, energy-only)
#   hybrid          SGC-NPT: baseline ("batch", full outputs), best (energy-only MC)
#   kawasaki_npt    Kawasaki-NPT: baseline, merge_nocompile, turbo_energy_only,
#                   nomerge_nocompile
#   npt             pure NPT (--md-only): turbo, merge/no-merge, checkpointing, batch
#   kawasaki_npt_settings / sgc_npt_settings
#                   the NPT settings inside each hybrid
#   kawasaki_wide / sgc_wide / npt_wide / kawasaki_npt_wide
#                   recommended settings at widths 4 8 12
#   sgc_compile     SGC MC, compile on vs off, widths 1 and 4
#   sgc_npt_ckpt    SGC-NPT with checkpointing + energy-only MC, widths 1 2 4
#   sgc_npt_108 / sgc_npt_256
#                   SGC-NPT on 108 / 256-atom cells, widths 1 4 8 (16)
#   kawasaki_2048   Kawasaki MC turbo vs merge_nocompile, widths 1 and 2
#   npt_2048 / kawasaki_npt_2048 / sgc_npt_2048
#                   width 1, fast settings vs activation checkpointing
#   kawasaki_compile_shapes
#                   Kawasaki MC turbo, static vs dynamic compile shapes, widths 1 4
#   npt_compile / kawasaki_npt_compile / sgc_npt_compile
#                   compiled MD with dynamic shapes (turbo; compile w/o merge) vs
#                   the eager best, widths 1 4 (UMAWrapper keeps fairchem's dynamic
#                   compile single-process since 2026-09-30)
#   npt_compile_recheck / kawasaki_npt_compile_recheck
#                   npt_compile / kawasaki_npt_compile at width 1 (Kawasaki-NPT also 4)
#                   with TORCH_LOGS=recompiles on every compiled case and versions.txt
#                   recording the code that ran; npt adds turbo_static, to test whether
#                   static shapes alone reproduce the MD recompiles.
#   Config names ending in _static / _dynamic set NVALCHEMI_UMA_COMPILE_SHAPES.
#   mole_profile    profile_mole_overhead.py
#
# Environment:
#   KERNELS        space- or '+'-separated kernel list (KERNEL=... also works)
#   OUTPUT_ROOT    results go to $OUTPUT_ROOT/<RUN_TAG>/<kernel>/<config>_w<width>/
#                  (default ./efficiency_matrix)
#   RUN_TAG        default: UTC timestamp
#   PYTHON         interpreter with nvalchemi + fairchem (default: python)
#   EXPECT_GPU     substring the GPU name must contain, e.g. A100-SXM4-80GB; the
#                  job exits 3 on any other card. Set it for comparisons: a
#                  generic "a100" GPU type can also match 40 GB PCIe cards, and
#                  a faster card swamps the effect being measured.
#   WIDTHS         default widths (1 2 4); CASE_TIMEOUT per case (default 3h)
#
#SBATCH --job-name=uma-efficiency-matrix
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
KERNELS="${KERNELS:-${KERNEL:-}}"
KERNELS="${KERNELS//+/ }"
: "${KERNELS:?KERNELS (space-separated) or KERNEL: see the kernel list above}"
TOOLKIT_DIR="${NVALCHEMI_TOOLKIT_DIR:-${PWD}}"
TESTS="${TOOLKIT_DIR}/benchmark/uma_efficiency"
[[ -f "${TESTS}/plan_run.py" ]] || { echo "run from the nvalchemi-toolkit root or set NVALCHEMI_TOOLKIT_DIR" >&2; exit 2; }
PYTHON="${PYTHON:-python}"
RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
EXPECT_GPU="${EXPECT_GPU:-}"
WIDTHS="${WIDTHS:-1 2 4}"
SGC_BEST="compile=false,merge_mole=false,tf32=true,activation_checkpointing=false"
MERGE_NOCOMPILE="compile=false,merge_mole=true,tf32=true,activation_checkpointing=false"
CASE_TIMEOUT="${CASE_TIMEOUT:-3h}"
JOB_ID="${SLURM_JOB_ID:-local}"
THREADS="${SLURM_CPUS_PER_TASK:-8}"
stamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }

base="${OUTPUT_ROOT:-${PWD}/efficiency_matrix}/${RUN_TAG}"
mkdir -p "${base}"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
gpu_info="$(printf '%s\nhost=%s\njob=%s' "${gpu_name}" "$(hostname)" "${JOB_ID}")"
if [[ -n "${EXPECT_GPU}" && "${gpu_name}" != *"${EXPECT_GPU}"* ]]; then
  echo "landed on '${gpu_name}' ($(hostname)), expected ${EXPECT_GPU}; refusing to run" >&2
  exit 3
fi

gpu_args=()
if [[ "${CUDA_VISIBLE_DEVICES:-}" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  gpu_args=(-i "${CUDA_VISIBLE_DEVICES%%,*}")
fi
nvidia-smi "${gpu_args[@]}" --query-gpu=timestamp,memory.used,utilization.gpu,power.draw \
  --format=csv,nounits -l 5 > "${base}/gpu_monitor_${JOB_ID}.csv" 2>&1 &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT
echo "RUN_TAG=${RUN_TAG}  results: ${base}"

for KERNEL in ${KERNELS}; do
root="${base}/${KERNEL}"
mkdir -p "${root}"
printf '%s\n' "${gpu_info}" > "${root}/gpu.txt"
# Which code ran: versions, the nvalchemi source and commit, and whether UMAWrapper
# has the compile_shapes option.
"${PYTHON}" - > "${root}/versions.txt" 2>&1 <<'PY' || true
import pathlib, subprocess, torch, fairchem.core, nvalchemi
from nvalchemi.models.uma import UMAWrapper
src = pathlib.Path(nvalchemi.__file__).resolve().parent
commit = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(src), "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True).stdout.strip() or "unknown"
print(f"torch={torch.__version__}\nfairchem_core={fairchem.core.__version__}\nnvalchemi_src={src}\n"
      f"nvalchemi_commit={commit}\numa_compile_shapes_option={hasattr(UMAWrapper, '_static_compile')}")
PY
widths="${WIDTHS}"
echo "$(stamp) === kernel ${KERNEL} ==="

if [[ "${KERNEL}" == mole_profile ]]; then
  set +e
  /usr/bin/time -f 'wall_seconds=%e\nmax_rss_kib=%M\nexit_status=%x' -o "${root}/time.txt" \
    timeout --kill-after=300 "${CASE_TIMEOUT}" \
    env PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
      OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS=1 \
      "${PYTHON}" "${TESTS}/profile_mole_overhead.py" --output-dir "${root}" \
      > "${root}/stdout.log" 2> "${root}/stderr.log"
  status=$?
  set -e
  printf 'exit_status=%s\n' "${status}" > "${root}/status.txt"
  echo "  -> exit ${status}"
  continue
fi

case "${KERNEL}" in
  kawasaki)
    script="${TESTS}/benchmark_batched_pure_kawasaki.py"
    common=(--n-atoms 500 --n-blocks 100 --temperature-k 1200.0 --cutoff-angstrom 3.40
            --composition-seed 2026090102)
    width_flag=--batch-width
    configs=(baseline best merge_nocompile)
    config_args() {
      case "$1" in
        baseline) echo "--inference-settings turbo --species-blind-proposals" ;;
        best) echo "--inference-settings turbo --energy-only" ;;
        merge_nocompile) echo "--inference-settings ${MERGE_NOCOMPILE} --energy-only" ;;
      esac
    } ;;
  sgc)
    script="${TESTS}/benchmark_batched_pure_sgc.py"
    common=(--n-atoms 500 --n-blocks 20 --temperature-k 1200.0 --delta-mu-ev 0.5
            --composition-seed 2026090102)
    width_flag=--batch-width
    configs=(baseline best)
    config_args() {
      case "$1" in
        baseline) echo "--inference-settings batch" ;;
        best) echo "--inference-settings ${SGC_BEST} --energy-only" ;;
      esac
    } ;;
  hybrid)
    script="${TESTS}/benchmark_hybrid_sgc_npt_single_point.py"
    common=(--n-atoms 500 --n-blocks 20 --temperature-k 1200.0 --delta-mu-ev 0.5)
    width_flag=--n-walkers
    configs=(baseline best)
    config_args() {
      case "$1" in
        baseline) echo "--inference-settings batch" ;;
        best) echo "--inference-settings ${SGC_BEST} --mc-energy-only" ;;
      esac
    } ;;
  kawasaki_npt)
    script="${TESTS}/benchmark_hybrid_sgc_npt_single_point.py"
    common=(--sampler kawasaki --n-atoms 500 --n-blocks 20 --temperature-k 1200.0 --kawasaki-cutoff-angstrom 3.40)
    width_flag=--n-walkers
    configs=(baseline merge_nocompile turbo_energy_only nomerge_nocompile)
    config_args() {
      case "$1" in
        baseline) echo "--inference-settings turbo --species-blind-proposals" ;;
        merge_nocompile) echo "--inference-settings ${MERGE_NOCOMPILE} --mc-energy-only" ;;
        turbo_energy_only) echo "--inference-settings turbo --mc-energy-only" ;;
        nomerge_nocompile) echo "--inference-settings ${SGC_BEST} --mc-energy-only" ;;
      esac
    } ;;
  npt|kawasaki_npt_settings|sgc_npt_settings)
    script="${TESTS}/benchmark_hybrid_sgc_npt_single_point.py"
    width_flag=--n-walkers
    case "${KERNEL}" in
      npt)
        common=(--md-only --n-atoms 500 --n-blocks 20 --temperature-k 1200.0)
        configs=(turbo merge_nocompile nomerge_nocompile merge_nocompile_ckpt batch)
        mc_flag=() ;;
      kawasaki_npt_settings)
        common=(--sampler kawasaki --n-atoms 500 --n-blocks 20 --temperature-k 1200.0 --kawasaki-cutoff-angstrom 3.40)
        configs=(turbo merge_nocompile nomerge_nocompile merge_nocompile_ckpt batch)
        mc_flag=(--mc-energy-only) ;;
      sgc_npt_settings)
        common=(--n-atoms 500 --n-blocks 20 --temperature-k 1200.0 --delta-mu-ev 0.5)
        configs=(nomerge_nocompile nomerge_nocompile_ckpt batch)
        mc_flag=(--mc-energy-only) ;;
    esac
    config_args() {
      local spec
      case "$1" in
        turbo) spec=turbo ;;
        merge_nocompile) spec="${MERGE_NOCOMPILE}" ;;
        nomerge_nocompile) spec="${SGC_BEST}" ;;
        merge_nocompile_ckpt) spec="compile=false,merge_mole=true,tf32=true,activation_checkpointing=true" ;;
        nomerge_nocompile_ckpt) spec="compile=false,merge_mole=false,tf32=true,activation_checkpointing=true" ;;
        batch) spec=batch ;;
      esac
      echo "--inference-settings ${spec} ${mc_flag[*]:-}"
    } ;;
  kawasaki_wide|kawasaki_2048|kawasaki_compile_shapes)
    script="${TESTS}/benchmark_batched_pure_kawasaki.py"
    width_flag=--batch-width
    if [[ "${KERNEL}" == kawasaki_wide ]]; then
      common=(--n-atoms 500 --n-blocks 100 --temperature-k 1200.0 --cutoff-angstrom 3.40 --composition-seed 2026090102)
      configs=(best_static); widths="4 8 12"
    elif [[ "${KERNEL}" == kawasaki_2048 ]]; then
      common=(--n-atoms 2048 --n-blocks 10 --temperature-k 1200.0 --cutoff-angstrom 3.40 --composition-seed 2026090102)
      configs=(best_static merge_nocompile); widths="1 2"
    else
      common=(--n-atoms 500 --n-blocks 100 --temperature-k 1200.0 --cutoff-angstrom 3.40 --composition-seed 2026090102)
      configs=(best_static best_dynamic); widths="1 4"
    fi
    config_args() {
      case "$1" in
        best) echo "--inference-settings turbo --energy-only" ;;
        merge_nocompile) echo "--inference-settings ${MERGE_NOCOMPILE} --energy-only" ;;
      esac
    } ;;
  sgc_wide|sgc_compile)
    script="${TESTS}/benchmark_batched_pure_sgc.py"
    common=(--n-atoms 500 --n-blocks 20 --temperature-k 1200.0 --delta-mu-ev 0.5 --composition-seed 2026090102)
    width_flag=--batch-width
    if [[ "${KERNEL}" == sgc_wide ]]; then
      configs=(best); widths="4 8 12"
    else
      configs=(best compile_energy_only_static compile_energy_only_dynamic); widths="1 4"
    fi
    config_args() {
      case "$1" in
        best) echo "--inference-settings ${SGC_BEST} --energy-only" ;;
        compile_energy_only) echo "--inference-settings compile=true,merge_mole=false,tf32=true,activation_checkpointing=false --energy-only" ;;
      esac
    } ;;
  npt_wide|kawasaki_npt_wide|sgc_npt_ckpt|sgc_npt_108|sgc_npt_256|npt_2048|kawasaki_npt_2048|sgc_npt_2048|npt_compile|kawasaki_npt_compile|sgc_npt_compile|npt_compile_recheck|kawasaki_npt_compile_recheck)
    script="${TESTS}/benchmark_hybrid_sgc_npt_single_point.py"
    width_flag=--n-walkers
    sgc_common=(--n-blocks 20 --temperature-k 1200.0 --delta-mu-ev 0.5)
    kaw_common=(--sampler kawasaki --temperature-k 1200.0 --kawasaki-cutoff-angstrom 3.40)
    mc_flag=(--mc-energy-only)
    case "${KERNEL}" in
      npt_wide)
        common=(--md-only --n-atoms 500 --n-blocks 20 --temperature-k 1200.0); mc_flag=()
        configs=(merge_nocompile); widths="4 8 12" ;;
      kawasaki_npt_wide)
        common=("${kaw_common[@]}" --n-atoms 500 --n-blocks 20)
        configs=(merge_nocompile); widths="4 8 12" ;;
      sgc_npt_ckpt)
        common=(--n-atoms 500 "${sgc_common[@]}")
        configs=(nomerge_nocompile_ckpt); widths="1 2 4" ;;
      sgc_npt_108)
        common=(--n-atoms 108 "${sgc_common[@]}")
        configs=(nomerge_nocompile); widths="1 4 8 16" ;;
      sgc_npt_256)
        common=(--n-atoms 256 "${sgc_common[@]}")
        configs=(nomerge_nocompile); widths="1 4 8" ;;
      npt_2048)
        common=(--md-only --n-atoms 2048 --n-blocks 10 --temperature-k 1200.0); mc_flag=()
        configs=(merge_nocompile merge_nocompile_ckpt nomerge_nocompile); widths="1" ;;
      kawasaki_npt_2048)
        common=("${kaw_common[@]}" --n-atoms 2048 --n-blocks 10)
        configs=(merge_nocompile merge_nocompile_ckpt); widths="1" ;;
      sgc_npt_2048)
        common=(--n-atoms 2048 --n-blocks 10 --temperature-k 1200.0 --delta-mu-ev 0.5)
        configs=(nomerge_nocompile_ckpt nomerge_nocompile); widths="1" ;;
      npt_compile)
        common=(--md-only --n-atoms 500 --n-blocks 20 --temperature-k 1200.0); mc_flag=()
        configs=(merge_nocompile turbo_dynamic compile_nomerge_dynamic); widths="1 4" ;;
      kawasaki_npt_compile)
        common=("${kaw_common[@]}" --n-atoms 500 --n-blocks 20)
        configs=(merge_nocompile turbo_dynamic); widths="1 4" ;;
      sgc_npt_compile)
        common=(--n-atoms 500 "${sgc_common[@]}")
        configs=(nomerge_nocompile compile_nomerge_dynamic); widths="1 4" ;;
      npt_compile_recheck)
        common=(--md-only --n-atoms 500 --n-blocks 20 --temperature-k 1200.0); mc_flag=()
        configs=(merge_nocompile turbo_dynamic turbo_static); widths="1" ;;
      kawasaki_npt_compile_recheck)
        common=("${kaw_common[@]}" --n-atoms 500 --n-blocks 20)
        configs=(merge_nocompile turbo_dynamic); widths="1 4" ;;
    esac
    config_args() {
      local spec
      case "$1" in
        merge_nocompile) spec="${MERGE_NOCOMPILE}" ;;
        nomerge_nocompile) spec="${SGC_BEST}" ;;
        merge_nocompile_ckpt) spec="compile=false,merge_mole=true,tf32=true,activation_checkpointing=true" ;;
        nomerge_nocompile_ckpt) spec="compile=false,merge_mole=false,tf32=true,activation_checkpointing=true" ;;
        turbo) spec=turbo ;;
        compile_nomerge) spec="compile=true,merge_mole=false,tf32=true,activation_checkpointing=false" ;;
      esac
      echo "--inference-settings ${spec} ${mc_flag[*]:-}"
    } ;;
  *) echo "unknown KERNEL=${KERNEL}; skipping" >&2; continue ;;
esac

for config in "${configs[@]}"; do
  for width in ${widths}; do
    case_dir="${root}/${config}_w${width}"
    mkdir -p "${case_dir}/run"
    config_base="${config%_static}"; config_base="${config_base%_dynamic}"
    shape_env=()
    case "${config}" in
      *_static) shape_env=(NVALCHEMI_UMA_COMPILE_SHAPES=static) ;;
      *_dynamic) shape_env=(NVALCHEMI_UMA_COMPILE_SHAPES=dynamic) ;;
    esac
    case "${KERNEL}:${config}" in
      *_recheck:turbo*|*_recheck:compile*) shape_env+=(TORCH_LOGS=recompiles) ;;
    esac
    read -r -a extra <<< "$(config_args "${config_base}")"
    echo "$(stamp) ${KERNEL} ${config} width=${width}"
    set +e
    /usr/bin/time -f 'wall_seconds=%e\nmax_rss_kib=%M\nexit_status=%x' -o "${case_dir}/time.txt" \
      timeout --kill-after=300 "${CASE_TIMEOUT}" \
      env ${shape_env[@]+"${shape_env[@]}"} PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 \
        OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS=1 \
        "${PYTHON}" "${script}" \
        "${common[@]}" "${width_flag}" "${width}" "${extra[@]}" --device cuda \
        --output "${case_dir}/run/metrics.json" \
        > "${case_dir}/stdout.log" 2> "${case_dir}/stderr.log"
    status=$?
    set -e
    printf 'exit_status=%s\n' "${status}" > "${case_dir}/status.txt"
    echo "  -> exit ${status}"
  done
done
echo "efficiency matrix (${KERNEL}) written under ${root}"
done
