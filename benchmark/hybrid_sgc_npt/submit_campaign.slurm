#!/bin/bash
# Submit the hybrid SGC-NPT efficiency campaign as a 3-task job array, one
# task per system size, each on its own single A100 40GB GPU.
#
#   sbatch benchmark/hybrid_sgc_npt/submit_campaign.slurm
#
# Edit the SBATCH directives and the "cluster-specific setup" block below for
# your site (partition/account/module names). Everything else is generic.
#
#SBATCH --job-name=hybrid_sgc_npt
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --output=benchmark/hybrid_sgc_npt/logs/%x_%A_%a.out
#SBATCH --error=benchmark/hybrid_sgc_npt/logs/%x_%A_%a.err
# #SBATCH --partition=<partition>
# #SBATCH --account=<account>

set -euo pipefail

# --- cluster-specific setup -------------------------------------------
# module load cuda/12.4
# module load uv
# --------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
mkdir -p benchmark/hybrid_sgc_npt/logs

# UMA ships in its own extras group (fairchem-core pins a newer e3nn than the
# MACE/CUDA stack); sync it once, ahead of time, into a dedicated venv:
#   UV_PROJECT_ENVIRONMENT=.venv-uma uv sync --extra uma --extra ase
export UV_PROJECT_ENVIRONMENT=.venv-uma

# The UMA checkpoint is gated on Hugging Face. Authenticate once ahead of
# time (`hf auth login`, cached under $HF_HOME) or export HF_TOKEN
# in the submitting shell's environment before `sbatch`.
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"

SIZES=(500 1372 2048)
N_ATOMS="${SIZES[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT_ROOT="$REPO_ROOT/benchmark/hybrid_sgc_npt/checkpoints"

# Set REFERENCE_ENERGIES_JSON (in the submitting shell, before sbatch) to a
# reference_energy_calibration.py output to use the calibrated, trustworthy
# delta_mu -- recommended, see run_campaign.py's module docstring
# "Chemical-potential calibration". Leave it unset to fall back to the
# literal, uncalibrated DELTA_MU_EV sweep (fine for a pure throughput
# benchmark, not for real phase-diagram production).
REFERENCE_ENERGIES_JSON="${REFERENCE_ENERGIES_JSON:-}"
ALLOW_UNRESOLVED_REFERENCE="${ALLOW_UNRESOLVED_REFERENCE:-0}"
delta_mu_args=()
if [[ -n "${REFERENCE_ENERGIES_JSON}" ]]; then
  delta_mu_args=(--reference-energies-json "${REFERENCE_ENERGIES_JSON}")
  if [[ "${ALLOW_UNRESOLVED_REFERENCE}" == "1" ]]; then
    delta_mu_args+=(--allow-unresolved-reference)
  fi
fi

echo "[task $SLURM_ARRAY_TASK_ID] n_atoms=$N_ATOMS on $(hostname), GPU=$CUDA_VISIBLE_DEVICES"

# Re-running an already-completed campaign is a no-op per state point:
# CampaignScheduler reconstructs completed run_ids from FinalStateStore on
# startup, so a --requeue'd or resubmitted task resumes automatically instead
# of redoing finished state points.
srun uv run python benchmark/hybrid_sgc_npt/run_campaign.py \
    --n-atoms "$N_ATOMS" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --device cuda \
    "${delta_mu_args[@]}"
