#!/bin/bash
# Submit the hybrid SGC-NPT AuPt coarse-schedule campaign on Quest.
# One task per system size (500 / 1372 / 2048 atoms), one A100 40GB GPU
# each, on the gengpu partition under allocation p33063. Requeuing or
# resubmitting resumes automatically: CampaignScheduler reconstructs
# completed run_ids from FinalStateStore on startup (see
# benchmark/hybrid_sgc_npt/README.md).
#
#   export QUEST_ENV=/home/$USER/envs/nvalchemi-toolkit-uma
#   export RUN_ROOT=/home/$USER/runs
#   sbatch hpc/quest/submit_hybrid_sgc_npt.sbatch
#
# Run hpc/quest/bootstrap_env.sh first (on a login node, not in this job)
# to build QUEST_ENV.
#
#SBATCH --job-name=hybrid_sgc_npt
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --partition=gengpu
#SBATCH --account=p33063
#SBATCH --requeue
#SBATCH --output=hpc/quest/logs/%x_%A_%a.out
#SBATCH --error=hpc/quest/logs/%x_%A_%a.err

set -euo pipefail

: "${QUEST_ENV:?Set QUEST_ENV to the environment built by bootstrap_env.sh, e.g. /home/\$USER/envs/nvalchemi-toolkit-uma}"
: "${RUN_ROOT:?Set RUN_ROOT to a directory for campaign checkpoints, e.g. /home/\$USER/runs}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p hpc/quest/logs

# The UMA checkpoint is gated on Hugging Face. huggingface-cli login caches
# the token under the standard ~/.cache/huggingface by default, so this does
# not override HF_HOME unless you already set it yourself (e.g. because you
# logged in against a non-default HF_HOME). If your existing login lives
# somewhere else, export HF_HOME to match before you sbatch this.

SIZES=(500 1372 2048)
N_ATOMS="${SIZES[$SLURM_ARRAY_TASK_ID]}"
CHECKPOINT_ROOT="$RUN_ROOT/hybrid_sgc_npt/checkpoints"

echo "[task $SLURM_ARRAY_TASK_ID] n_atoms=$N_ATOMS on $(hostname), GPU=$CUDA_VISIBLE_DEVICES"

"$QUEST_ENV/bin/python" benchmark/hybrid_sgc_npt/run_campaign.py \
    --n-atoms "$N_ATOMS" \
    --checkpoint-root "$CHECKPOINT_ROOT" \
    --device cuda
