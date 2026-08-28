#!/usr/bin/env bash
# Install or update the nvalchemi-toolkit UMA environment on Quest.
#
# Run this on a Quest LOGIN node (or an interactive srun --pty session
# with internet access) -- never inside an sbatch job. The gengpu compute
# nodes have no outbound internet, so package downloads must happen here
# first; submit_hybrid_sgc_npt.sbatch only ever runs the environment this
# script already built.
#
#   export QUEST_ENV=/home/$USER/envs/nvalchemi-toolkit-uma
#   bash hpc/quest/bootstrap_env.sh
#
# Safe to re-run: uv sync is a no-op when the lockfile/extras did not
# change, so this doubles as the pick-up-new-dependencies step after a
# git pull that touched pyproject.toml/uv.lock. Run it again any time
# after pulling -- push_to_quest.sh does this automatically.

set -euo pipefail

: "${QUEST_ENV:?Set QUEST_ENV to the dedicated UMA environment path, e.g. /home/\$USER/envs/nvalchemi-toolkit-uma}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

# --- cluster-specific setup -------------------------------------------
# uv provisions its own Python interpreter and pulls prebuilt CUDA wheels
# from the pytorch-cu126/cu130 indexes (see pyproject.toml), so neither a
# module-load of python nor a system CUDA toolkit is required just to
# build this environment. Uncomment if your site needs something extra
# (e.g. a module that puts a working curl/CA bundle on PATH).
# module purge
# module load cuda/12.6
# ------------------------------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then
  echo "[bootstrap] uv not found; installing to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

export UV_PROJECT_ENVIRONMENT="$QUEST_ENV"
uv sync --extra uma --extra ase

# The UMA checkpoint is gated on Hugging Face. Authenticate once ahead of
# time and cache the token where jobs can find it:
#   "$QUEST_ENV/bin/huggingface-cli" login
# or export HF_TOKEN in the submitting shell before sbatch.

"$QUEST_ENV/bin/python" "$SCRIPT_DIR/verify_env.py"

echo "[bootstrap] environment ready at $QUEST_ENV"
