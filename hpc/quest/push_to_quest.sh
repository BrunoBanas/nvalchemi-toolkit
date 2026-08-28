#!/usr/bin/env bash
# Push local nvalchemi-toolkit changes to GitHub, then (optionally) update
# the checkout on Quest over SSH and rebuild the environment. Run this on
# your Mac after editing the toolkit or adding a new campaign schedule.
#
#   bash hpc/quest/push_to_quest.sh "add cooling schedule for AuPd"
#
# Set QUEST_USER and QUEST_DIR to also drive the remote update automatically;
# leave them unset to just push and get the manual follow-up command printed.
#   export QUEST_USER=mnb1893
#   export QUEST_DIR=/home/mnb1893/nvalchemi-toolkit
#   export QUEST_HOST=quest.northwestern.edu   # default shown

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

if [[ $# -lt 1 ]]; then
  echo "Usage: push_to_quest.sh \"commit message\"" >&2
  exit 1
fi
MESSAGE="$1"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "[push] staging and committing local changes on $BRANCH:"
  git status --short
  git add -A
  git commit -m "$MESSAGE"
else
  echo "[push] working tree is clean, nothing to commit"
fi

git push origin "$BRANCH"

QUEST_USER="${QUEST_USER:-}"
QUEST_HOST="${QUEST_HOST:-quest.northwestern.edu}"
QUEST_DIR="${QUEST_DIR:-}"

if [[ -z "$QUEST_USER" || -z "$QUEST_DIR" ]]; then
  cat <<MANUAL_EOF
[push] pushed $BRANCH to origin. On Quest, run:
  cd <path-to-nvalchemi-toolkit-clone> && git pull && bash hpc/quest/bootstrap_env.sh

Set QUEST_USER and QUEST_DIR (and optionally QUEST_HOST) to have this
script run that for you automatically next time, for example:
  export QUEST_USER=mnb1893
  export QUEST_DIR=/home/mnb1893/nvalchemi-toolkit
MANUAL_EOF
  exit 0
fi

echo "[push] updating $QUEST_USER@$QUEST_HOST:$QUEST_DIR"
ssh "$QUEST_USER@$QUEST_HOST" \
  "cd '$QUEST_DIR' && git pull --ff-only origin '$BRANCH' && bash hpc/quest/bootstrap_env.sh"
echo "[push] Quest checkout updated and environment synced."
