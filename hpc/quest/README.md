# Quest deployment

Northwestern Quest scripts for running `benchmark/hybrid_sgc_npt/run_campaign.py`
(and future hybrid MC-MD schedules) on the gengpu partition. This supersedes
the old `hybrid_uma_mc_alchemi_toolkit` wrapper package, which pulled
`nvalchemi-toolkit[uma]` from PyPI; this repo now carries the hybrid MC-MD
code (`nvalchemi.hybrid`, `nvalchemi.mc`, `nvalchemi.scheduling`) directly,
so Quest needs a clone of this fork instead.

## One-time setup

```bash
# on Quest, a login node
git clone https://github.com/BrunoBanas/nvalchemi-toolkit.git
cd nvalchemi-toolkit
export QUEST_ENV=/home/$USER/envs/nvalchemi-toolkit-uma
bash hpc/quest/bootstrap_env.sh
"$QUEST_ENV/bin/huggingface-cli" login   # once, for the gated UMA checkpoint
```

`bootstrap_env.sh` installs `uv` if needed (no module system dependency),
then runs `uv sync --extra uma --extra ase` into `QUEST_ENV`. Run it on a
login node with internet access, never inside an sbatch job -- the gengpu
compute nodes cannot reach the package indexes.

## Submitting a campaign

```bash
export QUEST_ENV=/home/$USER/envs/nvalchemi-toolkit-uma
export RUN_ROOT=/home/$USER/runs
sbatch hpc/quest/submit_hybrid_sgc_npt.sbatch
```

Three-task array (500 / 1372 / 2048 atoms), one A100 each, 48h walltime,
`--requeue` set. Checkpoints land in `$RUN_ROOT/hybrid_sgc_npt/checkpoints/`;
resubmitting the same command resumes unfinished state points automatically
(`CampaignScheduler` reconstructs completed run_ids from the checkpoint
directory on startup -- see `benchmark/hybrid_sgc_npt/README.md`).

## Iterating: local changes to a running Quest checkout

```bash
bash hpc/quest/push_to_quest.sh "widen the AuPt delta_mu grid"
```

Commits and pushes local changes to `origin/main`, then (if `QUEST_USER`
and `QUEST_DIR` are set) SSHes in, runs `git pull --ff-only`, and re-runs
`bootstrap_env.sh` so a `pyproject.toml`/dependency change is picked up
too. Without those set, it just pushes and prints the two commands to run
on Quest by hand:

```bash
export QUEST_USER=mnb1893
export QUEST_DIR=/home/mnb1893/nvalchemi-toolkit
```

A pure code/schedule change (no new dependency) only needs `git pull` on
Quest; `bootstrap_env.sh` re-running `uv sync` after that is a fast no-op.

## Adding a new schedule

A new campaign script belongs next to `run_campaign.py`, e.g.
`benchmark/<new_schedule>/run_campaign.py`, following the same
`RunSpec` / `CampaignSpec` / `CampaignScheduler` pattern. Add a matching
`hpc/quest/submit_<new_schedule>.sbatch` here (copy
`submit_hybrid_sgc_npt.sbatch` and change the array size, sizes list, and
script path) -- `bootstrap_env.sh` and `push_to_quest.sh` are shared across
every schedule.
