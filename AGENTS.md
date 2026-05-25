# AGENTS.md

This repo is the **trainer** for a heuristic-learning toy experiment.

Use `train.py` as the default multi-round Heuristic Learning orchestrator.
Claude Code is the optimizer for a round; it should not own or rewrite the
training loop.

## Architecture: trainer / data split

The trainer code lives here. The policy being optimized, plus all per-experiment
data, lives in a separate directory:

- **Trainer (this repo)**: `train.py`, `snake_hl/*` modules (env, eval, baselines,
  failure_report, html_replay), tests.
- **Data**: `~/Repos/snake-data/<exp-name>/`. Each experiment is a directory
  (optionally its own git repo) containing `policy.py`, `heuristic_notes.md`,
  `runs/`, `replays/`, etc.

`train.py` requires `--exp <name>`. It copies the experiment data into trainer
working slots (`snake_hl/policy.py`, `experiments/*`) before running, and copies
back to the data directory after every round.

## Goal

Improve the explicit heuristic system for single-snake play. The mutable system is:

- `policy.py` (in the data dir; surfaces inside trainer as `snake_hl/policy.py`)
- `heuristic_notes.md` (in the data dir; surfaces as `experiments/heuristic_notes.md`)

## Allowed Edits During HL Optimization

Optimization agents may edit only:

- `snake_hl/policy.py` — the heuristic system being optimized
- `experiments/heuristic_notes.md` — high-level cross-round notes
- `experiments/runs/<ts>/round-NN/journal.md` — per-round experiment log
  (train.py pre-creates the template; the optimizer appends after each experiment)
- `experiments/runs/<ts>/round-NN/scripts/` — any diagnostic scripts the
  optimizer wants to keep around (don't litter the repo root)

Do not edit:

- `snake_hl/env.py`
- `snake_hl/eval.py`
- `snake_hl/baselines.py`
- `snake_hl/html_replay.py`
- `train.py` — the orchestrator itself
- train/eval seed definitions
- score formula
- generated reports or replays, except by running repo commands
- the snapshot files train.py writes to round directories
  (`round-NN/policy.py`, `round-NN/heuristic_notes.md`, `round-NN/replays/`) —
  these are historical checkpoints, read-only from the optimizer's perspective

## Rules

- Do not hard-code specific seeds or replay paths.
- Do not inspect eval replay details while tuning. Eval is for occasional generalization checks.
- Keep the policy readable. Prefer small named helper functions over large opaque formulas.
  "Readable" does not mean "conservative" — adding new helpers to support an
  algorithmic / structural change is encouraged.
- Record useful lessons in `experiments/heuristic_notes.md`.
- Record every individual experiment (hypothesis, change, score, decision) in the
  per-round `journal.md`. This is your long-term memory across context compactions.
- Before any new experiment, search `journal.md` for the same change signature —
  don't retry what's already been recorded as failed.
- If a heuristic does not help, remove or simplify it during compression.
- For comparing across rounds, read snapshots under `experiments/runs/<ts>/round-NN/`
  rather than relying on git history.
- The trainer's git history contains only trainer-code commits. Do not consult
  trainer git history for policy hints — that history is intentionally separate
  from experiment data and may reflect a completely different lineage.

## Standard Commands

```bash
# Run training rounds against an experiment
python3 train.py --exp <exp-name> --rounds 1 --optimizer claude

# Prepare prompt without invoking the LLM (for debugging)
python3 train.py --exp <exp-name> --rounds 1 --optimizer none --dry-run

# Direct eval / failure-report on the currently loaded data slot
python3 -m snake_hl.eval --policy current --split train
python3 -m snake_hl.eval --policy current --split eval
python3 -m snake_hl.failure_report --policy current --split train --limit 5
```

Note: direct `snake_hl.eval` / `failure_report` invocations require that
`snake_hl/policy.py` is already present (i.e., a previous `train.py --exp ...`
populated the slot, or you manually copied a `policy.py` into place).
