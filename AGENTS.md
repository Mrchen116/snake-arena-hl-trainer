# AGENTS.md

This repo is the **trainer** for a heuristic-learning toy experiment.

Use `train.py` as the default multi-round Heuristic Learning orchestrator.
Claude Code is the optimizer for a round; it should not own or rewrite the
training loop.

## Architecture: trainer / data

Trainer code (this repo) and per-experiment data co-exist in this checkout
but are cleanly separated:

- **Trainer code** (committed to git): `train.py`, `snake_hl/*` modules (env,
  eval, baselines, failure_report, policy_runtime, replay, html_replay), tests.
- **Data** (gitignored): `experiments/<exp-name>/` holds `policy.py`,
  `heuristic_notes.md`, `runs/`, `replays/`, `reports/`, etc. for one
  experiment. Optionally has its own `.git/` for experiment-internal versioning.

There is **no global policy slot**. Each experiment's `policy.py` lives
under `experiments/<exp>/` and is both the canonical state and the runtime
file. `train.py` sets `SNAKE_POLICY_PATH=experiments/<exp>/policy.py` in
the environment before any `snake_hl.*` module is imported; the
`policy_runtime` loader reads that path. This means multiple `train.py`
processes can run in parallel against different experiments without
clobbering each other.

The optimizer is launched with `cwd = experiments/<exp>/`. From the
optimizer's perspective `policy.py` is just a file in the current directory
— there is no separate trainer-side path to keep in sync.

## Allowed Edits During HL Optimization

Optimization agents may edit only files inside their experiment directory:

- `policy.py` (cwd-relative) — the file you edit during optimization.
- `heuristic_notes.md` (cwd-relative) — high-level cross-round notes.
- `runs/<ts>/round-NN/journal.md` (cwd-relative) — per-round experiment log
  (pre-created by train.py; append rows after each experiment).
- `runs/<ts>/round-NN/scripts/` (cwd-relative) — diagnostic scripts you want
  to keep around.

Do not edit:

- Anything under `<repo-root>/snake_hl/` (trainer modules)
- `<repo-root>/train.py`, `<repo-root>/AGENTS.md`, `<repo-root>/README.md`
- train/eval seed definitions
- score formula
- generated reports or replays, except by running repo commands

Trainer paths are blocked at the permission + sandbox layers, so attempts to
edit them will fail at tool-invocation time.

## Rules

- Do not hard-code specific seeds or replay paths.
- Do not inspect eval replay details while tuning. Eval is for occasional
  generalization checks.
- Keep the policy readable. Prefer small named helper functions over large
  opaque formulas. "Readable" does not mean "conservative" — adding new
  helpers to support an algorithmic / structural change is encouraged.
- Record useful lessons in `heuristic_notes.md`.
- Record every individual experiment (hypothesis, change, score, decision) in
  the per-round `journal.md`. This is your long-term memory across context
  compactions.
- Before any new experiment, search `journal.md` for the same change signature
  — don't retry what's already been recorded as failed.
- If a heuristic does not help, remove or simplify it during compression.
- For comparing across rounds, read snapshots under `runs/<ts>/round-NN/`
  rather than relying on git history.
- The trainer's git history contains only trainer-code commits. Do not consult
  trainer git history for policy hints — that history is intentionally
  separate from experiment data.

## Standard Commands

`train.py` runs from the trainer root and operates against one experiment at
a time:

```bash
# Run training rounds
python3 train.py --exp <exp-name> --rounds 1 --optimizer claude

# Fork from an existing experiment in one shot
python3 train.py --exp <new-exp> --new-from <base-exp> --rounds 5 --optimizer claude

# Dry-run: build prompt + journal template without invoking the LLM
python3 train.py --exp <exp-name> --rounds 1 --optimizer none --dry-run
```

From within an experiment dir (typical for the optimizer subprocess), direct
module invocations work because the trainer venv has `snake_hl` installed
and `train.py` already exported `SNAKE_POLICY_PATH` to the subprocess env:

```bash
.venv/bin/python -m snake_hl.eval --policy current --split train
.venv/bin/python -m snake_hl.eval --policy current --split eval
.venv/bin/python -m snake_hl.failure_report --policy current --split train --limit 5
```

If you ever run these tools outside the trainer (i.e. not via `train.py`),
you need to export `SNAKE_POLICY_PATH=<absolute-path-to-policy.py>` first or
they will error out.
