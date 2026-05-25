# Snake Arena HL

A tiny single-snake heuristic-learning arena. The goal is to let a coding agent
iterate on a readable rule system and improve it with fast, hard feedback.

## Reference

This project is a small toy experiment inspired by Jiayi Weng's heuristic
learning write-up:

- Article: <https://trinkle23897.github.io/learning-beyond-gradients/>
- Artifacts: <https://github.com/Trinkle23897/learning-beyond-gradients>

The point is not to train a neural network. The point is to test whether a
coding agent can improve an explicit heuristic system by repeatedly observing
failures, editing code/notes, and checking a hard reward.

## Overall Idea

The experiment maps the heuristic-learning loop onto a small Snake game:

- Environment: deterministic single-snake arena.
- Heuristic system: `policy.py` (the model) plus `heuristic_notes.md`
  (cross-round notes). Both live in a separate **data directory** (see below),
  not in this repository.
- Agent: later, Claude Code CLI or another coding agent.
- Feedback: train score, death reasons, failure report, and replay JSON opened
  through one reusable HTML viewer.
- Reward: food eaten, survival steps, and whether the snake survives the episode.
- Generalization check: train seeds versus held-out eval seeds.

The agent should not modify the environment or scoring. It should only improve
the readable policy and maintain notes about what worked, what failed, and what
should be compressed away.

## Trainer / data split

This repository contains the **trainer** code only:

- `train.py` — multi-round orchestrator
- `snake_hl/*` — environment, evaluator, baselines, failure-report tooling
- `tests/`

The policy being optimized, plus all per-experiment data (notes, runs, replays,
training curve, trials log), lives under `~/Repos/snake-data/<exp-name>/`. Each
experiment is its own directory; you can `cp -r` between them to fork lineages,
move them out of `~/Repos/snake-data/` to archive, etc.

When you run `train.py --exp <name>`, it copies the experiment data into trainer
working slots, runs the rounds, and copies modified files back after every round.
The trainer never auto-commits — versioning of experiment data is up to you,
done inside each `~/Repos/snake-data/<name>/`.

## Plan

1. Build the minimal arena and baseline policy.
   - Single snake, fixed board size, deterministic seeds, no internal walls.
   - Keep the current policy intentionally weak so improvement is visible.

2. Build human-readable feedback.
   - Generate episode JSON for individual seeds.
   - Keep one reusable HTML replay viewer instead of copying the UI into every
     episode artifact.
   - Generate `experiments/reports/train-failures.md` with the worst train
     episodes and replay links.
   - Generate a baseline snapshot for current, eval, random, and safe-greedy.

3. Lock the optimization boundary.
   - Use `AGENTS.md` and `EVAL_PROTOCOL.md` to define what an optimization
     agent may edit.
   - Only `snake_hl/policy.py` and `experiments/heuristic_notes.md` should
     change during heuristic-learning rounds.

4. Run the first coding-agent iteration.
   - Feed Claude Code CLI the failure report, linked HTML replays, notes, and
     policy file.
   - Ask it to improve train score without hard-coding seeds or touching the
     evaluator.
   - Re-run train and then eval to see whether the change generalizes.

5. Repeat and compress.
   - Keep useful heuristics.
   - Remove rules that overfit or do not improve score.
   - Periodically simplify the policy and notes so the heuristic system does
     not become a pile of special cases.

## Scope

Version 0 is intentionally small:

- Single snake on a 12x12 board
- One food item at a time
- No internal walls
- Death on wall or self collision
- 300-step episode cap
- Text replay and reusable HTML replay viewer

The heuristic-learning target is `snake_hl/policy.py`. The environment and
evaluation code should stay fixed while optimizing a policy.

## Quick Start

```bash
# Install trainer
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest

# Tell train.py where your experiments live. Put this in ~/.zshrc or ~/.bashrc.
# train.py refuses to start without it (and deliberately doesn't hardcode a
# default — that would leak the path to the optimizer subprocess).
export SNAKE_DATA_HOME=/path/to/snake-data

# Fork + run in one command (creates <new-exp-name> from r14-baseline,
# then trains 5 rounds).
python3 train.py --exp <new-exp-name> --new-from r14-baseline --rounds 5 --optimizer claude

# Subsequent runs against the same experiment
python3 train.py --exp <new-exp-name> --rounds 5 --optimizer claude

# Inspect the loaded experiment's working slots directly
python3 -m snake_hl.eval --policy current --split train
python3 -m snake_hl.failure_report --policy current --split train --limit 5
```

`snake_hl.eval` and other direct module invocations operate on whatever is
currently in the trainer's working slots (`snake_hl/policy.py`,
`experiments/*`). Those slots are populated by the most recent `train.py --exp ...`
copy-in.

`snake_hl.html_replay` writes two artifacts:

- `experiments/replays/<policy>/seed-0000.json`: episode data.
- `experiments/replay_viewer/index.html`: shared UI for all replay data.
- `experiments/replays/index.json`: replay index used by the viewer dropdown.

For browser links with `?episode=...`, serve the project directory first:

```bash
python3 -m http.server 8000
```

Then open:

```text
http://localhost:8000/experiments/replay_viewer/index.html
```

The viewer loads `experiments/replays/index.json` and gives you a dropdown of
available replays. Direct `?episode=...` links still work for reports and
bookmarks. If you open the viewer directly from the filesystem, use its JSON
file picker.

## Evaluation

Splits are deterministic:

- `train`: seeds `0..49`
- `eval`: seeds `1000..1049`

Episode score:

```text
score = food_eaten * 20 + steps * 0.2 + (50 if survived else -30)
```

The CLI also reports average food, average steps, survival rate, and death
reasons.

## Agent Loop

For optimization rounds, `train.py` is the orchestration layer and Claude Code
is the optimizer. The train script:

1. Copies the experiment data from `~/Repos/snake-data/<exp>/` into the trainer's
   working slots
2. Prepares per-round feedback (failure report, journal template, prompt)
3. Asks the optimizer for one policy update
4. Hash-checks edit boundaries (forbidden to touch trainer code)
5. Runs tests, records train metrics, runs held-out eval
6. Snapshots the policy / notes / replays into the round directory
7. Copies modified files back to `~/Repos/snake-data/<exp>/`

There is no auto-commit. Versioning the experiment data is done manually inside
the data directory whenever you want a tag.

### Environment

`train.py` reads local secrets from `.env` and also sets the non-secret Claude
Code defaults internally:

- `ENABLE_TOOL_SEARCH=false`
- `ANTHROPIC_BASE_URL=https://api.kimi.com/coding/`

Only `ANTHROPIC_API_KEY` must be provided locally. Do not commit API keys.
Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

Then fill in `ANTHROPIC_API_KEY`. `.env` is ignored by git.

### Running training

```bash
# Dry-run: build the prompt + journal for inspection, don't invoke Claude
python3 train.py --exp <name> --optimizer none --dry-run

# One Claude Code optimization round
python3 train.py --exp <name> --rounds 1 --optimizer claude

# Multi-round
python3 train.py --exp <name> --rounds 5 --optimizer claude
```

Each round writes artifacts under `experiments/runs/<timestamp>/round-XX/`
(these are the trainer's local working copies; the canonical persisted copies
end up in `~/Repos/snake-data/<exp>/runs/<timestamp>/round-XX/` after copy-out).

### Round artifacts

Per round, the optimizer + trainer together produce:

- `optimizer-prompt.md`, `claude-output.txt` — what was sent / received
- `journal.md` — experiment-by-experiment log (the optimizer's long-term memory
  across context compactions)
- `pytest.txt`, `train-before.json`, `train-after.json`, `eval-after.json`
- `summary.json` — round summary
- `policy.py`, `heuristic_notes.md` — snapshots of the round-end state
- `replays/` — replay snapshots
- `scripts/` — any diagnostic scripts the optimizer chose to keep

### Edit boundaries

Optimizers may edit:

- `snake_hl/policy.py`
- `experiments/heuristic_notes.md`
- `experiments/runs/<ts>/round-NN/journal.md` and `scripts/`

Edits to trainer code (`snake_hl/env.py`, `eval.py`, `baselines.py`, `train.py`, etc.)
are detected by a hash diff and rejected.

### Starting a new experiment

```bash
# Fork from an existing baseline
cp -r ~/Repos/snake-data/r14-baseline ~/Repos/snake-data/<new-name>
cd ~/Repos/snake-data/<new-name> && git init && git add -A && git commit -m "fresh fork"

# Run training against it
cd ~/Repos/snake-arena-hl
python3 train.py --exp <new-name> --rounds 5 --optimizer claude
```

To start from genuine zero (no policy lineage):

```bash
mkdir ~/Repos/snake-data/<new-name>
# place a minimal policy.py and empty heuristic_notes.md
cd ~/Repos/snake-data/<new-name> && git init && git add -A && git commit -m "zero baseline"
python3 train.py --exp <new-name> --rounds 5 --optimizer claude
```

### Archiving an experiment

Just move it out of `~/Repos/snake-data/` to physically isolate it from future
optimizer runs:

```bash
mv ~/Repos/snake-data/<old-name> ~/Repos/snake-data-archive/<old-name>
```
