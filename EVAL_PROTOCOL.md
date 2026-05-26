# Evaluation Protocol

This experiment follows the "Learning Beyond Gradients" shape:

- Environment: deterministic Snake arena
- Heuristic system: `policy.py` plus `heuristic_notes.md`
- Agent: a coding agent that edits the heuristic system
- Feedback: scores, failure report, and HTML replays
- Generalization: train seeds versus eval seeds

## Splits

- Train: seeds `0..49`
- Eval: seeds `1000..1049`

Agents may use train failure reports and train HTML replays for diagnosis.
Agents should use eval only as a held-out score check.

## Score

```text
score = food_eaten * 20 + steps * 0.2 + (50 if survived else -30)
```

Primary metrics:

- `avg_score`
- `avg_food`
- `survival_rate`
- `death_reasons`

## Trial Flow

Use `train.py` as the default orchestration layer. Claude Code is the optimizer
inside each round, not the owner of the whole training loop.

1. `train.py` evaluates current train performance.
2. `train.py` regenerates `experiments/<exp>/reports/train-failures.md` and replay data.
3. `train.py` writes an optimizer prompt under `experiments/<exp>/runs/<timestamp>/round-XX/`.
4. Claude Code edits files inside the experiment directory only — primarily
   `policy.py` and `heuristic_notes.md`. Trainer paths are blocked at the
   permission + sandbox layers.
5. `train.py` reloads the policy via `policy_runtime.reload()`.
6. `train.py` runs tests and train evaluation.
7. `train.py` runs eval only after a meaningful train improvement.
8. `train.py` writes round summaries and appends metrics to `experiments/<exp>/trials.jsonl`.

Manual diagnosis should follow the same boundaries:

- Read `experiments/<exp>/heuristic_notes.md`.
- Read `experiments/<exp>/reports/train-failures.md`.
- Inspect linked train replays if needed.
- Edit only files under `experiments/<exp>/` (primarily `policy.py` and `heuristic_notes.md`).
- Run train evaluation.
- Run eval evaluation only after a meaningful train improvement.
- Summarize what changed and whether it helped.

## Baseline Artifact Generation

```bash
python3 -m snake_hl.baseline_snapshot --reset-trials
python3 -m snake_hl.failure_report --policy current --split train --limit 5
```
