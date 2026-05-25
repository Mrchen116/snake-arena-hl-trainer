from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from snake_hl.eval import append_trial, evaluate


BASELINES: tuple[tuple[str, str], ...] = (
    ("current", "train"),
    ("current", "eval"),
    ("safe_greedy", "train"),
    ("random", "train"),
)


def write_snapshot(output: Path, reset_trials: bool = False) -> Path:
    timestamp = datetime.now(timezone.utc).isoformat()
    summaries = []
    for policy_name, split in BASELINES:
        summary, _ = evaluate(policy_name, split)
        summary["run_type"] = "baseline"
        summary["created_at"] = timestamp
        summaries.append(summary)

    lines = [
        "# Baseline Snapshot",
        "",
        f"- created_at: {timestamp}",
        "",
        "| policy | split | avg_score | avg_food | avg_steps | survival_rate | death_reasons |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for summary in summaries:
        lines.append(
            "| {policy} | {split} | {avg_score} | {avg_food} | {avg_steps} | {survival_rate} | `{death_reasons}` |".format(
                **summary
            )
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `current` is intentionally weak: it chases food while avoiding immediate death.",
            "- `safe_greedy` is a reference baseline, not the target implementation.",
            "- Optimize against train, then check eval occasionally for generalization.",
            "",
        ]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")

    trial_path = output.parent / "trials.jsonl"
    if reset_trials:
        trial_path.write_text("", encoding="utf-8")
    for summary in summaries:
        append_trial(summary, trial_path)
    (output.parent / "baseline_snapshot.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Write baseline score artifacts.")
    parser.add_argument("--output", type=Path, default=Path("experiments/baseline_snapshot.md"))
    parser.add_argument("--reset-trials", action="store_true")
    args = parser.parse_args()

    path = write_snapshot(args.output, reset_trials=args.reset_trials)
    print(path)


if __name__ == "__main__":
    main()
