from __future__ import annotations

import argparse
from pathlib import Path

from snake_hl.eval import POLICIES, evaluate
from snake_hl.env import simulate_episode
from snake_hl.html_replay import write_episode_json, write_replay_index, write_viewer


def last_actions(actions: tuple[str, ...], limit: int = 30) -> str:
    start = max(0, len(actions) - limit)
    return "\n".join(f"{index + 1}: {action}" for index, action in enumerate(actions[start:], start))


def write_failure_report(policy_name: str, split: str, limit: int, output: Path) -> Path:
    summary, results = evaluate(policy_name, split)
    worst = sorted(results, key=lambda result: (result.survived, result.score, result.food_eaten))[:limit]

    lines = [
        f"# Failure Report: {policy_name} / {split}",
        "",
        "## Summary",
        "",
        f"- avg_score: {summary['avg_score']}",
        f"- avg_food: {summary['avg_food']}",
        f"- avg_steps: {summary['avg_steps']}",
        f"- survival_rate: {summary['survival_rate']}",
        f"- death_reasons: {summary['death_reasons']}",
        "",
        "## Worst Episodes",
        "",
    ]

    experiments_dir = output.parent.parent
    replay_dir = experiments_dir / "replays" / policy_name / split
    viewer_path = experiments_dir / "replay_viewer" / "index.html"
    write_viewer(viewer_path)

    for rank, result in enumerate(worst, 1):
        replay_path = replay_dir / f"seed-{result.seed:04d}.json"
        write_episode_json(policy_name, result.seed, replay_path)
        full_result = simulate_episode(POLICIES[policy_name], seed=result.seed, keep_states=True)
        relative_viewer = Path("../replay_viewer/index.html")
        relative_replay = Path("../replays") / policy_name / split / replay_path.name
        replay_url = f"{relative_viewer.as_posix()}?episode={relative_replay.as_posix()}"
        lines.extend(
            [
                f"### {rank}. seed {result.seed}",
                "",
                f"- score: {result.score:.1f}",
                f"- food: {result.food_eaten}",
                f"- steps: {result.steps}",
                f"- death: {result.death_reason or 'survived'}",
                f"- replay data: [{replay_path.name}]({relative_replay.as_posix()})",
                f"- viewer: [open replay]({replay_url})",
                "",
                "Last actions:",
                "",
                "```text",
                last_actions(full_result.actions),
                "```",
                "",
            ]
        )

    write_replay_index(experiments_dir / "replays")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate failure report and HTML replays.")
    parser.add_argument("--policy", choices=sorted(POLICIES), default="current")
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("reports/train-failures.md"))
    args = parser.parse_args()

    path = write_failure_report(args.policy, args.split, args.limit, args.output)
    print(path)


if __name__ == "__main__":
    main()
