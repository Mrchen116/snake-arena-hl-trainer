from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from snake_hl import baselines, policy
from snake_hl.env import EpisodeResult, simulate_episode

POLICIES = {
    "current": policy.choose_action,
    "random": baselines.random_policy,
    "greedy": baselines.greedy_policy,
    "safe_greedy": baselines.safe_greedy_policy,
}


def seeds_for_split(split: str) -> range:
    if split == "train":
        return range(0, 200)
    if split == "eval":
        return range(1000, 1200)
    raise ValueError(f"Unknown split: {split}")


def summarize(results: list[EpisodeResult]) -> dict:
    deaths = Counter(result.death_reason or "survived" for result in results)
    return {
        "episodes": len(results),
        "avg_score": round(mean(result.score for result in results), 3),
        "avg_food": round(mean(result.food_eaten for result in results), 3),
        "avg_steps": round(mean(result.steps for result in results), 3),
        "survival_rate": round(sum(result.survived for result in results) / len(results), 3),
        "death_reasons": dict(sorted(deaths.items())),
    }


def evaluate(policy_name: str, split: str) -> tuple[dict, list[EpisodeResult]]:
    choose_action = POLICIES[policy_name]
    results = [simulate_episode(choose_action, seed=seed) for seed in seeds_for_split(split)]
    summary = summarize(results)
    summary["policy"] = policy_name
    summary["split"] = split
    return summary, results


def append_trial(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Snake Arena HL policies.")
    parser.add_argument("--policy", choices=sorted(POLICIES), default="current")
    parser.add_argument("--split", choices=("train", "eval"), default="train")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    parser.add_argument("--record", action="store_true", help="Append summary to experiments/trials.jsonl.")
    args = parser.parse_args()

    summary, _ = evaluate(args.policy, args.split)
    if args.record:
        append_trial(summary, Path("experiments/trials.jsonl"))

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"{summary['policy']} on {summary['split']}:")
        print(f"  avg_score:     {summary['avg_score']}")
        print(f"  avg_food:      {summary['avg_food']}")
        print(f"  avg_steps:     {summary['avg_steps']}")
        print(f"  survival_rate: {summary['survival_rate']}")
        print(f"  death_reasons: {summary['death_reasons']}")


if __name__ == "__main__":
    main()
