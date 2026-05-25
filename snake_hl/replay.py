from __future__ import annotations

import argparse

from snake_hl import baselines, policy
from snake_hl.env import SnakeState, simulate_episode

POLICIES = {
    "current": policy.choose_action,
    "random": baselines.random_policy,
    "greedy": baselines.greedy_policy,
    "safe_greedy": baselines.safe_greedy_policy,
}


def render_state(state: SnakeState) -> str:
    chars = [["." for _ in range(state.width)] for _ in range(state.height)]
    fx, fy = state.food
    if 0 <= fx < state.width and 0 <= fy < state.height:
        chars[fy][fx] = "*"
    for index, (x, y) in enumerate(reversed(state.snake)):
        if 0 <= x < state.width and 0 <= y < state.height:
            chars[y][x] = "o"
    hx, hy = state.head
    if 0 <= hx < state.width and 0 <= hy < state.height:
        chars[hy][hx] = "@"
    border = "#" * (state.width + 2)
    body = ["#" + "".join(row) + "#" for row in chars]
    return "\n".join([border, *body, border])


def render_episode(policy_name: str, seed: int, every: int = 1) -> str:
    result = simulate_episode(POLICIES[policy_name], seed=seed, keep_states=True)
    lines = [
        f"policy={policy_name} seed={seed} score={result.score:.2f} "
        f"food={result.food_eaten} steps={result.steps} death={result.death_reason or 'survived'}",
        "",
    ]
    for index, state in enumerate(result.states):
        if index % every != 0 and index != len(result.states) - 1:
            continue
        action = result.actions[index - 1] if index > 0 else "START"
        lines.append(f"step={state.steps} action={action} food={state.food_eaten}")
        lines.append(render_state(state))
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a text replay for one episode.")
    parser.add_argument("--policy", choices=sorted(POLICIES), default="current")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--every", type=int, default=10)
    args = parser.parse_args()
    print(render_episode(args.policy, args.seed, every=max(1, args.every)))


if __name__ == "__main__":
    main()
