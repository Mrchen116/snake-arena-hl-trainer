from __future__ import annotations

from random import Random

from snake_hl.env import ACTIONS, DELTAS, SnakeState, add_points, flood_fill_area, manhattan, safe_actions


def random_policy(state: SnakeState) -> str:
    actions = safe_actions(state) or ACTIONS
    return Random(state.steps + state.food_eaten * 1009).choice(actions)


def greedy_policy(state: SnakeState) -> str:
    actions = safe_actions(state)
    if not actions:
        return state.direction
    return min(actions, key=lambda action: manhattan(add_points(state.head, DELTAS[action]), state.food))


def safe_greedy_policy(state: SnakeState) -> str:
    actions = safe_actions(state)
    if not actions:
        return state.direction

    def score(action: str) -> tuple[int, int]:
        new_head = add_points(state.head, DELTAS[action])
        return (
            flood_fill_area(state, new_head),
            -manhattan(new_head, state.food),
        )

    return max(actions, key=score)
