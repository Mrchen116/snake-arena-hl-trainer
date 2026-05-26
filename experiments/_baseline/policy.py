from __future__ import annotations

from snake_hl.env import DELTAS, SnakeState, add_points, manhattan, safe_actions


def choose_action(state: SnakeState) -> str:
    actions = safe_actions(state)
    if not actions:
        return state.direction
    return min(actions, key=lambda a: manhattan(add_points(state.head, DELTAS[a]), state.food))
