from __future__ import annotations

from dataclasses import dataclass
from random import Random
from typing import Iterable

Action = str
Point = tuple[int, int]

ACTIONS: tuple[Action, ...] = ("UP", "DOWN", "LEFT", "RIGHT")
DELTAS: dict[Action, Point] = {
    "UP": (0, -1),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
}
OPPOSITE: dict[Action, Action] = {
    "UP": "DOWN",
    "DOWN": "UP",
    "LEFT": "RIGHT",
    "RIGHT": "LEFT",
}


@dataclass(frozen=True)
class SnakeState:
    width: int
    height: int
    snake: tuple[Point, ...]
    food: Point
    direction: Action
    steps: int = 0
    food_eaten: int = 0
    done: bool = False
    death_reason: str | None = None

    @property
    def head(self) -> Point:
        return self.snake[0]

    @property
    def occupied(self) -> frozenset[Point]:
        return frozenset(self.snake)


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    score: float
    food_eaten: int
    steps: int
    survived: bool
    death_reason: str | None
    states: tuple[SnakeState, ...]
    actions: tuple[Action, ...]


def add_points(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def manhattan(a: Point, b: Point) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class SnakeEnv:
    def __init__(self, width: int = 12, height: int = 12, max_steps: int = 300, seed: int = 0):
        if width < 5 or height < 5:
            raise ValueError("SnakeEnv requires width and height >= 5")
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.seed = seed
        self.rng = Random(seed)
        self.state = self._initial_state()

    def _initial_state(self) -> SnakeState:
        mid_x = self.width // 2
        mid_y = self.height // 2
        snake = ((mid_x, mid_y), (mid_x - 1, mid_y), (mid_x - 2, mid_y))
        food = self._sample_food(snake)
        return SnakeState(
            width=self.width,
            height=self.height,
            snake=snake,
            food=food,
            direction="RIGHT",
        )

    def _sample_food(self, snake: Iterable[Point]) -> Point:
        occupied = set(snake)
        empty = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in occupied
        ]
        if not empty:
            return (-1, -1)
        return self.rng.choice(empty)

    def legal_actions(self, state: SnakeState | None = None) -> tuple[Action, ...]:
        current = state or self.state
        return tuple(action for action in ACTIONS if action != OPPOSITE[current.direction])

    def step(self, action: Action) -> SnakeState:
        if action not in ACTIONS:
            raise ValueError(f"Unknown action: {action}")
        state = self.state
        if state.done:
            return state
        if action == OPPOSITE[state.direction]:
            action = state.direction

        new_head = add_points(state.head, DELTAS[action])
        snake_without_tail = state.snake[:-1]
        death_reason = self._death_reason(new_head, snake_without_tail)
        ate_food = new_head == state.food
        if death_reason:
            self.state = SnakeState(
                width=state.width,
                height=state.height,
                snake=(new_head,) + state.snake,
                food=state.food,
                direction=action,
                steps=state.steps + 1,
                food_eaten=state.food_eaten,
                done=True,
                death_reason=death_reason,
            )
            return self.state

        if ate_food:
            new_snake = (new_head,) + state.snake
            new_food = self._sample_food(new_snake)
            food_eaten = state.food_eaten + 1
        else:
            new_snake = (new_head,) + snake_without_tail
            new_food = state.food
            food_eaten = state.food_eaten

        done = state.steps + 1 >= self.max_steps
        self.state = SnakeState(
            width=state.width,
            height=state.height,
            snake=new_snake,
            food=new_food,
            direction=action,
            steps=state.steps + 1,
            food_eaten=food_eaten,
            done=done,
            death_reason=None,
        )
        return self.state

    def _death_reason(self, point: Point, occupied_body: tuple[Point, ...]) -> str | None:
        x, y = point
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return "wall"
        if point in occupied_body:
            return "self"
        return None


def is_safe_action(state: SnakeState, action: Action) -> bool:
    if action == OPPOSITE[state.direction]:
        return False
    new_head = add_points(state.head, DELTAS[action])
    x, y = new_head
    if x < 0 or x >= state.width or y < 0 or y >= state.height:
        return False
    return new_head not in state.snake[:-1]


def safe_actions(state: SnakeState) -> tuple[Action, ...]:
    return tuple(action for action in ACTIONS if is_safe_action(state, action))


def flood_fill_area(state: SnakeState, start: Point) -> int:
    x, y = start
    if x < 0 or x >= state.width or y < 0 or y >= state.height:
        return 0
    blocked = set(state.snake[:-1])
    if start in blocked:
        return 0

    seen = {start}
    stack = [start]
    while stack:
        point = stack.pop()
        for delta in DELTAS.values():
            nxt = add_points(point, delta)
            nx, ny = nxt
            if (
                0 <= nx < state.width
                and 0 <= ny < state.height
                and nxt not in blocked
                and nxt not in seen
            ):
                seen.add(nxt)
                stack.append(nxt)
    return len(seen)


def simulate_episode(
    choose_action,
    seed: int,
    width: int = 12,
    height: int = 12,
    max_steps: int = 300,
    keep_states: bool = False,
) -> EpisodeResult:
    env = SnakeEnv(width=width, height=height, max_steps=max_steps, seed=seed)
    states: list[SnakeState] = [env.state] if keep_states else []
    actions: list[Action] = []

    while not env.state.done:
        action = choose_action(env.state)
        if action not in ACTIONS:
            action = "UP"
        actions.append(action)
        env.step(action)
        if keep_states:
            states.append(env.state)

    survived = env.state.death_reason is None
    score = env.state.food_eaten * 20 + env.state.steps * 0.2 + (50 if survived else -30)
    return EpisodeResult(
        seed=seed,
        score=score,
        food_eaten=env.state.food_eaten,
        steps=env.state.steps,
        survived=survived,
        death_reason=env.state.death_reason,
        states=tuple(states),
        actions=tuple(actions),
    )
