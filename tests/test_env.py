from snake_hl.baselines import greedy_policy
from snake_hl.env import SnakeEnv, flood_fill_area, safe_actions, simulate_episode


def test_initial_state_is_valid() -> None:
    env = SnakeEnv(seed=1)

    assert env.state.width == 12
    assert env.state.height == 12
    assert len(env.state.snake) == 3
    assert env.state.food not in env.state.snake
    assert set(safe_actions(env.state)) == {"UP", "DOWN", "RIGHT"}


def test_step_eats_food_and_grows() -> None:
    env = SnakeEnv(seed=1)
    state = env.state
    env.state = type(state)(
        width=state.width,
        height=state.height,
        snake=state.snake,
        food=(state.head[0] + 1, state.head[1]),
        direction=state.direction,
    )

    next_state = env.step("RIGHT")

    assert next_state.food_eaten == 1
    assert len(next_state.snake) == 4
    assert not next_state.done


def test_wall_collision_ends_episode() -> None:
    env = SnakeEnv(seed=1)
    for _ in range(20):
        state = env.step("RIGHT")
        if state.done:
            break

    assert env.state.done
    assert env.state.death_reason == "wall"


def test_flood_fill_area_is_positive_for_open_board() -> None:
    env = SnakeEnv(seed=1)

    assert flood_fill_area(env.state, (env.state.head[0] + 1, env.state.head[1])) > 100


def test_simulate_episode_returns_metrics() -> None:
    result = simulate_episode(greedy_policy, seed=2, keep_states=True)

    assert result.seed == 2
    assert result.steps > 0
    assert result.states[0].steps == 0
    assert len(result.actions) == result.steps
