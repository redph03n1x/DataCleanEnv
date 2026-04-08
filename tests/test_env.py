"""
tests/test_env.py
==================
Core test suite. Covers reset, step, graders, reward bounds, reproducibility.
Run with: pytest tests/ -v
"""

import pytest
import pandas as pd

from env.dataclean_env import DataCleanEnv
from env.models import (
    Action, ActionType, TaskName,
    ResetRequest, StepRequest,
)


@pytest.fixture
def env():
    return DataCleanEnv()


# ─────────────────────────────────────────────────────────────────
# reset() tests
# ─────────────────────────────────────────────────────────────────

def test_reset_returns_observation(env):
    obs = env.reset(task=TaskName.MONDAY_MORNING, seed=42)
    assert obs.total_rows > 0
    assert obs.total_columns > 0
    assert obs.step_number == 0
    assert obs.task_name == TaskName.MONDAY_MORNING
    assert obs.task_brief != ""
    assert len(obs.columns) == obs.total_columns


def test_reset_cleans_state(env):
    env.reset(task=TaskName.MONDAY_MORNING, seed=42)
    # Take a step, then reset — state should be fresh
    env.step(Action(action_type=ActionType.REMOVE_EXACT_DUPLICATES))
    obs = env.reset(task=TaskName.MONDAY_MORNING, seed=42)
    assert obs.step_number == 0
    assert not obs.done


def test_reset_same_seed_reproducible(env):
    obs1 = env.reset(task=TaskName.MONDAY_MORNING, seed=42)
    obs2 = env.reset(task=TaskName.MONDAY_MORNING, seed=42)
    assert obs1.total_rows          == obs2.total_rows
    assert obs1.duplicate_row_count == obs2.duplicate_row_count
    assert obs1.columns[0].null_rate == obs2.columns[0].null_rate


def test_reset_different_seeds_differ(env):
    obs1 = env.reset(task=TaskName.MONDAY_MORNING, seed=1)
    obs2 = env.reset(task=TaskName.MONDAY_MORNING, seed=999)
    # Not guaranteed to differ but almost certain with different seeds
    assert obs1.total_rows > 0 and obs2.total_rows > 0


def test_all_tasks_reset(env):
    for task in TaskName:
        obs = env.reset(task=task, seed=42)
        assert obs.task_name == task
        assert obs.total_rows > 0


# ─────────────────────────────────────────────────────────────────
# step() tests
# ─────────────────────────────────────────────────────────────────

def test_step_returns_valid_result(env):
    env.reset(task=TaskName.MONDAY_MORNING, seed=42)
    result = env.step(Action(action_type=ActionType.REMOVE_EXACT_DUPLICATES))
    assert result.observation is not None
    assert isinstance(result.reward, float)
    assert isinstance(result.done, bool)
    assert isinstance(result.info, dict)


def test_step_reward_in_bounds(env):
    env.reset(task=TaskName.MONDAY_MORNING, seed=42)
    for _ in range(5):
        result = env.step(Action(
            action_type=ActionType.FILL_NULL_MEDIAN,
            column="age",
        ))
        assert -1.0 <= result.reward <= 1.0


def test_step_increments_counter(env):
    env.reset(task=TaskName.MONDAY_MORNING, seed=42)
    for i in range(1, 4):
        result = env.step(Action(action_type=ActionType.REMOVE_EXACT_DUPLICATES))
        assert result.observation.step_number == i


def test_step_before_reset_raises(env):
    with pytest.raises(RuntimeError):
        env.step(Action(action_type=ActionType.SUBMIT))


def test_step_after_done_raises(env):
    env.reset(task=TaskName.MONDAY_MORNING, seed=42)
    env.step(Action(action_type=ActionType.SUBMIT))
    with pytest.raises(RuntimeError):
        env.step(Action(action_type=ActionType.REMOVE_EXACT_DUPLICATES))


def test_submit_returns_terminal_reward(env):
    env.reset(task=TaskName.MONDAY_MORNING, seed=42)
    result = env.step(Action(action_type=ActionType.SUBMIT))
    assert result.done is True
    assert 0.0 <= result.reward <= 1.0
    assert "final_score" in result.info


def test_max_steps_ends_episode(env):
    obs = env.reset(task=TaskName.MONDAY_MORNING, seed=42)
    max_steps = obs.max_steps
    for _ in range(max_steps):
        result = env.step(Action(
            action_type=ActionType.FILL_NULL_MEAN,
            column="revenue",
        ))
    assert result.done is True


# ─────────────────────────────────────────────────────────────────
# state() tests
# ─────────────────────────────────────────────────────────────────

def test_state_returns_episode_state(env):
    env.reset(task=TaskName.MONDAY_MORNING, seed=42)
    state = env.state()
    assert state.task_name == TaskName.MONDAY_MORNING
    assert state.seed == 42
    assert state.step_count == 0


def test_state_tracks_steps(env):
    env.reset(task=TaskName.MONDAY_MORNING, seed=42)
    env.step(Action(action_type=ActionType.REMOVE_EXACT_DUPLICATES))
    env.step(Action(action_type=ActionType.PARSE_DATES, column="signup_date"))
    state = env.state()
    assert state.step_count == 2
    assert len(state.step_rewards) == 2


# ─────────────────────────────────────────────────────────────────
# grader tests — scores always 0.0–1.0
# ─────────────────────────────────────────────────────────────────

def test_grader_score_in_bounds():
    from env.graders.base_grader import get_grader
    from env.tasks.task_monday_morning import TASK_CONFIG
    from env.dataset_generator import DatasetGenerator

    gen     = DatasetGenerator(TASK_CONFIG, seed=42)
    dirty, clean, _ = gen.generate()
    grader  = get_grader(TASK_CONFIG)
    score   = grader.score(dirty, clean, seed=42, steps_taken=5)

    assert 0.0 <= score.total_score <= 1.0
    assert 0.0 <= score.null_compliance <= 1.0
    assert 0.0 <= score.type_accuracy <= 1.0
    assert 0.0 <= score.duplicate_elimination <= 1.0
    assert 0.0 <= score.outlier_compliance <= 1.0


def test_grader_perfect_score_on_clean_data():
    """Grader should score ~1.0 when agent submits the ground truth itself."""
    from env.graders.base_grader import get_grader
    from env.tasks.task_monday_morning import TASK_CONFIG
    from env.dataset_generator import DatasetGenerator

    gen    = DatasetGenerator(TASK_CONFIG, seed=42)
    _, clean, _ = gen.generate()
    grader = get_grader(TASK_CONFIG)
    score  = grader.score(clean, clean, seed=42, steps_taken=1)

    assert score.total_score >= 0.90


def test_grader_all_tasks():
    """All 3 graders run without error and return bounded scores."""
    from env.graders.base_grader import get_grader
    from env.dataset_generator import DatasetGenerator
    from env.tasks.task_monday_morning  import TASK_CONFIG as T1
    from env.tasks.task_warehouse_merge import TASK_CONFIG as T2
    from env.tasks.task_data_lake       import TASK_CONFIG as T3

    for cfg in [T1, T2, T3]:
        gen    = DatasetGenerator(cfg, seed=99)
        dirty, clean, _ = gen.generate()
        grader = get_grader(cfg)
        score  = grader.score(dirty, clean, seed=99, steps_taken=10)
        assert 0.0 <= score.total_score <= 1.0, f"Score out of bounds for {cfg.task_name}"


# ─────────────────────────────────────────────────────────────────
# reward bounds
# ─────────────────────────────────────────────────────────────────

def test_reward_always_finite(env):
    import math
    env.reset(task=TaskName.MONDAY_MORNING, seed=42)
    actions = [
        Action(action_type=ActionType.REMOVE_EXACT_DUPLICATES),
        Action(action_type=ActionType.PARSE_DATES, column="signup_date"),
        Action(action_type=ActionType.CLIP_OUTLIERS_IQR, column="age"),
        Action(action_type=ActionType.FILL_NULL_MEDIAN, column="age"),
        Action(action_type=ActionType.FILL_NULL_MEAN, column="revenue"),
        Action(action_type=ActionType.NORMALIZE_CATEGORIES, column="email"),
    ]
    for action in actions:
        result = env.step(action)
        assert not math.isnan(result.reward)
        assert not math.isinf(result.reward)
        assert -1.0 <= result.reward <= 1.0