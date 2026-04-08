"""
env/dataclean_env.py
=====================
The main environment class. This is the thing the agent talks to.

Exposes:
  reset(task, seed)  → Observation
  step(action)       → StepResult
  state()            → EpisodeState

All three are the methods that become HTTP endpoints in server/app.py.
This class is pure Python — knows nothing about HTTP.

State diagram:
  IDLE ──reset()──► RUNNING ──step()──► RUNNING
                              └──submit/max_steps──► DONE
                    DONE ──reset()──► RUNNING (new episode)
"""

from __future__ import annotations

import random
from typing import Dict, Optional

import numpy as np
import pandas as pd

from env.cleaning_engine     import CleaningEngine
from env.dataset_generator   import DatasetGenerator
from env.graders.base_grader import get_grader
from env.models import (
    Action, ActionType, ColumnProfile, DType,
    EpisodeState, GraderScore,
    LastActionResult, Observation, StepResult,
    TaskConfig, TaskName, TopValue, ValueRange,
)
from env.reward_calculator import RewardCalculator
from env.tasks.task_monday_morning  import TASK_CONFIG as TASK1
from env.tasks.task_warehouse_merge import TASK_CONFIG as TASK2
from env.tasks.task_data_lake       import TASK_CONFIG as TASK3


TASK_REGISTRY: Dict[TaskName, TaskConfig] = {
    TaskName.MONDAY_MORNING:   TASK1,
    TaskName.WAREHOUSE_MERGE:  TASK2,
    TaskName.DATA_LAKE_CRISIS: TASK3,
}


class DataCleanEnv:
    """
    The RL environment.

    One instance per server process. Each reset() starts a fresh episode.
    The DataFrame lives here — not in EpisodeState (Pydantic can't hold it).
    """

    def __init__(self) -> None:
        self._cleaner    = CleaningEngine()
        self._rewarder   = RewardCalculator()

        # ── Active episode data ────────────────────────────────────
        self._current_df:    Optional[pd.DataFrame] = None
        self._truth_df:      Optional[pd.DataFrame] = None
        self._secondary:     Dict[str, pd.DataFrame] = {}
        self._prev_df:       Optional[pd.DataFrame] = None
        self._episode_state: Optional[EpisodeState] = None
        self._task_config:   Optional[TaskConfig]   = None

    # ──────────────────────────────────────────────────────────────
    # reset()
    # ──────────────────────────────────────────────────────────────

    def reset(
        self,
        task: TaskName = TaskName.MONDAY_MORNING,
        seed: Optional[int] = None,
    ) -> Observation:
        """
        Start a fresh episode.
        - Loads task config
        - Generates (dirty, clean) dataset pair
        - Initialises episode state
        - Returns initial observation
        """
        if seed is None:
            seed = random.randint(1, 9000)  # training seeds only (test seeds 9001–10000)

        self._task_config = TASK_REGISTRY[task]
        gen = DatasetGenerator(self._task_config, seed=seed)
        dirty_df, clean_df, secondary = gen.generate()

        self._current_df  = dirty_df
        self._truth_df    = clean_df
        self._secondary   = secondary
        self._prev_df     = dirty_df.copy()

        self._episode_state = EpisodeState(
            task_name=task,
            seed=seed,
            step_count=0,
            max_steps=self._task_config.max_steps,
            done=False,
            initial_row_count=len(dirty_df),
            current_row_count=len(dirty_df),
            initial_null_total=int(dirty_df.isna().sum().sum()),
            current_null_total=int(dirty_df.isna().sum().sum()),
        )

        return self._build_observation(last_action_result=None)

    # ──────────────────────────────────────────────────────────────
    # step()
    # ──────────────────────────────────────────────────────────────

    def step(self, action: Action) -> StepResult:
        """
        Apply one cleaning action.
        Returns (observation, reward, done, info).
        """
        if self._episode_state is None or self._current_df is None:
            raise RuntimeError("Call reset() before step().")
        if self._episode_state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        # ── Terminal action ───────────────────────────────────────
        if action.action_type == ActionType.SUBMIT:
            return self._handle_submit()

        # ── Apply cleaning operation ──────────────────────────────
        self._prev_df = self._current_df.copy()

        new_df, action_result = self._cleaner.apply(
            action=action,
            df=self._current_df,
            business_rules=self._task_config.business_rules,
            secondary_tables=self._secondary,
        )

        # Update secondary tables if a merge happened
        if action.action_type == ActionType.MERGE_TABLE and action.table_name:
            self._secondary.pop(action.table_name, None)

        self._current_df = new_df

        # ── Compute step reward ───────────────────────────────────
        reward = self._rewarder.step_reward(
            prev_df=self._prev_df,
            new_df=new_df,
            action=action,
            action_result=action_result,
            action_history=self._episode_state.action_history,
        )

        # ── Update episode state ──────────────────────────────────
        self._episode_state.step_count      += 1
        self._episode_state.cumulative_reward += reward
        self._episode_state.step_rewards.append(reward)
        self._episode_state.current_row_count = len(new_df)
        self._episode_state.current_null_total = int(new_df.isna().sum().sum())
        self._episode_state.action_history.append({
            "step":        self._episode_state.step_count,
            "action_type": action.action_type.value,
            "column":      action.column,
            "rule_id":     action.rule_id,
            "table_name":  action.table_name,
            "reward":      reward,
        })

        # ── Check if episode ends (step budget exhausted) ─────────
        done = self._episode_state.step_count >= self._episode_state.max_steps
        if done:
            self._episode_state.done = True

        obs = self._build_observation(last_action_result=action_result)
        obs.done = done

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "step":          self._episode_state.step_count,
                "max_steps":     self._episode_state.max_steps,
                "action_result": action_result.message,
                "rows_affected": action_result.rows_affected,
            },
        )

    # ──────────────────────────────────────────────────────────────
    # state()
    # ──────────────────────────────────────────────────────────────

    def state(self) -> EpisodeState:
        """
        Return full internal episode state.
        Used by /state endpoint — for debuggers and evaluators, not agents.
        """
        if self._episode_state is None:
            return EpisodeState(
                task_name=TaskName.MONDAY_MORNING,
                seed=0,
                done=False,
            )
        return self._episode_state

    # ──────────────────────────────────────────────────────────────
    # PRIVATE — submit handling
    # ──────────────────────────────────────────────────────────────

    def _handle_submit(self) -> StepResult:
        """Agent called submit(). Run grader, compute terminal reward, end episode."""
        grader = get_grader(self._task_config)
        score  = grader.score(
            agent_df=self._current_df,
            truth_df=self._truth_df,
            seed=self._episode_state.seed,
            steps_taken=self._episode_state.step_count,
        )

        self._episode_state.done         = True
        self._episode_state.final_score  = score
        self._episode_state.step_count  += 1
        self._episode_state.step_rewards.append(score.total_score)
        self._episode_state.cumulative_reward += score.total_score
        self._episode_state.action_history.append({
            "step":        self._episode_state.step_count,
            "action_type": ActionType.SUBMIT.value,
            "column":      None,
            "reward":      score.total_score,
        })

        action_result = LastActionResult(
            action_taken="submit()",
            message=(
                f"Episode complete. Final score: {score.total_score:.4f} | "
                f"null={score.null_compliance:.3f} | "
                f"type={score.type_accuracy:.3f} | "
                f"dupes={score.duplicate_elimination:.3f} | "
                f"outliers={score.outlier_compliance:.3f} | "
                f"rules={score.business_rule_score:.3f}"
            ),
            was_effective=True,
            rows_affected=0,
        )

        obs = self._build_observation(last_action_result=action_result)
        obs.done = True

        return StepResult(
            observation=obs,
            reward=score.total_score,
            done=True,
            info={
                "final_score":            score.total_score,
                "null_compliance":        score.null_compliance,
                "type_accuracy":          score.type_accuracy,
                "duplicate_elimination":  score.duplicate_elimination,
                "outlier_compliance":     score.outlier_compliance,
                "business_rule_score":    score.business_rule_score,
                "notes":                  score.notes,
                "steps_taken":            self._episode_state.step_count,
            },
        )

    # ──────────────────────────────────────────────────────────────
    # PRIVATE — observation builder
    # ──────────────────────────────────────────────────────────────

    def _build_observation(
        self,
        last_action_result: Optional[LastActionResult],
    ) -> Observation:
        """Build the Observation the agent sees from the current DataFrame."""
        df     = self._current_df
        state  = self._episode_state
        config = self._task_config

        columns = [
            self._profile_column(df, col)
            for col in df.columns
        ]

        return Observation(
            total_rows=len(df),
            total_columns=len(df.columns),
            duplicate_row_count=int(df.duplicated(keep='first').sum()),
            step_number=state.step_count,
            max_steps=state.max_steps,
            steps_remaining=max(0, state.max_steps - state.step_count),
            task_name=config.task_name,
            task_brief=config.task_brief,
            columns=columns,
            available_tables=list(self._secondary.keys()),
            available_rules=[r.rule_id for r in config.business_rules],
            last_action_result=last_action_result,
            done=state.done,
        )

    @staticmethod
    def _profile_column(df: pd.DataFrame, col: str) -> ColumnProfile:
        """Compute the statistical profile of one column."""
        series = df[col]
        n      = len(series)
        nulls  = series.isna().sum()

        # dtype family
        if pd.api.types.is_integer_dtype(series):
            dtype = DType.INTEGER
        elif pd.api.types.is_float_dtype(series):
            dtype = DType.FLOAT
        elif pd.api.types.is_bool_dtype(series):
            dtype = DType.BOOLEAN
        elif pd.api.types.is_datetime64_any_dtype(series):
            dtype = DType.DATE
        else:
            # check for mixed-type indicator
            non_null = series.dropna()
            types    = set(type(v).__name__ for v in non_null.iloc[:100])
            dtype    = DType.MIXED if len(types) > 1 else DType.STRING

        # outlier score
        outlier_score = 0.0
        value_range   = None
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) >= 4:
                q1, q3 = clean.quantile([0.25, 0.75])
                iqr    = q3 - q1
                if iqr > 0:
                    n_out = ((clean < q1 - 1.5*iqr) | (clean > q3 + 1.5*iqr)).sum()
                    outlier_score = float(n_out / len(clean))
                value_range = ValueRange(
                    min=float(clean.min()),
                    max=float(clean.max()),
                    mean=float(clean.mean()),
                    std=float(clean.std()),
                )

        # format consistency
        format_consistency = 1.0
        if dtype in (DType.STRING, DType.MIXED, DType.DATE):
            non_null = series.dropna().astype(str)
            if len(non_null) > 0:
                if any(kw in col.lower() for kw in ["date", "_at", "time"]):
                    import re
                    iso_ok = non_null.str.match(r"^\d{4}-\d{2}-\d{2}").mean()
                    format_consistency = float(iso_ok)
                elif "email" in col.lower():
                    lc = non_null.str.strip().str.lower()
                    format_consistency = float((non_null == lc).mean())
                elif "country" in col.lower():
                    iso2 = (non_null.str.len() == 2).mean()
                    format_consistency = float(iso2)

        # top values
        top_vals = []
        try:
            vc = series.value_counts().head(5)
            for val, cnt in vc.items():
                top_vals.append(TopValue(
                    value=val,
                    count=int(cnt),
                    frequency=round(float(cnt) / n, 4),
                ))
        except Exception:
            pass

        return ColumnProfile(
            column_name=col,
            inferred_dtype=dtype,
            null_rate=round(float(nulls) / n, 4) if n > 0 else 0.0,
            unique_rate=round(float(series.nunique()) / n, 4) if n > 0 else 0.0,
            outlier_score=round(outlier_score, 4),
            format_consistency=round(format_consistency, 4),
            value_range=value_range,
            top_values=top_vals,
            row_count=n,
        )