"""
env/reward_calculator.py
=========================
Computes step-level and terminal rewards.

Step reward:  called after every action (dense signal)
Terminal reward: called when agent submits (grader comparison)

Both are deterministic. Same inputs → same output always.
Reward is always clipped to [-1.0, 1.0].
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from env.models import Action, ActionType, LastActionResult, TaskConfig


class RewardCalculator:
    """
    Stateless. Called by DataCleanEnv after every step.
    """

    # ── Weights ──────────────────────────────────────────────────
    W_NULL_IMPROVEMENT    =  0.15
    W_NULL_WORSENED       = -0.10
    W_DUPE_REMOVED        =  0.10
    W_FORMAT_IMPROVED     =  0.10
    W_TYPE_IMPROVED       =  0.08
    W_OUTLIER_REDUCED     =  0.08
    W_ROWS_DELETED        = -0.15   # over-aggressive dropping
    W_ZERO_EFFECT         = -0.20   # wasted step
    W_REPEATED_ACTION     = -0.05   # loop detection
    W_SHAPE_CORRUPTED     = -0.30   # wrong merge / catastrophic change

    def step_reward(
        self,
        prev_df:        pd.DataFrame,
        new_df:         pd.DataFrame,
        action:         Action,
        action_result:  LastActionResult,
        action_history: list,
    ) -> float:
        """
        Compute reward for one step.

        Compares dataset statistics before and after the action.
        Positive reward for improvement, negative for harm or waste.
        """
        reward = 0.0

        # ── Penalise zero-effect actions immediately ───────────────
        if not action_result.was_effective:
            reward += self.W_ZERO_EFFECT
            return round(float(np.clip(reward, -1.0, 1.0)), 4)

        # ── Penalise repeated identical action on same column ──────
        if self._is_repeated(action, action_history):
            reward += self.W_REPEATED_ACTION

        # ── Penalise catastrophic row loss (>30% rows dropped) ─────
        row_loss_pct = (len(prev_df) - len(new_df)) / max(len(prev_df), 1)
        if row_loss_pct > 0.30:
            reward += self.W_SHAPE_CORRUPTED
            return round(float(np.clip(reward, -1.0, 1.0)), 4)

        # ── Null improvement ───────────────────────────────────────
        prev_null_rate = self._overall_null_rate(prev_df)
        new_null_rate  = self._overall_null_rate(new_df)
        null_delta     = prev_null_rate - new_null_rate   # positive = improvement

        if null_delta > 0.001:
            reward += self.W_NULL_IMPROVEMENT * min(null_delta * 10, 1.0)
        elif null_delta < -0.001:
            reward += self.W_NULL_WORSENED

        # ── Duplicate reduction ────────────────────────────────────
        prev_dupes = prev_df.duplicated().sum()
        new_dupes  = new_df.duplicated().sum()
        if new_dupes < prev_dupes:
            dupe_improvement = (prev_dupes - new_dupes) / max(prev_dupes, 1)
            reward += self.W_DUPE_REMOVED * dupe_improvement

        # ── Format consistency improvement ─────────────────────────
        if action.column and action.column in new_df.columns:
            col = action.column
            prev_fmt = self._format_consistency(prev_df, col)
            new_fmt  = self._format_consistency(new_df, col)
            fmt_delta = new_fmt - prev_fmt
            if fmt_delta > 0.01:
                reward += self.W_FORMAT_IMPROVED * fmt_delta

        # ── Type improvement ───────────────────────────────────────
        if action.action_type in (
            ActionType.CAST_TO_INTEGER,
            ActionType.CAST_TO_FLOAT,
            ActionType.PARSE_DATES,
        ) and action_result.was_effective:
            reward += self.W_TYPE_IMPROVED

        # ── Outlier score reduction ────────────────────────────────
        if action.column and action.column in new_df.columns:
            col = action.column
            if pd.api.types.is_numeric_dtype(new_df[col]):
                prev_out = self._outlier_score(prev_df, col)
                new_out  = self._outlier_score(new_df, col)
                out_delta = prev_out - new_out
                if out_delta > 0.01:
                    reward += self.W_OUTLIER_REDUCED * min(out_delta * 2, 1.0)

        # ── Over-aggressive row deletion ───────────────────────────
        if action.action_type == ActionType.DROP_ROWS_WITH_NULL:
            if row_loss_pct > 0.10:
                penalty = (row_loss_pct - 0.10) * 2.0
                reward -= penalty * abs(self.W_ROWS_DELETED)

        return round(float(np.clip(reward, -1.0, 1.0)), 4)

    # ──────────────────────────────────────────────────────────────
    # STATISTICS HELPERS
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _overall_null_rate(df: pd.DataFrame) -> float:
        total = df.size
        if total == 0:
            return 0.0
        return float(df.isna().sum().sum()) / total

    @staticmethod
    def _format_consistency(df: pd.DataFrame, col: str) -> float:
        """
        Proxy for format consistency:
        - For date-like columns: fraction of values parseable as ISO dates
        - For string columns: fraction of values that are stripped + lowercase
        - Returns 1.0 for non-string columns
        """
        series = df[col].dropna().astype(str)
        if series.empty:
            return 1.0

        # Date column heuristic
        if any(kw in col.lower() for kw in ["date", "at", "time", "year"]):
            iso_pattern = r"^\d{4}-\d{2}-\d{2}"
            matches = series.str.match(iso_pattern).sum()
            return float(matches) / len(series)

        # String column: check lowercase + stripped
        normalised = series.str.strip().str.lower()
        matches    = (series == normalised).sum()
        return float(matches) / len(series)

    @staticmethod
    def _outlier_score(df: pd.DataFrame, col: str) -> float:
        """
        IQR-based outlier score: fraction of values outside 1.5×IQR.
        Returns 0.0 for non-numeric or columns with no variance.
        """
        series = df[col].dropna()
        if len(series) < 4:
            return 0.0
        try:
            q1, q3 = series.quantile([0.25, 0.75])
            iqr    = q3 - q1
            if iqr == 0:
                return 0.0
            lower  = q1 - 1.5 * iqr
            upper  = q3 + 1.5 * iqr
            n_out  = ((series < lower) | (series > upper)).sum()
            return float(n_out) / len(series)
        except Exception:
            return 0.0

    @staticmethod
    def _is_repeated(action: Action, history: list) -> bool:
        """True if the exact same action+column was already taken."""
        key = (action.action_type, action.column, action.rule_id, action.table_name)
        return key in {
            (h.get("action_type"), h.get("column"), h.get("rule_id"), h.get("table_name"))
            for h in history
        }