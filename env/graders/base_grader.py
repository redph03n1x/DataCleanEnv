"""
env/graders/base_grader.py
===========================
Base grader + all three task graders.

Each grader compares agent's cleaned DataFrame against
the hidden ground truth and returns a GraderScore (0.0–1.0).

All scoring is deterministic. No randomness. Same inputs → same score.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np
import pandas as pd

from env.models import GraderScore, TaskName, TaskConfig


# ─────────────────────────────────────────────────────────────────
# BASE GRADER
# ─────────────────────────────────────────────────────────────────

class BaseGrader(ABC):

    def __init__(self, task_config: TaskConfig):
        self.config = task_config

    @abstractmethod
    def score(
        self,
        agent_df:   pd.DataFrame,
        truth_df:   pd.DataFrame,
        seed:       int,
        steps_taken: int,
    ) -> GraderScore:
        """Compare agent_df to truth_df and return a GraderScore."""

    # ── Shared metric helpers ──────────────────────────────────────

    @staticmethod
    def null_compliance(agent_df: pd.DataFrame, truth_df: pd.DataFrame) -> float:
        """
        How close are per-column null rates to ground truth?
        Score = 1 - average absolute difference in null rate per column.
        """
        common_cols = [c for c in truth_df.columns if c in agent_df.columns]
        if not common_cols:
            return 0.0
        diffs = []
        for col in common_cols:
            truth_rate = truth_df[col].isna().mean()
            agent_rate = agent_df[col].isna().mean()
            diffs.append(abs(truth_rate - agent_rate))
        return float(np.clip(1.0 - np.mean(diffs), 0.0, 1.0))

    @staticmethod
    def type_accuracy(agent_df: pd.DataFrame, truth_df: pd.DataFrame) -> float:
        """Fraction of columns where dtype family matches ground truth."""
        common_cols = [c for c in truth_df.columns if c in agent_df.columns]
        if not common_cols:
            return 0.0
        matches = 0
        for col in common_cols:
            t_type = _dtype_family(truth_df[col])
            a_type = _dtype_family(agent_df[col])
            if t_type == a_type:
                matches += 1
        return float(matches) / len(common_cols)

    @staticmethod
    def duplicate_elimination(agent_df: pd.DataFrame, truth_df: pd.DataFrame) -> float:
        """
        Score based on how close agent's duplicate count is to truth's.
        Truth typically has 0 duplicates.
        """
        truth_dupes = truth_df.duplicated().sum()
        agent_dupes = agent_df.duplicated().sum()
        if truth_dupes == 0 and agent_dupes == 0:
            return 1.0
        if agent_dupes == 0 and truth_dupes > 0:
            return 1.0  # agent did better than truth (unusual but ok)
        max_dupes = max(truth_dupes, agent_dupes, 1)
        return float(np.clip(1.0 - (agent_dupes / max_dupes), 0.0, 1.0))

    @staticmethod
    def outlier_compliance(agent_df: pd.DataFrame, truth_df: pd.DataFrame) -> float:
        """
        For numeric columns: compare IQR-based outlier rates.
        Score = 1 - average absolute difference in outlier rate.
        """
        num_cols = [
            c for c in truth_df.columns
            if c in agent_df.columns
            and pd.api.types.is_numeric_dtype(truth_df[c])
            and pd.api.types.is_numeric_dtype(agent_df[c])
        ]
        if not num_cols:
            return 1.0
        diffs = []
        for col in num_cols:
            t_out = _iqr_outlier_rate(truth_df[col])
            a_out = _iqr_outlier_rate(agent_df[col])
            diffs.append(abs(t_out - a_out))
        return float(np.clip(1.0 - np.mean(diffs) * 5, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────
# TASK 1 GRADER — Monday Morning
# ─────────────────────────────────────────────────────────────────

class Task1Grader(BaseGrader):

    def score(
        self,
        agent_df:    pd.DataFrame,
        truth_df:    pd.DataFrame,
        seed:        int,
        steps_taken: int,
    ) -> GraderScore:

        weights = self.config.grader_weights
        notes:  List[str] = []

        # null compliance (35%)
        nc = self.null_compliance(agent_df, truth_df)
        notes.append(f"null_compliance={nc:.3f}")

        # type accuracy (15%)
        ta = self.type_accuracy(agent_df, truth_df)
        notes.append(f"type_accuracy={ta:.3f}")

        # duplicate elimination (25%)
        de = self.duplicate_elimination(agent_df, truth_df)
        notes.append(f"duplicate_elimination={de:.3f}")

        # outlier compliance (25%)
        oc = self.outlier_compliance(agent_df, truth_df)
        notes.append(f"outlier_compliance={oc:.3f}")

        # format consistency bonus — dates and email
        fmt_bonus = _format_bonus(agent_df, truth_df, ["signup_date", "email"])
        notes.append(f"format_bonus(internal)={fmt_bonus:.3f}")

        # blend format bonus into outlier score for Task 1
        oc_blended = (oc + fmt_bonus) / 2.0

        total = (
            weights.get("null_compliance",       0.35) * nc
          + weights.get("type_accuracy",         0.15) * ta
          + weights.get("duplicate_elimination", 0.25) * de
          + weights.get("outlier_compliance",    0.25) * oc_blended
        )

        return GraderScore(
            null_compliance=nc,
            type_accuracy=ta,
            duplicate_elimination=de,
            outlier_compliance=oc_blended,
            business_rule_score=1.0,
            total_score=round(float(np.clip(total, 0.0, 1.0)), 4),
            task_name=TaskName.MONDAY_MORNING,
            steps_taken=steps_taken,
            seed=seed,
            notes=notes,
        )


# ─────────────────────────────────────────────────────────────────
# TASK 2 GRADER — Warehouse Merge
# ─────────────────────────────────────────────────────────────────

class Task2Grader(BaseGrader):

    def score(
        self,
        agent_df:    pd.DataFrame,
        truth_df:    pd.DataFrame,
        seed:        int,
        steps_taken: int,
    ) -> GraderScore:

        weights = self.config.grader_weights
        notes:  List[str] = []

        nc = self.null_compliance(agent_df, truth_df)
        ta = self.type_accuracy(agent_df, truth_df)
        notes.append(f"null_compliance={nc:.3f}")
        notes.append(f"type_accuracy={ta:.3f}")

        # Business rule 1: salary normalization
        br_salary = _salary_rule_score(agent_df, truth_df)
        notes.append(f"br_salary={br_salary:.3f}")

        # Business rule 2: country ISO-2
        br_country = _country_code_score(agent_df, truth_df)
        notes.append(f"br_country={br_country:.3f}")

        # Business rule 3: birth year validity
        br_birthyear = _birth_year_score(agent_df, truth_df)
        notes.append(f"br_birthyear={br_birthyear:.3f}")

        # Referential integrity (extra: orders have customers)
        ref_integrity = _referential_integrity_score(agent_df)
        notes.append(f"ref_integrity={ref_integrity:.3f}")

        # Aggregate business rule score
        brs = (br_salary + br_country + br_birthyear + ref_integrity) / 4.0

        total = (
            weights.get("null_compliance",    0.20) * nc
          + weights.get("type_accuracy",      0.20) * ta
          + weights.get("business_rule_score",0.60) * brs
        )

        return GraderScore(
            null_compliance=nc,
            type_accuracy=ta,
            duplicate_elimination=1.0,
            outlier_compliance=1.0,
            business_rule_score=brs,
            total_score=round(float(np.clip(total, 0.0, 1.0)), 4),
            task_name=TaskName.WAREHOUSE_MERGE,
            steps_taken=steps_taken,
            seed=seed,
            notes=notes,
        )


# ─────────────────────────────────────────────────────────────────
# TASK 3 GRADER — Data Lake Crisis
# ─────────────────────────────────────────────────────────────────

class Task3Grader(BaseGrader):

    def score(
        self,
        agent_df:    pd.DataFrame,
        truth_df:    pd.DataFrame,
        seed:        int,
        steps_taken: int,
    ) -> GraderScore:

        weights = self.config.grader_weights
        notes:  List[str] = []

        nc = self.null_compliance(agent_df, truth_df)
        ta = self.type_accuracy(agent_df, truth_df)
        de = self.duplicate_elimination(agent_df, truth_df)
        oc = self.outlier_compliance(agent_df, truth_df)
        notes.append(f"null_compliance={nc:.3f}")
        notes.append(f"type_accuracy={ta:.3f}")
        notes.append(f"duplicate_elimination={de:.3f}")
        notes.append(f"outlier_compliance={oc:.3f}")

        # Business rules score
        br_price     = _price_negative_score(agent_df)
        br_category  = _category_canonical_score(agent_df, truth_df)
        br_timestamp = _timestamp_format_score(agent_df, truth_df)
        brs = (br_price + br_category + br_timestamp) / 3.0
        notes.append(f"br_price={br_price:.3f}")
        notes.append(f"br_category={br_category:.3f}")
        notes.append(f"br_timestamp={br_timestamp:.3f}")

        # Red herring check: agent should NOT have cleaned 'notes' column
        # (it wastes steps but we don't penalise the score directly —
        #  the step penalty already handled it via W_ZERO_EFFECT)

        total = (
            weights.get("null_compliance",       0.25) * nc
          + weights.get("type_accuracy",         0.20) * ta
          + weights.get("duplicate_elimination", 0.15) * de
          + weights.get("outlier_compliance",    0.20) * oc
          + weights.get("business_rule_score",   0.20) * brs
        )

        return GraderScore(
            null_compliance=nc,
            type_accuracy=ta,
            duplicate_elimination=de,
            outlier_compliance=oc,
            business_rule_score=brs,
            total_score=round(float(np.clip(total, 0.0, 1.0)), 4),
            task_name=TaskName.DATA_LAKE_CRISIS,
            steps_taken=steps_taken,
            seed=seed,
            notes=notes,
        )


# ─────────────────────────────────────────────────────────────────
# GRADER REGISTRY
# ─────────────────────────────────────────────────────────────────

def get_grader(task_config: TaskConfig) -> BaseGrader:
    """Factory: return the correct grader for a task."""
    mapping = {
        TaskName.MONDAY_MORNING:   Task1Grader,
        TaskName.WAREHOUSE_MERGE:  Task2Grader,
        TaskName.DATA_LAKE_CRISIS: Task3Grader,
    }
    cls = mapping.get(task_config.task_name)
    if cls is None:
        raise ValueError(f"No grader registered for task '{task_config.task_name}'")
    return cls(task_config)


# ─────────────────────────────────────────────────────────────────
# PRIVATE METRIC HELPERS
# ─────────────────────────────────────────────────────────────────

def _dtype_family(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "string"


def _iqr_outlier_rate(series: pd.Series) -> float:
    s = series.dropna()
    if len(s) < 4:
        return 0.0
    try:
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            return 0.0
        return float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean())
    except Exception:
        return 0.0


def _format_bonus(
    agent_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    cols: List[str],
) -> float:
    """Fraction of specified columns that match ground truth format."""
    scores = []
    for col in cols:
        if col not in agent_df.columns or col not in truth_df.columns:
            continue
        # date: check ISO pattern fraction
        if "date" in col.lower():
            import re
            iso = agent_df[col].dropna().astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$")
            scores.append(float(iso.mean()) if len(iso) else 1.0)
        # email: check lowercase fraction
        elif "email" in col.lower():
            lc = agent_df[col].dropna().astype(str)
            scores.append(float((lc == lc.str.strip().str.lower()).mean()))
    return float(np.mean(scores)) if scores else 1.0


def _salary_rule_score(agent_df: pd.DataFrame, truth_df: pd.DataFrame) -> float:
    """Fraction of salary values that are in dollar range (>= 1000)."""
    col = "annual_salary"
    if col not in agent_df.columns:
        return 0.0
    valid = agent_df[col].dropna()
    if len(valid) == 0:
        return 0.0
    in_range = (valid >= 1000).mean()
    return float(in_range)


def _country_code_score(agent_df: pd.DataFrame, truth_df: pd.DataFrame) -> float:
    """Fraction of country codes that are ISO-2 (2 characters)."""
    col = "country_code"
    if col not in agent_df.columns:
        return 0.0
    valid = agent_df[col].dropna().astype(str)
    if len(valid) == 0:
        return 0.0
    iso2 = (valid.str.len() == 2).mean()
    return float(iso2)


def _birth_year_score(agent_df: pd.DataFrame, truth_df: pd.DataFrame) -> float:
    """Fraction of birth years in valid range [1904, current_year]."""
    import datetime
    col = "birth_year"
    if col not in agent_df.columns:
        return 0.0
    valid = agent_df[col].dropna()
    if len(valid) == 0:
        return 0.0
    current_year = datetime.datetime.now().year
    in_range = ((valid >= 1904) & (valid <= current_year)).mean()
    return float(in_range)


def _referential_integrity_score(agent_df: pd.DataFrame) -> float:
    """
    If customer_id column exists and has nulls, score is penalised.
    Proxy for: all orders/transactions have valid customers.
    """
    col = "customer_id"
    if col not in agent_df.columns:
        return 1.0
    null_rate = float(agent_df[col].isna().mean())
    return float(np.clip(1.0 - null_rate * 5, 0.0, 1.0))


def _price_negative_score(agent_df: pd.DataFrame) -> float:
    """Fraction of price values that are non-negative."""
    col = "price"
    if col not in agent_df.columns:
        return 1.0
    valid = agent_df[col].dropna()
    if len(valid) == 0:
        return 1.0
    return float((valid >= 0).mean())


def _category_canonical_score(agent_df: pd.DataFrame, truth_df: pd.DataFrame) -> float:
    """Fraction of product_category values that are canonical."""
    from env.dataset_generator import CANONICAL_CATEGORIES
    col = "product_category"
    if col not in agent_df.columns:
        return 0.0
    valid = agent_df[col].dropna().astype(str)
    if len(valid) == 0:
        return 0.0
    canonical_set = set(CANONICAL_CATEGORIES)
    return float(valid.isin(canonical_set).mean())


def _timestamp_format_score(agent_df: pd.DataFrame, truth_df: pd.DataFrame) -> float:
    """Fraction of timestamps that are UTC ISO 8601 format."""
    col = "created_at"
    if col not in agent_df.columns:
        return 1.0
    valid = agent_df[col].dropna().astype(str)
    if len(valid) == 0:
        return 1.0
    import re
    iso_utc = valid.str.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    return float(iso_utc.mean())