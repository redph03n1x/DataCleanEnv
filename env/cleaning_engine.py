"""
env/cleaning_engine.py
=======================
Applies agent actions to a pandas DataFrame.

Each method takes the current DataFrame, applies one operation,
and returns (new_df, result_message, rows_affected).

The environment calls this, then computes reward from the
before/after comparison. This module knows nothing about reward.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from env.dataset_generator import ISO3_TO_ISO2, LEGACY_CATEGORY_MAP
from env.models import Action, ActionType, BusinessRule, LastActionResult


CleanResult = Tuple[pd.DataFrame, LastActionResult]


class CleaningEngine:
    """
    Stateless engine. Every method is a pure function:
      (df, **params) → (new_df, LastActionResult)

    No internal state. The environment owns the DataFrame.
    """

    # ──────────────────────────────────────────────────────────────
    # MAIN DISPATCHER
    # ──────────────────────────────────────────────────────────────

    def apply(
        self,
        action: Action,
        df: pd.DataFrame,
        business_rules: Optional[List[BusinessRule]] = None,
        secondary_tables: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> CleanResult:
        """
        Route an Action to the correct method.
        Returns (updated_df, LastActionResult).
        """
        a = action.action_type
        c = action.column

        # ── Imputation ────────────────────────────────────────────
        if a == ActionType.FILL_NULL_MEAN:
            return self.fill_null_mean(df, c)
        if a == ActionType.FILL_NULL_MEDIAN:
            return self.fill_null_median(df, c)
        if a == ActionType.FILL_NULL_MODE:
            return self.fill_null_mode(df, c)
        if a == ActionType.FILL_NULL_FORWARD:
            return self.fill_null_forward(df, c)
        if a == ActionType.DROP_ROWS_WITH_NULL:
            return self.drop_rows_with_null(df, c)

        # ── Type & Format ─────────────────────────────────────────
        if a == ActionType.CAST_TO_INTEGER:
            return self.cast_to_integer(df, c)
        if a == ActionType.CAST_TO_FLOAT:
            return self.cast_to_float(df, c)
        if a == ActionType.CAST_TO_STRING:
            return self.cast_to_string(df, c)
        if a == ActionType.PARSE_DATES:
            return self.parse_dates(df, c)
        if a == ActionType.NORMALIZE_CATEGORIES:
            return self.normalize_categories(df, c)

        # ── Deduplication ─────────────────────────────────────────
        if a == ActionType.REMOVE_EXACT_DUPLICATES:
            return self.remove_exact_duplicates(df)
        if a == ActionType.REMOVE_NEAR_DUPLICATES:
            return self.remove_near_duplicates(df, c)

        # ── Outlier Handling ──────────────────────────────────────
        if a == ActionType.CLIP_OUTLIERS_IQR:
            return self.clip_outliers_iqr(df, c)
        if a == ActionType.REMOVE_OUTLIER_ROWS:
            return self.remove_outlier_rows(df, c)

        # ── Structural ────────────────────────────────────────────
        if a == ActionType.MERGE_TABLE:
            return self.merge_table(df, action.table_name, secondary_tables or {})
        if a == ActionType.APPLY_BUSINESS_RULE:
            return self.apply_business_rule(df, action.rule_id, business_rules or [])

        # submit() is handled by the environment, not the engine
        return df, LastActionResult(
            action_taken=a.value,
            message="Unknown action — no operation performed.",
            was_effective=False,
        )

    # ──────────────────────────────────────────────────────────────
    # IMPUTATION
    # ──────────────────────────────────────────────────────────────

    def fill_null_mean(self, df: pd.DataFrame, col: str) -> CleanResult:
        if col not in df.columns:
            return df, self._bad(f"Column '{col}' not found.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            return df, self._bad(f"fill_null_mean requires numeric column, got {df[col].dtype}.")
        null_before = df[col].isna().sum()
        if null_before == 0:
            return df, self._noop(f"No nulls in '{col}'.")
        mean_val = df[col].mean()
        new_df   = df.copy()
        new_df[col] = new_df[col].fillna(mean_val)
        return new_df, LastActionResult(
            action_taken=f"fill_null_mean({col})",
            column_affected=col,
            rows_affected=int(null_before),
            message=f"Filled {null_before} nulls in '{col}' with mean={mean_val:.2f}.",
        )

    def fill_null_median(self, df: pd.DataFrame, col: str) -> CleanResult:
        if col not in df.columns:
            return df, self._bad(f"Column '{col}' not found.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            return df, self._bad(f"fill_null_median requires numeric column.")
        null_before = df[col].isna().sum()
        if null_before == 0:
            return df, self._noop(f"No nulls in '{col}'.")
        median_val = df[col].median()
        new_df     = df.copy()
        new_df[col] = new_df[col].fillna(median_val)
        return new_df, LastActionResult(
            action_taken=f"fill_null_median({col})",
            column_affected=col,
            rows_affected=int(null_before),
            message=f"Filled {null_before} nulls in '{col}' with median={median_val:.2f}.",
        )

    def fill_null_mode(self, df: pd.DataFrame, col: str) -> CleanResult:
        if col not in df.columns:
            return df, self._bad(f"Column '{col}' not found.")
        null_before = df[col].isna().sum()
        if null_before == 0:
            return df, self._noop(f"No nulls in '{col}'.")
        mode_series = df[col].mode()
        if mode_series.empty:
            return df, self._bad(f"Cannot compute mode for '{col}'.")
        mode_val = mode_series.iloc[0]
        new_df   = df.copy()
        new_df[col].fillna(mode_val, inplace=True)
        return new_df, LastActionResult(
            action_taken=f"fill_null_mode({col})",
            column_affected=col,
            rows_affected=int(null_before),
            message=f"Filled {null_before} nulls in '{col}' with mode='{mode_val}'.",
        )

    def fill_null_forward(self, df: pd.DataFrame, col: str) -> CleanResult:
        if col not in df.columns:
            return df, self._bad(f"Column '{col}' not found.")
        null_before = df[col].isna().sum()
        if null_before == 0:
            return df, self._noop(f"No nulls in '{col}'.")
        new_df = df.copy()
        new_df[col].ffill(inplace=True)
        remaining = new_df[col].isna().sum()
        filled    = null_before - remaining
        return new_df, LastActionResult(
            action_taken=f"fill_null_forward({col})",
            column_affected=col,
            rows_affected=int(filled),
            message=f"Forward-filled {filled} nulls in '{col}'. {remaining} still null (leading nulls).",
        )

    def drop_rows_with_null(self, df: pd.DataFrame, col: str) -> CleanResult:
        if col not in df.columns:
            return df, self._bad(f"Column '{col}' not found.")
        null_mask  = df[col].isna()
        null_count = null_mask.sum()
        if null_count == 0:
            return df, self._noop(f"No nulls in '{col}'.")
        new_df = df[~null_mask].reset_index(drop=True)
        return new_df, LastActionResult(
            action_taken=f"drop_rows_with_null({col})",
            column_affected=col,
            rows_affected=int(null_count),
            message=f"Dropped {null_count} rows where '{col}' was null.",
        )

    # ──────────────────────────────────────────────────────────────
    # TYPE & FORMAT
    # ──────────────────────────────────────────────────────────────

    def cast_to_integer(self, df: pd.DataFrame, col: str) -> CleanResult:
        if col not in df.columns:
            return df, self._bad(f"Column '{col}' not found.")
        new_df = df.copy()
        try:
            new_df[col] = pd.to_numeric(new_df[col], errors="coerce").astype("Int64")
            nulls_created = new_df[col].isna().sum() - df[col].isna().sum()
            return new_df, LastActionResult(
                action_taken=f"cast_to_integer({col})",
                column_affected=col,
                rows_affected=len(df),
                message=f"Cast '{col}' to integer. {nulls_created} values became null (unconvertible).",
            )
        except Exception as exc:
            return df, self._bad(f"cast_to_integer failed on '{col}': {exc}")

    def cast_to_float(self, df: pd.DataFrame, col: str) -> CleanResult:
        if col not in df.columns:
            return df, self._bad(f"Column '{col}' not found.")
        new_df = df.copy()
        try:
            new_df[col] = pd.to_numeric(new_df[col], errors="coerce")
            return new_df, LastActionResult(
                action_taken=f"cast_to_float({col})",
                column_affected=col,
                rows_affected=len(df),
                message=f"Cast '{col}' to float64.",
            )
        except Exception as exc:
            return df, self._bad(f"cast_to_float failed: {exc}")

    def cast_to_string(self, df: pd.DataFrame, col: str) -> CleanResult:
        if col not in df.columns:
            return df, self._bad(f"Column '{col}' not found.")
        new_df     = df.copy()
        new_df[col] = new_df[col].astype(str).replace("nan", None)
        return new_df, LastActionResult(
            action_taken=f"cast_to_string({col})",
            column_affected=col,
            rows_affected=len(df),
            message=f"Cast '{col}' to string.",
        )

    def parse_dates(self, df: pd.DataFrame, col: str) -> CleanResult:
        if col not in df.columns:
            return df, self._bad(f"Column '{col}' not found.")
        new_df  = df.copy()
        before  = new_df[col].copy()
        parsed = pd.to_datetime(new_df[col], errors="coerce")
        # Format as ISO YYYY-MM-DD (date columns) or YYYY-MM-DDTHH:MM:SSZ (timestamps)
        if parsed.dt.time.astype(str).eq("00:00:00").all():
            new_df[col] = parsed.dt.strftime("%Y-%m-%d")
        else:
            new_df[col] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        changed = (new_df[col] != before.astype(str)).sum()
        return new_df, LastActionResult(
            action_taken=f"parse_dates({col})",
            column_affected=col,
            rows_affected=int(changed),
            message=f"Standardized {changed} date values in '{col}' to ISO format.",
        )

    def normalize_categories(self, df: pd.DataFrame, col: str) -> CleanResult:
        if col not in df.columns:
            return df, self._bad(f"Column '{col}' not found.")
        new_df = df.copy()
        before = new_df[col].copy()
        new_df[col] = (
            new_df[col]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        changed = (new_df[col] != before.astype(str).str.strip().str.lower()).sum()
        return new_df, LastActionResult(
            action_taken=f"normalize_categories({col})",
            column_affected=col,
            rows_affected=int(changed),
            message=f"Normalized {changed} values in '{col}' to lowercase + stripped whitespace.",
        )

    # ──────────────────────────────────────────────────────────────
    # DEDUPLICATION
    # ──────────────────────────────────────────────────────────────

    def remove_exact_duplicates(self, df: pd.DataFrame) -> CleanResult:
        before   = len(df)
        new_df   = df.drop_duplicates().reset_index(drop=True)
        removed  = before - len(new_df)
        if removed == 0:
            return df, self._noop("No exact duplicate rows found.")
        return new_df, LastActionResult(
            action_taken="remove_exact_duplicates()",
            rows_affected=removed,
            message=f"Removed {removed} exact duplicate rows.",
        )

    def remove_near_duplicates(self, df: pd.DataFrame, col: str) -> CleanResult:
        if col not in df.columns:
            return df, self._bad(f"Column '{col}' not found.")
        values = df[col].fillna("").astype(str).tolist()
        to_drop = set()
        for i, val in enumerate(values):
            if i in to_drop:
                continue
            # find all other rows with fuzzy similarity > 90
            for j in range(i + 1, len(values)):
                if j in to_drop:
                    continue
                score = fuzz.ratio(val, values[j])
                if score > 90 and val != "":
                    to_drop.add(j)
        if not to_drop:
            return df, self._noop(f"No near-duplicate rows found on '{col}'.")
        new_df = df.drop(index=list(to_drop)).reset_index(drop=True)
        return new_df, LastActionResult(
            action_taken=f"remove_near_duplicates({col})",
            column_affected=col,
            rows_affected=len(to_drop),
            message=f"Removed {len(to_drop)} near-duplicate rows based on '{col}'.",
        )

    # ──────────────────────────────────────────────────────────────
    # OUTLIER HANDLING
    # ──────────────────────────────────────────────────────────────

    def clip_outliers_iqr(self, df: pd.DataFrame, col: str) -> CleanResult:
        if col not in df.columns:
            return df, self._bad(f"Column '{col}' not found.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            return df, self._bad(f"clip_outliers_iqr requires numeric column.")
        new_df = df.copy()
        q1, q3 = new_df[col].quantile([0.25, 0.75])
        iqr    = q3 - q1
        lower  = q1 - 1.5 * iqr
        upper  = q3 + 1.5 * iqr
        before = new_df[col].copy()
        new_df[col] = new_df[col].clip(lower=lower, upper=upper)
        changed = (new_df[col] != before).sum()
        if changed == 0:
            return df, self._noop(f"No outliers found in '{col}' (IQR range: [{lower:.2f}, {upper:.2f}]).")
        return new_df, LastActionResult(
            action_taken=f"clip_outliers_iqr({col})",
            column_affected=col,
            rows_affected=int(changed),
            message=f"Clipped {changed} outlier values in '{col}' to [{lower:.2f}, {upper:.2f}].",
        )

    def remove_outlier_rows(self, df: pd.DataFrame, col: str) -> CleanResult:
        if col not in df.columns:
            return df, self._bad(f"Column '{col}' not found.")
        if not pd.api.types.is_numeric_dtype(df[col]):
            return df, self._bad(f"remove_outlier_rows requires numeric column.")
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr    = q3 - q1
        lower  = q1 - 1.5 * iqr
        upper  = q3 + 1.5 * iqr
        mask   = (df[col] >= lower) & (df[col] <= upper) | df[col].isna()
        removed = (~mask).sum()
        if removed == 0:
            return df, self._noop(f"No outlier rows found in '{col}'.")
        new_df = df[mask].reset_index(drop=True)
        return new_df, LastActionResult(
            action_taken=f"remove_outlier_rows({col})",
            column_affected=col,
            rows_affected=int(removed),
            message=f"Dropped {removed} rows with outlier values in '{col}'.",
        )

    # ──────────────────────────────────────────────────────────────
    # STRUCTURAL TRANSFORMS
    # ──────────────────────────────────────────────────────────────

    def merge_table(
        self,
        df: pd.DataFrame,
        table_name: Optional[str],
        secondary_tables: Dict[str, pd.DataFrame],
    ) -> CleanResult:
        if not table_name:
            return df, self._bad("merge_table requires table_name.")
        if table_name not in secondary_tables:
            available = list(secondary_tables.keys())
            return df, self._bad(
                f"Table '{table_name}' not available. Available: {available}"
            )
        other = secondary_tables[table_name]

        # Infer join key: first column that appears in both DataFrames and ends with '_id'
        common = [c for c in df.columns if c in other.columns and c.endswith("_id")]
        if not common:
            common = [c for c in df.columns if c in other.columns]
        if not common:
            return df, self._bad(f"No common key column found to join '{table_name}'.")

        key     = common[0]
        before  = len(df)
        new_df  = pd.merge(df, other, on=key, how="left", suffixes=("", f"_{table_name}"))
        added_cols = [c for c in new_df.columns if c not in df.columns]
        return new_df, LastActionResult(
            action_taken=f"merge_table({table_name})",
            columns_affected=added_cols,
            rows_affected=len(new_df) - before,
            message=(
                f"Merged '{table_name}' on key '{key}'. "
                f"Added columns: {added_cols}. Rows: {before} → {len(new_df)}."
            ),
        )

    def apply_business_rule(
        self,
        df: pd.DataFrame,
        rule_id: Optional[str],
        business_rules: List[BusinessRule],
    ) -> CleanResult:
        if not rule_id:
            return df, self._bad("apply_business_rule requires rule_id.")

        rule = next((r for r in business_rules if r.rule_id == rule_id), None)
        if not rule:
            return df, self._bad(f"Business rule '{rule_id}' not found in task config.")

        col = rule.column
        if col not in df.columns:
            return df, self._bad(f"Column '{col}' for rule '{rule_id}' not in DataFrame.")

        new_df  = df.copy()
        changed = 0

        if rule.rule_type == "normalize_units":
            # e.g. salary in thousands → multiply values < threshold by multiplier
            threshold  = rule.parameters.get("threshold", 1000)
            multiplier = rule.parameters.get("multiplier", 1000)
            mask       = new_df[col].notna() & (new_df[col] < threshold)
            changed    = int(mask.sum())
            new_df.loc[mask, col] = new_df.loc[mask, col] * multiplier

        elif rule.rule_type == "validate_range":
            # remove rows outside valid range OR fix values (abs for negatives)
            fix    = rule.parameters.get("fix", "drop")
            min_v  = rule.parameters.get("min_value")
            max_v  = rule.parameters.get("max_value")
            min_yr = rule.parameters.get("min_year")
            max_yr_offset = rule.parameters.get("max_year_offset")

            if min_yr is not None:
                import datetime
                max_yr = datetime.datetime.now().year - (max_yr_offset or 0)
                mask   = new_df[col].notna() & (
                    (new_df[col] < min_yr) | (new_df[col] > max_yr)
                )
                changed = int(mask.sum())
                new_df  = new_df[~mask].reset_index(drop=True)
            elif min_v is not None:
                if fix == "abs":
                    mask    = new_df[col].notna() & (new_df[col] < min_v)
                    changed = int(mask.sum())
                    new_df.loc[mask, col] = new_df.loc[mask, col].abs()
                else:
                    mask    = new_df[col].notna() & (new_df[col] < min_v)
                    changed = int(mask.sum())
                    new_df  = new_df[~mask].reset_index(drop=True)

        elif rule.rule_type == "fix_encoding":
            mapping_table = rule.parameters.get("mapping_table")
            target_format = rule.parameters.get("target_format")

            if mapping_table == "iso3_to_iso2":
                mask    = new_df[col].isin(ISO3_TO_ISO2.keys())
                changed = int(mask.sum())
                new_df.loc[mask, col] = new_df.loc[mask, col].map(ISO3_TO_ISO2)

            elif target_format == "UTC_ISO8601":
                parsed = pd.to_datetime(new_df[col], infer_datetime_format=True, errors="coerce")
                new_df[col] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                changed = len(new_df)

        elif rule.rule_type == "category_map":
            mask    = new_df[col].isin(LEGACY_CATEGORY_MAP.keys())
            changed = int(mask.sum())
            new_df.loc[mask, col] = new_df.loc[mask, col].map(LEGACY_CATEGORY_MAP)

        if changed == 0:
            return df, self._noop(f"Business rule '{rule_id}' found nothing to fix.")

        return new_df, LastActionResult(
            action_taken=f"apply_business_rule({rule_id})",
            column_affected=col,
            rows_affected=changed,
            message=f"Applied rule '{rule_id}' on '{col}': {changed} values affected.",
        )

    # ──────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _bad(msg: str) -> LastActionResult:
        return LastActionResult(
            action_taken="error",
            message=msg,
            was_effective=False,
            rows_affected=0,
        )

    @staticmethod
    def _noop(msg: str) -> LastActionResult:
        return LastActionResult(
            action_taken="noop",
            message=msg,
            was_effective=False,
            rows_affected=0,
        )