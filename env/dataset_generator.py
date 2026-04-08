"""
env/dataset_generator.py
=========================
Generates (dirty_df, ground_truth_df) pairs for each task.

Architecture:
  1. generate_clean()  — Faker produces realistic, fully clean DataFrame
  2. corrupt()         — applies controlled dirtiness per ColumnCorruptionSpec
  3. generate()        — returns (dirty_df, clean_df, secondary_tables_dict)

Seeded via numpy.random.Generator so same seed always produces
identical (dirty, clean) pairs. This is the reproducibility guarantee.
"""

from __future__ import annotations

import re
import random
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from faker import Faker

from env.models import (
    TaskConfig, TaskName,
    ColumnCorruptionSpec,
)

# ── Category maps (used in Task 3) ──────────────────────────────

# 12 canonical ML categories
CANONICAL_CATEGORIES = [
    "Electronics", "Clothing", "Food", "Sports",
    "Home", "Books", "Toys", "Beauty",
    "Automotive", "Garden", "Office", "Health",
]

# 47 legacy values that map to the 12 canonical ones
LEGACY_CATEGORY_MAP: Dict[str, str] = {
    "electronics":         "Electronics",
    "Electrnics":          "Electronics",
    "ELECTRONICS":         "Electronics",
    "elec":                "Electronics",
    "gadgets":             "Electronics",
    "clothing":            "Clothing",
    "CLOTHING":            "Clothing",
    "apparel":             "Clothing",
    "Clothng":             "Clothing",
    "fashion":             "Clothing",
    "food":                "Food",
    "FOOD":                "Food",
    "groceries":           "Food",
    "beverages":           "Food",
    "edibles":             "Food",
    "sports":              "Sports",
    "SPORTS":              "Sports",
    "sporting goods":      "Sports",
    "fitness":             "Sports",
    "athletics":           "Sports",
    "home":                "Home",
    "HOME":                "Home",
    "household":           "Home",
    "furniture":           "Home",
    "decor":               "Home",
    "books":               "Books",
    "BOOKS":               "Books",
    "literature":          "Books",
    "reading":             "Books",
    "toys":                "Toys",
    "TOYS":                "Toys",
    "games":               "Toys",
    "kids":                "Toys",
    "beauty":              "Beauty",
    "BEAUTY":              "Beauty",
    "cosmetics":           "Beauty",
    "skincare":            "Beauty",
    "automotive":          "Automotive",
    "auto parts":          "Automotive",
    "cars":                "Automotive",
    "garden":              "Garden",
    "GARDEN":              "Garden",
    "plants":              "Garden",
    "office":              "Office",
    "stationery":          "Office",
    "health":              "Health",
    "wellness":            "Health",
}

# ISO-3 to ISO-2 mapping (subset for Task 2)
ISO3_TO_ISO2: Dict[str, str] = {
    "USA": "US", "GBR": "GB", "CAN": "CA", "AUS": "AU",
    "IND": "IN", "DEU": "DE", "FRA": "FR", "BRA": "BR",
    "JPN": "JP", "CHN": "CN", "MEX": "MX", "ITA": "IT",
    "ESP": "ES", "NLD": "NL", "RUS": "RU", "KOR": "KR",
    "SGP": "SG", "ZAF": "ZA", "ARG": "AR", "NGA": "NG",
}

DATE_FORMATS = ["%m/%d/%Y", "%d-%b-%Y", "%Y/%m/%d"]  # non-ISO formats to mix in


class DatasetGenerator:
    """
    Generates reproducible (dirty, clean) dataset pairs.

    Usage:
        gen = DatasetGenerator(task_config, seed=42)
        dirty_df, clean_df, secondary = gen.generate()
    """

    def __init__(self, task_config: TaskConfig, seed: int = 42):
        self.config = task_config
        self.seed   = seed
        self.rng    = np.random.default_rng(seed)
        self.fake   = Faker()
        Faker.seed(seed)
        random.seed(seed)

    # ──────────────────────────────────────────────────────────────
    # PUBLIC ENTRY POINT
    # ──────────────────────────────────────────────────────────────

    def generate(
        self,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Returns:
            dirty_df:    what the agent sees
            clean_df:    hidden ground truth for grader comparison
            secondaries: secondary table dict (empty for Task 1)
        """
        task = self.config.task_name

        if task == TaskName.MONDAY_MORNING:
            clean_df     = self._gen_sales_clean()
            secondary    = {}
        elif task == TaskName.WAREHOUSE_MERGE:
            clean_df, secondary = self._gen_customers_clean()
        else:
            clean_df, secondary = self._gen_lake_clean()

        dirty_df = self._corrupt(clean_df.copy())

        # For multi-table tasks, also dirty secondary tables
        dirty_secondary: Dict[str, pd.DataFrame] = {}
        for name, tbl in secondary.items():
            dirty_secondary[name] = self._corrupt_secondary(tbl.copy(), name)

        return dirty_df, clean_df, dirty_secondary

    # ──────────────────────────────────────────────────────────────
    # CLEAN DATASET GENERATORS
    # ──────────────────────────────────────────────────────────────

    def _gen_sales_clean(self) -> pd.DataFrame:
        """Task 1 — 500-row sales DataFrame, fully clean."""
        n = self.config.dataset_rows
        rows = []
        for _ in range(n):
            age = int(self.rng.integers(18, 70))
            rows.append({
                "customer_id":  self.fake.uuid4(),
                "full_name":    self.fake.name(),
                "email":        self.fake.email().lower(),
                "age":          age,
                "signup_date":  self._random_date_iso(),
                "revenue":      round(float(self.rng.uniform(10, 15000)), 2),
                "region":       self.fake.state_abbr(),
                "notes":        self.fake.sentence(),
            })
        return pd.DataFrame(rows)

    def _gen_customers_clean(
        self,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Task 2 — customers (2000 rows) + transactions secondary table."""
        n   = self.config.dataset_rows
        iso2_codes = list(ISO3_TO_ISO2.values())
        rows = []
        for _ in range(n):
            birth_year = int(self.rng.integers(1950, 2000))
            rows.append({
                "customer_id":    self.fake.uuid4(),
                "full_name":      self.fake.name(),
                "email":          self.fake.email().lower(),
                "annual_salary":  round(float(self.rng.uniform(25000, 150000)), 2),
                "country_code":   str(self.rng.choice(iso2_codes)),
                "birth_year":     birth_year,
                "city":           self.fake.city(),
                "subscription":   str(self.rng.choice(["free", "basic", "premium"])),
                "signup_date":    self._random_date_iso(),
                "is_active":      bool(self.rng.choice([True, False])),
                "credit_score":   int(self.rng.integers(300, 850)),
                "segment":        str(self.rng.choice(["A", "B", "C", "D"])),
            })
        customers = pd.DataFrame(rows)

        # Secondary: transactions table
        cust_ids = customers["customer_id"].tolist()
        t_rows = []
        for _ in range(800):
            t_rows.append({
                "transaction_id": self.fake.uuid4(),
                "customer_id":    str(self.rng.choice(cust_ids)),
                "amount":         round(float(self.rng.uniform(5, 5000)), 2),
                "tx_date":        self._random_date_iso(),
                "status":         str(self.rng.choice(["completed", "pending", "failed"])),
                "channel":        str(self.rng.choice(["web", "mobile", "store"])),
            })
        transactions = pd.DataFrame(t_rows)

        return customers, {"transactions": transactions}

    def _gen_lake_clean(
        self,
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Task 3 — customers + 4 secondary tables."""
        n = self.config.dataset_rows
        rows = []
        for _ in range(n):
            rows.append({
                "customer_id":    self.fake.uuid4(),
                "full_name":      self.fake.name(),
                "email":          self.fake.email().lower(),
                "age":            int(self.rng.integers(18, 75)),
                "tenure_months":  int(self.rng.integers(1, 120)),
                "monthly_spend":  round(float(self.rng.uniform(5, 2000)), 2),
                "product_category": str(self.rng.choice(CANONICAL_CATEGORIES)),
                "created_at":     self._random_timestamp_utc(),
                "location_id":    f"LOC_{self.rng.integers(1, 200):04d}",
                "is_churned":     bool(self.rng.choice([True, False])),
                "contract_type":  str(self.rng.choice(["monthly", "annual", "trial"])),
                "support_tickets": int(self.rng.integers(0, 20)),
                "last_login_days": int(self.rng.integers(0, 365)),
                "payment_method": str(self.rng.choice(["card", "bank", "wallet"])),
                "notes":          self.fake.sentence(),   # RED HERRING column
            })
        customers = pd.DataFrame(rows)

        cust_ids = customers["customer_id"].tolist()
        loc_ids  = customers["location_id"].tolist()

        # orders
        order_rows = []
        for _ in range(2000):
            order_rows.append({
                "order_id":      self.fake.uuid4(),
                "customer_id":   str(self.rng.choice(cust_ids)),
                "product_id":    f"PROD_{self.rng.integers(1, 500):04d}",
                "price":         round(float(self.rng.uniform(1, 1000)), 2),
                "order_date":    self._random_timestamp_utc(),
                "quantity":      int(self.rng.integers(1, 10)),
            })
        orders = pd.DataFrame(order_rows)

        order_ids = orders["order_id"].tolist()
        prod_ids  = orders["product_id"].unique().tolist()

        # products
        product_rows = []
        for pid in prod_ids[:200]:
            product_rows.append({
                "product_id":       pid,
                "product_name":     self.fake.bs().title(),
                "category":         str(self.rng.choice(CANONICAL_CATEGORIES)),
                "unit_cost":        round(float(self.rng.uniform(0.5, 500)), 2),
                "supplier_country": "US",
            })
        products = pd.DataFrame(product_rows)

        # returns
        return_rows = []
        for _ in range(300):
            return_rows.append({
                "return_id":    self.fake.uuid4(),
                "order_id":     str(self.rng.choice(order_ids)),
                "reason":       str(self.rng.choice(["defective", "wrong_item", "changed_mind"])),
                "refund_amount": round(float(self.rng.uniform(1, 500)), 2),
                "return_date":   self._random_timestamp_utc(),
            })
        returns = pd.DataFrame(return_rows)

        # locations
        unique_locs = list(set(loc_ids))
        location_rows = []
        for lid in unique_locs:
            location_rows.append({
                "location_id": lid,
                "city":        self.fake.city(),
                "state":       self.fake.state_abbr(),
                "country":     "US",
                "timezone":    "UTC",
            })
        locations = pd.DataFrame(location_rows)

        secondary = {
            "orders":    orders,
            "products":  products,
            "returns":   returns,
            "locations": locations,
        }
        return customers, secondary

    # ──────────────────────────────────────────────────────────────
    # CORRUPTION ENGINE — turns clean → dirty
    # ──────────────────────────────────────────────────────────────

    def _corrupt(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all ColumnCorruptionSpecs from task config to df."""
        # 1. Global duplicates
        if self.config.global_duplicate_rate > 0:
            df = self._inject_duplicates(df, self.config.global_duplicate_rate)

        # 2. Column-level corruptions
        for spec in self.config.column_corruptions:
            if spec.column_name not in df.columns:
                continue
            df = self._corrupt_column(df, spec)

        return df.reset_index(drop=True)

    def _corrupt_column(
        self, df: pd.DataFrame, spec: ColumnCorruptionSpec
    ) -> pd.DataFrame:
        col = spec.column_name
        n   = len(df)

        # Nulls
        if spec.null_rate > 0:
            null_idx = self.rng.choice(n, size=int(n * spec.null_rate), replace=False)
            df.loc[null_idx, col] = None

        # Outliers
        if spec.add_outliers and spec.outlier_value is not None:
            n_outliers = max(2, int(n * 0.006))
            out_idx = self.rng.choice(n, size=n_outliers, replace=False)
            df.loc[out_idx, col] = spec.outlier_value

        # Mixed date formats
        if spec.mixed_formats and "date" in col.lower() or "at" in col.lower():
            valid_mask = df[col].notna()
            valid_idx  = df.index[valid_mask].tolist()
            for idx in valid_idx:
                fmt = random.choice(DATE_FORMATS)
                try:
                    val = df.at[idx, col]
                    if isinstance(val, str) and len(val) == 10:
                        dt  = datetime.strptime(val[:10], "%Y-%m-%d")
                        df.at[idx, col] = dt.strftime(fmt)
                except (ValueError, TypeError):
                    pass

        # Mixed category formats (Task 3 product_category)
        if spec.mixed_formats and col == "product_category":
            valid_mask = df[col].notna()
            for idx in df.index[valid_mask]:
                canonical = df.at[idx, col]
                # randomly pick one of the legacy names that maps to this canonical
                legacy_options = [
                    k for k, v in LEGACY_CATEGORY_MAP.items()
                    if v == canonical
                ]
                if legacy_options:
                    df.at[idx, col] = random.choice(legacy_options)

        # Mixed case + whitespace
        if spec.mixed_case:
            valid_mask = df[col].notna()
            for idx in df.index[valid_mask]:
                val = str(df.at[idx, col])
                # randomly upper/lower some characters + add trailing space
                mutated = "".join(
                    c.upper() if random.random() > 0.6 else c.lower()
                    for c in val
                )
                if random.random() > 0.5:
                    mutated = mutated + " "
                df.at[idx, col] = mutated

        # ISO-3 country codes (Task 2)
        if col == "country_code" and spec.mixed_formats:
            iso2_to_iso3 = {v: k for k, v in ISO3_TO_ISO2.items()}
            valid_mask   = df[col].notna()
            for idx in df.index[valid_mask]:
                if random.random() > 0.5:
                    iso2 = str(df.at[idx, col])
                    df.at[idx, col] = iso2_to_iso3.get(iso2, iso2)

        # Salary in thousands (Task 2)
        if col == "annual_salary" and spec.add_outliers:
            valid_mask = df[col].notna()
            for idx in df.index[valid_mask]:
                if random.random() > 0.6:
                    val = df.at[idx, col]
                    if isinstance(val, (int, float)) and val > 1000:
                        df.at[idx, col] = round(val / 1000, 2)

        # Invalid birth years (Task 2)
        if col == "birth_year" and spec.add_outliers:
            n_bad = max(3, int(n * 0.01))
            bad_idx = self.rng.choice(n, size=n_bad, replace=False)
            for idx in bad_idx:
                df.loc[idx, col] = random.choice([
                    2090, 2150, 1800, 1750
                ])

        # Negative prices (Task 3)
        if col == "price" and spec.add_outliers:
            valid_mask = df[col].notna()
            for idx in df.index[valid_mask]:
                if random.random() > 0.85:
                    df.at[idx, col] = -abs(float(df.at[idx, col]))

        return df

    def _corrupt_secondary(
        self, df: pd.DataFrame, table_name: str
    ) -> pd.DataFrame:
        """Light corruption for secondary tables (referential issues injected at merge)."""
        n = len(df)
        # inject ~3% nulls in non-key columns
        for col in df.columns:
            if "id" not in col.lower() and self.rng.random() > 0.7:
                null_idx = self.rng.choice(n, size=max(1, int(n * 0.03)), replace=False)
                df.loc[null_idx, col] = None
        return df

    def _inject_duplicates(self, df: pd.DataFrame, rate: float) -> pd.DataFrame:
        """Duplicate a random subset of rows."""
        n_dups = max(1, int(len(df) * rate))
        dup_idx = self.rng.choice(len(df), size=n_dups, replace=False)
        dups    = df.iloc[dup_idx].copy()
        return pd.concat([df, dups], ignore_index=True)

    # ──────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────

    def _random_date_iso(self) -> str:
        """Return a random date string in ISO format (YYYY-MM-DD)."""
        start = date(2018, 1, 1)
        delta = timedelta(days=int(self.rng.integers(0, 2000)))
        return (start + delta).strftime("%Y-%m-%d")

    def _random_timestamp_utc(self) -> str:
        """Return a random UTC ISO 8601 timestamp string."""
        start = datetime(2015, 1, 1)
        delta = timedelta(seconds=int(self.rng.integers(0, 60 * 60 * 24 * 365 * 8)))
        return (start + delta).strftime("%Y-%m-%dT%H:%M:%SZ")