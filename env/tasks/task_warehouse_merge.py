"""
env/tasks/task_warehouse_merge.py
==================================
Task 2 — "The Data Warehouse Merge"
Difficulty: MEDIUM
Domain: Customer data across two source systems for a quarterly report
Max Steps: 30
"""
from env.models import (
    TaskConfig, TaskName, ColumnCorruptionSpec, BusinessRule
)

TASK_CONFIG = TaskConfig(
    task_name=TaskName.WAREHOUSE_MERGE,
    display_name="The Data Warehouse Merge",
    difficulty="medium",
    max_steps=30,
    dataset_rows=800,
    dataset_columns=12,
    success_threshold=0.50,
    task_brief=(
        "Two tables from different source systems must be merged and cleaned "
        "for the Q3 financial report. The tables have conflicting schemas "
        "and several business rule violations.\n\n"
        "Requirements:\n"
        "- Merge 'customers' (main) with 'transactions' table on customer_id\n"
        "- Drop rows with null customer_id — they are unrecoverable\n"
        "- All salary values must be in USD (values under 1000 are in thousands — multiply by 1000)\n"
        "- Country codes must be standardized to ISO-2 format (e.g. 'USA' → 'US')\n"
        "- Remove impossible birth years (future dates or age > 120)\n"
        "- Remove near-duplicate customers (same person, slight name variation)\n"
        "- Final dataset must have referential integrity: every order has a valid customer"
    ),

    column_corruptions=[
        ColumnCorruptionSpec(
            column_name="customer_id",
            null_rate=0.15,         # these rows must be DROPPED not imputed
        ),
        ColumnCorruptionSpec(
            column_name="annual_salary",
            null_rate=0.05,
            add_outliers=True,      # some values in thousands not dollars
        ),
        ColumnCorruptionSpec(
            column_name="country_code",
            mixed_formats=True,     # ISO-2 and ISO-3 mixed
        ),
        ColumnCorruptionSpec(
            column_name="birth_year",
            add_outliers=True,      # future years + impossibly old
        ),
        ColumnCorruptionSpec(
            column_name="full_name",
            null_rate=0.02,
        ),
    ],

    global_duplicate_rate=0.17,     # ~340 near-duplicates in 2000

    secondary_tables=["transactions"],

    business_rules=[
        BusinessRule(
            rule_id="salary_normalization",
            description="Salary values below 1000 are stored in thousands — multiply by 1000",
            column="annual_salary",
            rule_type="normalize_units",
            parameters={"threshold": 1000, "multiplier": 1000},
        ),
        BusinessRule(
            rule_id="country_iso2",
            description="Standardize country codes to ISO-2 format",
            column="country_code",
            rule_type="fix_encoding",
            parameters={"mapping": "iso3_to_iso2"},
        ),
        BusinessRule(
            rule_id="birth_year_validation",
            description="Remove rows with impossible birth years (future or age > 120)",
            column="birth_year",
            rule_type="validate_range",
            parameters={"min_year": 1904, "max_year_offset": 0},  # max = current year
        ),
    ],

    grader_weights={
        "null_compliance":       0.20,
        "type_accuracy":         0.20,
        "duplicate_elimination": 0.00,   # near-dedup handled by business_rule_score
        "outlier_compliance":    0.00,
        "business_rule_score":   0.60,   # this task is mostly about business rules
    },
)