"""
env/tasks/task_data_lake.py
============================
Task 3 — "The Data Lake Crisis"
Difficulty: HARD
Domain: Legacy system migration — 5 tables for an ML pipeline
Max Steps: 60
"""
from env.models import (
    TaskConfig, TaskName, ColumnCorruptionSpec, BusinessRule
)

TASK_CONFIG = TaskConfig(
    task_name=TaskName.DATA_LAKE_CRISIS,
    display_name="pipeline-prep",
    difficulty="hard",
    max_steps=60,
    dataset_rows=1000,      # main customers table; others are smaller
    dataset_columns=15,
    success_threshold=0.45,
    task_brief=(
        "Five tables from a legacy system migration. Unknown schema quality. "
        "10 years of historical data. Must produce a single clean unified "
        "dataset for a churn-prediction ML pipeline.\n\n"
        "ML Pipeline Requirements (strict):\n"
        "- No null values in any column\n"
        "- All numeric columns must be correct dtype (float64 or int64)\n"
        "- All timestamp columns in UTC ISO format\n"
        "- product_category must use canonical names (12 valid categories only)\n"
        "- price column: no negatives (returns were logged incorrectly)\n"
        "- No duplicate customer records\n"
        "- Referential integrity across all 5 tables\n\n"
        "Tables to merge (in correct order):\n"
        "  1. customers (main)\n"
        "  2. orders (join on customer_id)\n"
        "  3. products (join on product_id)\n"
        "  4. returns (join on order_id — careful, must be after orders merge)\n"
        "  5. locations (join on location_id — must be after customers merge)\n\n"
        "WARNING: The 'notes' free-text column appears dirty but the ML pipeline "
        "does not use it. Do not waste steps cleaning it."
    ),

    column_corruptions=[
        ColumnCorruptionSpec(
            column_name="customer_id",
            null_rate=0.03,
        ),
        ColumnCorruptionSpec(
            column_name="price",
            add_outliers=True,      # negative prices from mislogged returns
            null_rate=0.04,
        ),
        ColumnCorruptionSpec(
            column_name="created_at",
            mixed_formats=True,     # 4 timestamp formats + timezone issues
        ),
        ColumnCorruptionSpec(
            column_name="product_category",
            mixed_formats=True,     # 47 legacy values → must map to 12 canonical
            mixed_case=True,
        ),
        ColumnCorruptionSpec(
            column_name="age",
            null_rate=0.08,
            add_outliers=True,
        ),
        ColumnCorruptionSpec(
            column_name="tenure_months",
            null_rate=0.06,
        ),
        ColumnCorruptionSpec(
            column_name="monthly_spend",
            null_rate=0.05,
            add_outliers=True,
        ),
        ColumnCorruptionSpec(
            column_name="notes",
            mixed_case=True,        # RED HERRING — task brief says ignore this
        ),
    ],

    global_duplicate_rate=0.05,

    secondary_tables=["orders", "products", "returns", "locations"],

    business_rules=[
        BusinessRule(
            rule_id="price_fix_negatives",
            description="Negative prices represent mislogged returns — take absolute value",
            column="price",
            rule_type="validate_range",
            parameters={"min_value": 0, "fix": "abs"},
        ),
        BusinessRule(
            rule_id="category_canonicalize",
            description="Map 47 legacy product category strings to 12 canonical ML categories",
            column="product_category",
            rule_type="category_map",
            parameters={"mapping_table": "category_canonical_map"},
        ),
        BusinessRule(
            rule_id="timestamp_utc",
            description="Convert all timestamps to UTC ISO 8601 format",
            column="created_at",
            rule_type="fix_encoding",
            parameters={"target_format": "UTC_ISO8601"},
        ),
    ],

    grader_weights={
        "null_compliance":       0.25,
        "type_accuracy":         0.20,
        "duplicate_elimination": 0.15,
        "outlier_compliance":    0.20,
        "business_rule_score":   0.20,
    },
)