"""
env/tasks/task_monday_morning.py
================================
Task 1 — "The Analyst's Monday Morning"
Difficulty: EASY
Domain: Sales data for a marketing dashboard
Max Steps: 15
"""
from env.models import (
    TaskConfig, TaskName, ColumnCorruptionSpec,
    BusinessRule
)

TASK_CONFIG = TaskConfig(
    task_name=TaskName.MONDAY_MORNING,
    display_name="clean-ingest",
    difficulty="easy",
    max_steps=15,
    dataset_rows=500,
    dataset_columns=8,
    success_threshold=0.60,
    task_brief=(
        "A sales CSV arrived overnight. Marketing needs it ready by 9am for "
        "the revenue dashboard refresh.\n\n"
        "Requirements:\n"
        "- No null values in: age, revenue, signup_date, email\n"
        "- All dates must be in ISO format (YYYY-MM-DD)\n"
        "- Email addresses must be lowercase with no extra whitespace\n"
        "- Revenue values must be positive (0 to 100,000)\n"
        "- No duplicate customer records\n"
        "- The 'notes' column does not need cleaning"
    ),

    column_corruptions=[
        ColumnCorruptionSpec(
            column_name="age",
            null_rate=0.12,
            add_outliers=True,
            outlier_value=999,
        ),
        ColumnCorruptionSpec(
            column_name="revenue",
            null_rate=0.08,
        ),
        ColumnCorruptionSpec(
            column_name="signup_date",
            mixed_formats=True,      # 3 date formats mixed together
        ),
        ColumnCorruptionSpec(
            column_name="email",
            mixed_case=True,         # MiXeD CaSe + trailing whitespace
        ),
    ],

    global_duplicate_rate=0.046,     # ~23 duplicate rows in 500

    secondary_tables=[],
    business_rules=[],

    grader_weights={
        "null_compliance":       0.35,
        "type_accuracy":         0.15,
        "duplicate_elimination": 0.25,
        "outlier_compliance":    0.25,
        "business_rule_score":   0.00,
    },
)