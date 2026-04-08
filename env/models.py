"""
env/models.py
=============
All Pydantic v2 typed models for DataCleanEnv.

This file is the single source of truth for every data structure
that crosses a boundary in the system:
  - Agent ↔ Environment  (Observation, Action)
  - Environment ↔ Grader (EpisodeResult, GraderScore)
  - Environment ↔ HTTP   (ResetRequest, StepRequest, StateResponse)
  - Internal state       (ColumnProfile, EpisodeState, TaskConfig)

Rule: if two modules need to share a data structure, it lives here.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator, model_validator #ignore=True


# ─────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────

class DType(str, Enum):
    """Inferred data type of a column as seen by the agent."""
    INTEGER   = "integer"
    FLOAT     = "float"
    STRING    = "string"
    DATE      = "date"
    BOOLEAN   = "boolean"
    MIXED     = "mixed"      # multiple types coexisting — a cleaning signal


class TaskName(str, Enum):
    """The three environment tasks, easy → hard."""
    MONDAY_MORNING   = "monday_morning"    # Task 1 — Easy
    WAREHOUSE_MERGE  = "warehouse_merge"   # Task 2 — Medium
    DATA_LAKE_CRISIS = "data_lake_crisis"  # Task 3 — Hard


class ActionType(str, Enum):
    """
    All 18 discrete cleaning actions the agent can take.
    
    The agent selects one ActionType per step, plus an optional
    target column. Actions that target the whole dataset (like
    remove_exact_duplicates) ignore the column parameter.
    """
    # ── Imputation ──────────────────────────────────────────────
    FILL_NULL_MEAN      = "fill_null_mean"       # fill nulls with mean
    FILL_NULL_MEDIAN    = "fill_null_median"      # fill nulls with median (outlier-robust)
    FILL_NULL_MODE      = "fill_null_mode"        # fill nulls with most frequent value
    FILL_NULL_FORWARD   = "fill_null_forward"     # forward-fill (time-series)
    DROP_ROWS_WITH_NULL = "drop_rows_with_null"   # drop rows where column is null

    # ── Type & Format ────────────────────────────────────────────
    CAST_TO_INTEGER         = "cast_to_integer"
    CAST_TO_FLOAT           = "cast_to_float"
    CAST_TO_STRING          = "cast_to_string"
    PARSE_DATES             = "parse_dates"             # standardize all date formats → ISO
    NORMALIZE_CATEGORIES    = "normalize_categories"    # lowercase + strip whitespace

    # ── Deduplication ────────────────────────────────────────────
    REMOVE_EXACT_DUPLICATES = "remove_exact_duplicates"   # whole dataset, no column needed
    REMOVE_NEAR_DUPLICATES  = "remove_near_duplicates"    # fuzzy match on key column

    # ── Outlier Handling ─────────────────────────────────────────
    CLIP_OUTLIERS_IQR    = "clip_outliers_iqr"    # clip to 1.5×IQR range
    REMOVE_OUTLIER_ROWS  = "remove_outlier_rows"  # drop rows with extreme values

    # ── Structural (Task 2+) ─────────────────────────────────────
    MERGE_TABLE          = "merge_table"          # join an available secondary table
    APPLY_BUSINESS_RULE  = "apply_business_rule"  # apply a domain-specific rule

    # ── Terminal ─────────────────────────────────────────────────
    SUBMIT = "submit"   # end episode, trigger grader


# ─────────────────────────────────────────────────────────────────
# OBSERVATION SPACE — what the agent sees
# ─────────────────────────────────────────────────────────────────

class ValueRange(BaseModel):
    """Numeric statistics for a column. None for non-numeric columns."""
    min:  Optional[float] = None
    max:  Optional[float] = None
    mean: Optional[float] = None
    std:  Optional[float] = None


class TopValue(BaseModel):
    """One of the most frequent values in a column."""
    value: Any
    count: int
    frequency: float = Field(ge=0.0, le=1.0)


class ColumnProfile(BaseModel):
    """
    Statistical profile of a single column.
    
    This is all the agent ever sees about a column — no raw values.
    The agent must infer what cleaning is needed from these statistics.
    That inference is the learned skill.
    """
    column_name:         str
    inferred_dtype:      DType
    null_rate:           float = Field(ge=0.0, le=1.0,  description="Fraction of missing values")
    unique_rate:         float = Field(ge=0.0, le=1.0,  description="Fraction of unique values")
    outlier_score:       float = Field(ge=0.0, le=1.0,  description="IQR-based outlier severity")
    format_consistency:  float = Field(ge=0.0, le=1.0,  description="How consistent the format is (1.0 = fully consistent)")
    value_range:         Optional[ValueRange] = None     # populated for numeric columns
    top_values:          List[TopValue]       = Field(default_factory=list, max_length=5)
    row_count:           int                  = Field(ge=0)


class LastActionResult(BaseModel):
    """What happened after the last step. Helps agent track its own progress."""
    action_taken:      str
    column_affected:   Optional[str]   = None
    rows_affected:     int             = 0
    columns_affected:  List[str]       = Field(default_factory=list)
    message:           str             = ""       # human-readable result description
    was_effective:     bool            = True     # False if action had zero effect


class Observation(BaseModel):
    """
    Full observation returned to the agent after reset() or step().
    
    The agent receives this and must decide which action to take next.
    Ground truth is NEVER included here — only statistics and profile.
    """
    # ── Dataset-level stats ──────────────────────────────────────
    total_rows:           int  = Field(ge=0)
    total_columns:        int  = Field(ge=0)
    duplicate_row_count:  int  = Field(ge=0)
    step_number:          int  = Field(ge=0)
    max_steps:            int  = Field(ge=1)
    steps_remaining:      int  = Field(ge=0)

    # ── Task context ─────────────────────────────────────────────
    task_name:   TaskName
    task_brief:  str   = Field(description="Human-readable cleaning goal / purpose statement")

    # ── Column profiles ──────────────────────────────────────────
    columns: List[ColumnProfile] = Field(default_factory=list)

    # ── Multi-table context (Task 2+) ────────────────────────────
    available_tables:  List[str] = Field(default_factory=list,  description="Secondary tables available for merge")
    available_rules:   List[str] = Field(default_factory=list,  description="Named business rules available to apply")

    # ── Feedback from last action ────────────────────────────────
    last_action_result: Optional[LastActionResult] = None

    # ── Episode health ───────────────────────────────────────────
    done: bool = False


# ─────────────────────────────────────────────────────────────────
# ACTION SPACE — what the agent does
# ─────────────────────────────────────────────────────────────────

class Action(BaseModel):
    """
    One cleaning operation the agent wants to apply.
    
    Every step, the agent sends exactly one Action.
    The environment applies it and returns the updated Observation.
    
    Column-targeting actions require `column` to be set.
    Dataset-wide actions (remove_exact_duplicates, submit) ignore `column`.
    Merge actions require `table_name`. Business rule actions require `rule_id`.
    """
    action_type:  ActionType
    column:       Optional[str] = None   # target column (most actions)
    table_name:   Optional[str] = None   # for MERGE_TABLE
    rule_id:      Optional[str] = None   # for APPLY_BUSINESS_RULE

    @model_validator(mode="after")
    def validate_action_parameters(self) -> "Action":
        """Ensure actions that need parameters have them."""
        column_required = {
            ActionType.FILL_NULL_MEAN,
            ActionType.FILL_NULL_MEDIAN,
            ActionType.FILL_NULL_MODE,
            ActionType.FILL_NULL_FORWARD,
            ActionType.DROP_ROWS_WITH_NULL,
            ActionType.CAST_TO_INTEGER,
            ActionType.CAST_TO_FLOAT,
            ActionType.CAST_TO_STRING,
            ActionType.PARSE_DATES,
            ActionType.NORMALIZE_CATEGORIES,
            ActionType.REMOVE_NEAR_DUPLICATES,
            ActionType.CLIP_OUTLIERS_IQR,
            ActionType.REMOVE_OUTLIER_ROWS,
        }
        if self.action_type in column_required and not self.column:
            raise ValueError(
                f"Action '{self.action_type}' requires a 'column' parameter."
            )
        if self.action_type == ActionType.MERGE_TABLE and not self.table_name:
            raise ValueError("MERGE_TABLE requires 'table_name'.")
        if self.action_type == ActionType.APPLY_BUSINESS_RULE and not self.rule_id:
            raise ValueError("APPLY_BUSINESS_RULE requires 'rule_id'.")
        return self

    def to_log_string(self) -> str:
        """Compact string for [STEP] log lines in inference.py."""
        if self.action_type == ActionType.SUBMIT:
            return "submit()"
        if self.action_type == ActionType.REMOVE_EXACT_DUPLICATES:
            return "remove_exact_duplicates()"
        if self.column:
            return f"{self.action_type.value}({self.column})"
        if self.table_name:
            return f"merge_table({self.table_name})"
        if self.rule_id:
            return f"apply_business_rule({self.rule_id})"
        return self.action_type.value


# ─────────────────────────────────────────────────────────────────
# STEP RESULT — what the environment returns from step()
# ─────────────────────────────────────────────────────────────────

class StepResult(BaseModel):
    """
    Everything the environment returns after processing one Action.
    This is what the training loop unpacks as (obs, reward, done, info).
    """
    observation:  Observation
    reward:       float = Field(description="Step-level reward, clipped to [-1.0, 1.0]")
    done:         bool  = False
    info:         Dict[str, Any] = Field(default_factory=dict)

    @field_validator("reward")
    @classmethod
    def reward_must_be_finite(cls, v: float) -> float:
        import math
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Reward must be a finite number.")
        return round(float(v), 4)


# ─────────────────────────────────────────────────────────────────
# GRADER OUTPUT — the terminal score at submit()
# ─────────────────────────────────────────────────────────────────

class GraderScore(BaseModel):
    """
    Deterministic score produced by comparing agent's cleaned dataset
    to the hidden ground truth. All components are in [0.0, 1.0].
    """
    # ── Component scores ─────────────────────────────────────────
    null_compliance:          float = Field(ge=0.0, le=1.0)
    type_accuracy:            float = Field(ge=0.0, le=1.0)
    duplicate_elimination:    float = Field(ge=0.0, le=1.0)
    outlier_compliance:       float = Field(ge=0.0, le=1.0)
    business_rule_score:      float = Field(ge=0.0, le=1.0, default=1.0)

    # ── Final weighted score ─────────────────────────────────────
    total_score: float = Field(ge=0.0, le=1.0)

    # ── Metadata ─────────────────────────────────────────────────
    task_name:    TaskName
    steps_taken:  int
    seed:         int
    notes:        List[str] = Field(default_factory=list)  # human-readable breakdown


# ─────────────────────────────────────────────────────────────────
# EPISODE STATE — internal tracking (not sent to agent directly)
# ─────────────────────────────────────────────────────────────────

class EpisodeState(BaseModel):
    """
    Full internal state of a running episode.
    
    This is what state() returns. The agent does NOT receive this
    during normal play — it receives only Observation. This is for
    debuggers, evaluators, and the HuggingFace validator.
    """
    task_name:       TaskName
    seed:            int
    step_count:      int = 0
    max_steps:       int = 15
    done:            bool = False

    # ── Reward tracking ──────────────────────────────────────────
    cumulative_reward:   float = 0.0
    step_rewards:        List[float] = Field(default_factory=list)

    # ── Action history ───────────────────────────────────────────
    action_history: List[Dict[str, Any]] = Field(default_factory=list)

    # ── Dataset shape tracking ───────────────────────────────────
    initial_row_count:   int = 0
    current_row_count:   int = 0
    initial_null_total:  int = 0
    current_null_total:  int = 0

    # ── Grader result (populated at submit) ──────────────────────
    final_score: Optional[GraderScore] = None

    class Config:
        # DataFrames cannot be stored in Pydantic — they live in
        # DataCleanEnv instance directly. This model tracks metadata only.
        arbitrary_types_allowed = True


# ─────────────────────────────────────────────────────────────────
# HTTP REQUEST / RESPONSE MODELS — server layer contracts
# ─────────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    """Body for POST /reset"""
    task: TaskName = TaskName.MONDAY_MORNING
    seed: Optional[int] = None   # None = random seed each call


class StepRequest(BaseModel):
    """Body for POST /step"""
    action_type: ActionType
    column:      Optional[str] = None
    table_name:  Optional[str] = None
    rule_id:     Optional[str] = None

    def to_action(self) -> Action:
        return Action(
            action_type=self.action_type,
            column=self.column,
            table_name=self.table_name,
            rule_id=self.rule_id,
        )


class ResetResponse(BaseModel):
    """Response from POST /reset"""
    observation: Observation
    info: Dict[str, Any] = Field(default_factory=dict)


class StepResponse(BaseModel):
    """Response from POST /step"""
    observation: Observation
    reward:      float
    done:        bool
    info:        Dict[str, Any] = Field(default_factory=dict)


class StateResponse(BaseModel):
    """Response from GET /state — full internal snapshot"""
    episode_state:  EpisodeState
    observation:    Optional[Observation] = None
    is_active:      bool = False   # False if no episode is running


class HealthResponse(BaseModel):
    """Response from GET /health"""
    status:       str = "ok"
    environment:  str = "dataclean-env"
    version:      str = "1.0.0"
    tasks:        List[str] = Field(
        default_factory=lambda: [t.value for t in TaskName]
    )


class TaskListResponse(BaseModel):
    """Response from GET /tasks"""
    tasks: List[Dict[str, Any]]


# ─────────────────────────────────────────────────────────────────
# TASK CONFIGURATION — defines each task's parameters
# ─────────────────────────────────────────────────────────────────

class ColumnCorruptionSpec(BaseModel):
    """
    Defines how a single column should be dirtied by the corruption engine.
    Used internally by dataset_generator.py.
    """
    column_name:      str
    null_rate:        float = Field(ge=0.0, le=1.0, default=0.0)
    add_outliers:     bool  = False
    outlier_value:    Optional[Any] = None          # e.g. 999 for age
    mixed_formats:    bool  = False                  # e.g. date format mixing
    mixed_case:       bool  = False                  # e.g. EMAIL vs email
    duplicate_rate:   float = Field(ge=0.0, le=1.0, default=0.0)


class BusinessRule(BaseModel):
    """
    A named domain rule that can be applied via APPLY_BUSINESS_RULE action.
    Used in Task 2 and Task 3.
    """
    rule_id:      str
    description:  str
    column:       str
    rule_type:    Literal["normalize_units", "validate_range", "fix_encoding", "category_map"]
    parameters:   Dict[str, Any] = Field(default_factory=dict)


class TaskConfig(BaseModel):
    """
    Full specification for one task. Loaded by the task files in env/tasks/.
    Drives dataset generation, grader weights, and episode parameters.
    """
    task_name:         TaskName
    display_name:      str
    difficulty:        Literal["easy", "medium", "hard"]
    max_steps:         int
    dataset_rows:      int
    dataset_columns:   int
    task_brief:        str                            # shown to agent
    success_threshold: float = Field(ge=0.0, le=1.0) # minimum score to count as success

    # ── Corruption spec ──────────────────────────────────────────
    column_corruptions:  List[ColumnCorruptionSpec] = Field(default_factory=list)
    global_duplicate_rate: float = Field(ge=0.0, le=1.0, default=0.05)

    # ── Multi-table (Task 2+) ────────────────────────────────────
    secondary_tables:   List[str]         = Field(default_factory=list)
    business_rules:     List[BusinessRule] = Field(default_factory=list)

    # ── Grader weights (must sum to 1.0) ─────────────────────────
    grader_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "null_compliance":       0.30,
            "type_accuracy":         0.20,
            "duplicate_elimination": 0.20,
            "outlier_compliance":    0.15,
            "business_rule_score":   0.15,
        }
    )

    @field_validator("grader_weights")
    @classmethod
    def weights_must_sum_to_one(cls, v: Dict[str, float]) -> Dict[str, float]:
        total = round(sum(v.values()), 6)
        if abs(total - 1.0) > 1e-4:
            raise ValueError(f"Grader weights must sum to 1.0, got {total}.")
        return v