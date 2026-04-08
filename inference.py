"""
inference.py
============
Baseline inference script for DataCleanEnv.

Reads credentials from environment variables:
  API_BASE_URL  — LLM API endpoint
  MODEL_NAME    — model identifier
  HF_TOKEN      — HuggingFace / API key

Runs one episode per task (all 3) and emits mandatory
[START] / [STEP] / [END] log lines to stdout.

Usage:
  export API_BASE_URL="https://router.huggingface.co/v1"
  export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
  export HF_TOKEN="hf_..."
  python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import List, Optional

import httpx
from openai import OpenAI

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
ENV_URL      = os.getenv("ENV_URL",      "http://localhost:7860")   # or HF Space URL
BENCHMARK    = "dataclean-env"

MAX_STEPS   = 20          # safety ceiling per episode
TEMPERATURE = 0.2         # low = more deterministic baseline

TASKS = [
    {"task": "monday_morning",   "seed": 42},
    {"task": "warehouse_merge",  "seed": 42},
    {"task": "data_lake_crisis", "seed": 42},
]


# ─────────────────────────────────────────────────────────────────
# LOGGING HELPERS  (mandatory format — do not change field names)
# ─────────────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err_val  = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={err_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────────
# ENVIRONMENT HTTP HELPERS
# ─────────────────────────────────────────────────────────────────

def env_reset(task: str, seed: int) -> dict:
    resp = httpx.post(
        f"{ENV_URL}/reset",
        json={"task": task, "seed": seed},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action_type: str, column: Optional[str] = None,
             table_name: Optional[str] = None, rule_id: Optional[str] = None) -> dict:
    payload = {"action_type": action_type}
    if column:     payload["column"]     = column
    if table_name: payload["table_name"] = table_name
    if rule_id:    payload["rule_id"]    = rule_id
    resp = httpx.post(f"{ENV_URL}/step", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
You are a data cleaning agent operating on a dirty dataset.
Each turn you receive the current dataset profile (statistics only — no raw values)
and must choose exactly one cleaning action.

CRITICAL RULES:
- NEVER repeat an action you already took on the same column
- Call submit() when all required columns are clean OR you have 3 steps left
- Do NOT clean columns not mentioned in the task brief

Available actions:
  fill_null_mean(column)         — fill nulls with mean (numeric columns)
  fill_null_median(column)       — fill nulls with median (use when outliers present)
  fill_null_mode(column)         — fill nulls with most frequent value (categorical)
  fill_null_forward(column)      — forward fill (time-series columns)
  drop_rows_with_null(column)    — drop rows where column is null (use sparingly)
  cast_to_integer(column)        — convert to integer type
  cast_to_float(column)          — convert to float type
  cast_to_string(column)         — convert to string type
  parse_dates(column)            — standardize all date formats to ISO
  normalize_categories(column)   — lowercase + strip whitespace
  remove_exact_duplicates()      — remove identical rows (no column needed)
  remove_near_duplicates(column) — fuzzy deduplication on key column
  clip_outliers_iqr(column)      — clip values to 1.5xIQR range
  remove_outlier_rows(column)    — drop rows with extreme outlier values
  merge_table(table_name)        — join a secondary table (Task 2+)
  apply_business_rule(rule_id)   — apply a named business rule (Task 2+)
  submit()                       — end episode, receive final score

Strategy (follow this order strictly):
1. Remove duplicates FIRST
2. Fix formats (parse_dates, normalize_categories)
3. Clip outliers BEFORE imputing
4. Impute nulls (fill_null_median for numeric, fill_null_mode for categorical)
5. Merge tables if available_tables is non-empty
6. Apply business rules if available_rules is non-empty
7. Call submit() when done or 3 steps remaining

Respond with ONLY a JSON object, no other text:
{"action_type": "...", "column": "..." or null, "table_name": "..." or null, "rule_id": "..." or null}
""").strip()


# ─────────────────────────────────────────────────────────────────
# AGENT DECISION
# ─────────────────────────────────────────────────────────────────

def build_user_prompt(obs: dict, step: int, last_reward: float, history: list) -> str:
    cols = obs.get("columns", [])
    col_lines = []
    for c in cols:
        col_lines.append(
            f"  {c['column_name']}: dtype={c['inferred_dtype']}, "
            f"null_rate={c['null_rate']:.3f}, "
            f"outlier_score={c['outlier_score']:.3f}, "
            f"format_consistency={c['format_consistency']:.3f}"
        )

    last_action = obs.get("last_action_result")
    last_msg = last_action.get("message", "") if last_action else "None"

    # Build history block so agent knows what it already did
    history_block = "\n".join(f"  Step {i+1}: {h}" for i, h in enumerate(history)) if history else "  None yet"

    steps_remaining = obs.get("steps_remaining", 0)
    submit_warning = "\n⚠️  CALL submit() NOW - only 3 steps left!" if steps_remaining <= 3 else ""

    return textwrap.dedent(f"""
    STEP {step} | Steps remaining: {steps_remaining}{submit_warning}
    Last reward: {last_reward:.2f}

    Task brief:
    {obs.get('task_brief', '')}

    Column profiles:
    {chr(10).join(col_lines)}

    Actions already taken (DO NOT REPEAT THESE):
    {history_block}

    Last action result: {last_msg}

    Choose your next action (JSON only):
    """).strip()


def decide_action(client: OpenAI, obs: dict, step: int, last_reward: float, history: list) -> dict:
    user_prompt = build_user_prompt(obs, step, last_reward, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1,        # lower = less random
            max_tokens=100,
        )
        raw = (completion.choices[0].message.content or "").strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as exc:
        print(f"[DEBUG] LLM parse error: {exc}", flush=True)
        return {"action_type": "submit", "column": None}


# ─────────────────────────────────────────────────────────────────
# RUN ONE EPISODE
# ─────────────────────────────────────────────────────────────────
def run_episode(client: OpenAI, task: str, seed: int) -> float:
    """Run a complete episode. Returns final score in [0, 1]."""
    rewards: List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False
    last_reward = 0.0
    action_history_log: List[str] = []          # ← ADDED: track history

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_resp = env_reset(task, seed)
        obs        = reset_resp["observation"]

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done", False):
                break

            action_dict = decide_action(                # ← CHANGED: pass history
                client, obs, step, last_reward, action_history_log
            )
            action_type = action_dict.get("action_type", "submit")
            column      = action_dict.get("column")
            table_name  = action_dict.get("table_name")
            rule_id     = action_dict.get("rule_id")

            # Build the log action string
            if action_type == "submit":
                action_str = "submit()"
            elif action_type == "remove_exact_duplicates":
                action_str = "remove_exact_duplicates()"
            elif column:
                action_str = f"{action_type}({column})"
            elif table_name:
                action_str = f"merge_table({table_name})"
            elif rule_id:
                action_str = f"apply_business_rule({rule_id})"
            else:
                action_str = action_type

            action_history_log.append(action_str)       # ← ADDED: record action

            try:
                step_resp   = env_step(action_type, column, table_name, rule_id)
                reward      = float(step_resp.get("reward", 0.0))
                done        = bool(step_resp.get("done", False))
                obs         = step_resp.get("observation", obs)
                error_msg   = None

                # Extract final score from info on submit
                if done and action_type == "submit":
                    score = float(step_resp.get("info", {}).get("final_score", reward))
                elif done:
                    score = float(sum(rewards) + reward) / max(len(rewards) + 1, 1)

            except Exception as exc:
                reward    = 0.0
                done      = True
                error_msg = str(exc)
                score     = float(sum(rewards)) / max(len(rewards), 1)

            rewards.append(reward)
            last_reward  = reward
            steps_taken  = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)

            if done:
                break

        score   = min(max(score, 0.0), 1.0)
        success = score >= 0.5

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        score   = float(sum(rewards)) / max(len(rewards), 1) if rewards else 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score
# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    if not API_KEY:
        print("[ERROR] HF_TOKEN or API_KEY environment variable not set.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    all_scores = []
    for task_cfg in TASKS:
        score = run_episode(client, task=task_cfg["task"], seed=task_cfg["seed"])
        all_scores.append(score)
        print(f"[INFO] {task_cfg['task']} score: {score:.3f}", flush=True)

    avg = sum(all_scores) / len(all_scores)
    print(f"\n[SUMMARY] Scores: {[round(s, 3) for s in all_scores]}", flush=True)
    print(f"[SUMMARY] Average: {avg:.3f}", flush=True)


if __name__ == "__main__":
    main()