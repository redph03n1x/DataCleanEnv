"""
server/app.py
==============
FastAPI application. Maps HTTP endpoints to DataCleanEnv methods.

  POST /reset    → env.reset()
  POST /step     → env.step()
  GET  /state    → env.state()
  GET  /health   → alive check
  GET  /tasks    → list available tasks

One env instance per process. Episodes are stateful (reset → step → step → submit).
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from env.dataclean_env import DataCleanEnv, TASK_REGISTRY
from env.models import (
    HealthResponse,
    ResetRequest, ResetResponse,
    StateResponse,
    StepRequest, StepResponse,
    TaskListResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("dataclean-env")

# ── Single environment instance (shared across requests) ──────────
_env: DataCleanEnv = DataCleanEnv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("DataCleanEnv server starting up.")
    yield
    logger.info("DataCleanEnv server shutting down.")


app = FastAPI(
    title="DataCleanEnv",
    description=(
        "OpenEnv-compatible RL environment for data cleaning. "
        "Agents learn to clean dirty datasets toward a stated purpose."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────
# MIDDLEWARE — request timing log
# ─────────────────────────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.1f}ms)")
    return response


# ─────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["meta"])
async def health():
    """Ping endpoint. Must return 200 for HuggingFace validator."""
    return HealthResponse()


@app.get("/tasks", response_model=TaskListResponse, tags=["meta"])
async def list_tasks():
    """List available tasks with their configurations."""
    tasks = []
    for name, config in TASK_REGISTRY.items():
        tasks.append({
            "task_name":         config.task_name.value,
            "display_name":      config.display_name,
            "difficulty":        config.difficulty,
            "max_steps":         config.max_steps,
            "dataset_rows":      config.dataset_rows,
            "dataset_columns":   config.dataset_columns,
            "success_threshold": config.success_threshold,
            "has_business_rules": len(config.business_rules) > 0,
            "secondary_tables":  config.secondary_tables,
        })
    return TaskListResponse(tasks=tasks)


@app.post("/reset", response_model=ResetResponse, tags=["env"])
async def reset(body: ResetRequest = ResetRequest()):
    """
    Start a new episode. Returns the initial observation.

    Called once at the beginning of each episode.
    Generates a fresh (dirty, clean) dataset pair using the given seed.
    If seed is null, a random training seed is chosen.
    """
    try:
        obs = _env.reset(task=body.task, seed=body.seed)
        return ResetResponse(
            observation=obs,
            info={
                "task":      obs.task_name.value,
                "seed":      _env.state().seed,
                "max_steps": obs.max_steps,
            },
        )
    except Exception as exc:
        logger.exception("reset() failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/step", response_model=StepResponse, tags=["env"])
async def step(body: StepRequest):
    """
    Apply one cleaning action. Returns (observation, reward, done, info).

    Called repeatedly during an episode until done=true.
    When the agent calls submit(), the grader runs and final reward is returned.
    """
    try:
        action = body.to_action()
        result = _env.step(action)
        return StepResponse(
            observation=result.observation,
            reward=result.reward,
            done=result.done,
            info=result.info,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("step() failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/state", response_model=StateResponse, tags=["env"])
async def state():
    """
    Return the full internal episode state.

    Used by evaluators and debuggers — not by agents during normal play.
    Returns is_active=false if no episode is running.
    """
    episode_state = _env.state()
    is_active     = (episode_state.seed != 0 and not episode_state.done)
    try:
        obs = _env._build_observation(last_action_result=None)
    except Exception:
        obs = None
    return StateResponse(
        episode_state=episode_state,
        observation=obs,
        is_active=is_active,
    )


# ─────────────────────────────────────────────────────────────────
# GLOBAL ERROR HANDLER
# ─────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception on {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "path": str(request.url.path)},
    )