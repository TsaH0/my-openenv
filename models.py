"""
models.py - Typed Pydantic models for the SQL Query Learning Environment.

Defines Action, Observation, and State models compatible with the OpenEnv specification.
Uses Pydantic BaseModel for typed validation. Falls back gracefully if openenv-core
is not installed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Try to import OpenEnv base classes for full spec compliance
try:
    from openenv.core.env_server.interfaces import (
        Action as _BaseAction,
        Observation as _BaseObservation,
        State as _BaseState,
    )
except ImportError:
    # openenv-core not installed — use plain Pydantic
    _BaseAction = BaseModel
    _BaseObservation = BaseModel
    _BaseState = BaseModel


class SQLAction(_BaseAction):
    """
    Action: a SQL query submitted by the agent.

    Fields:
        query: The SQL query string to execute.
        task_id: Optional task ID to target (defaults to current task).
        difficulty: Task difficulty tier — 'easy', 'medium', or 'hard'.
    """

    query: str
    task_id: Optional[str] = None
    difficulty: str = "easy"


class SQLObservation(_BaseObservation):
    """
    Observation returned after executing a SQL action.

    Fields:
        result: Rows returned by the query (list of dicts).
        error: Error message if the query failed, else empty.
        reward: Step reward in [0.0, 1.0].
        done: Whether the episode is finished.
        message: Human-readable grader feedback.
        schema_info: Database schema description.
        task_description: Natural-language task objective.
        expected_columns: Column names the answer should include.
        step_count: Steps taken so far in this episode.
        score_breakdown: Sub-scores (correctness, efficiency, etc.).
    """

    result: List[Dict[str, Any]] = Field(default_factory=list)
    error: str = ""
    reward: float = 0.0
    done: bool = False
    message: str = ""
    schema_info: str = ""
    task_description: str = ""
    expected_columns: List[str] = Field(default_factory=list)
    step_count: int = 0
    score_breakdown: Dict[str, float] = Field(default_factory=dict)


class SQLState(_BaseState):
    """
    Full state of an ongoing episode.

    Fields:
        episode_id: Unique episode identifier.
        step_count: Steps taken so far.
        current_task_id: Active task ID.
        current_difficulty: Active difficulty tier.
        total_reward: Cumulative reward.
        tasks_completed: Count of tasks solved.
        best_reward: Best single-step reward.
        last_query: Most recent SQL query submitted.
        last_error: Most recent error (empty if none).
    """

    episode_id: str = ""
    step_count: int = 0
    current_task_id: str = ""
    current_difficulty: str = "easy"
    total_reward: float = 0.0
    tasks_completed: int = 0
    best_reward: float = 0.0
    last_query: str = ""
    last_error: str = ""
