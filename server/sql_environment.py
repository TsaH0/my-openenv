"""
server/sql_environment.py - Core SQL Query Learning Environment.

Implements the OpenEnv Environment interface with:
  - reset(): Initialize a fresh episode with a SQLite database
  - step(action): Execute an agent SQL query, grade it, return observation
  - state: Current episode metadata

The environment hosts an in-memory SQLite database seeded with e-commerce
data across 4 tables (customers, products, orders, order_items).
"""

from __future__ import annotations

import sqlite3
import uuid
from typing import Optional

from models import SQLAction, SQLObservation, SQLState
from server.tasks import TASKS, SCHEMA_INFO, grade, seed_database


# Max steps per episode before forced termination
MAX_STEPS = 20


class SQLEnvironment:
    """
    SQL Query Learning Environment.

    At each step the agent submits a SQL query string. The environment:
      1. Executes the query against the local SQLite database.
      2. Compares the result against the reference answer for the active task.
      3. Returns an observation with the result, reward, and feedback.

    Episode flow:
      reset(difficulty='easy'|'medium'|'hard') -> SQLObservation
      step(SQLAction)                          -> SQLObservation
      state                                    -> SQLState
    """

    def __init__(self) -> None:
        self._conn: Optional[sqlite3.Connection] = None
        self._state = SQLState()
        self._current_task: Optional[dict] = None
        self._task_ids_by_difficulty = {
            "easy":   ["easy_1", "easy_2", "easy_3"],
            "medium": ["medium_1", "medium_2", "medium_3"],
            "hard":   ["hard_1", "hard_2", "hard_3"],
        }
        self._task_index: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, difficulty: str = "easy", task_id: Optional[str] = None) -> SQLObservation:
        """
        Start a new episode.

        Args:
            difficulty: Task difficulty tier ('easy', 'medium', 'hard').
            task_id: Optional specific task ID to start with.

        Returns:
            Initial SQLObservation with schema info and task description.
        """
        # Fresh in-memory SQLite database for each episode
        if self._conn:
            self._conn.close()
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        seed_database(self._conn)

        # Select the first task for the given difficulty
        if task_id and task_id in TASKS:
            self._current_task = TASKS[task_id]
        else:
            ids = self._task_ids_by_difficulty.get(difficulty, ["easy_1"])
            self._task_index = 0
            self._current_task = TASKS[ids[0]]

        # Reset state
        self._state = SQLState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            current_task_id=self._current_task["id"],
            current_difficulty=self._current_task["difficulty"],
            total_reward=0.0,
            tasks_completed=0,
            best_reward=0.0,
            last_query="",
            last_error="",
        )

        return SQLObservation(
            result=[],
            error="",
            reward=0.0,
            done=False,
            message=(
                "Episode started. Write a SQL query to complete the task.\n"
                "You can explore the schema using: "
                "SELECT name FROM sqlite_master WHERE type='table';"
            ),
            schema_info=SCHEMA_INFO,
            task_description=self._current_task["description"],
            expected_columns=self._current_task["expected_columns"],
            step_count=0,
            score_breakdown={},
        )

    def step(self, action: SQLAction) -> SQLObservation:
        """
        Execute a SQL query and return graded observation.

        Args:
            action: SQLAction with the query string to execute.

        Returns:
            SQLObservation with result rows, reward, and feedback.
        """
        if self._conn is None or self._current_task is None:
            return SQLObservation(
                error="Environment not initialised. Call reset() first.",
                reward=0.0,
                done=True,
                message="Call reset() to start a new episode.",
            )

        self._state.step_count += 1
        self._state.last_query = action.query

        # Determine which task to grade
        task_id = action.task_id or self._state.current_task_id

        # Override difficulty if requested
        if action.difficulty and action.difficulty != self._state.current_difficulty:
            ids = self._task_ids_by_difficulty.get(action.difficulty, ["easy_1"])
            task_id = ids[0]
            self._current_task = TASKS.get(task_id, self._current_task)
            self._state.current_task_id = task_id
            self._state.current_difficulty = action.difficulty

        # Grade the query
        reward, message, agent_rows, expected_rows = grade(
            task_id, action.query, self._conn
        )

        # Update state
        self._state.total_reward += reward
        if reward > self._state.best_reward:
            self._state.best_reward = reward

        # Perfect score: advance to next task (if any)
        done = False
        if reward >= 1.0:
            self._state.tasks_completed += 1
            message += " Moving to next task..."
            advanced = self._advance_task()
            if not advanced:
                done = True
                message = "All tasks completed! Episode done."

        # Force done after max steps
        if self._state.step_count >= MAX_STEPS:
            done = True
            message += f" Episode ended (max {MAX_STEPS} steps reached)."

        self._state.last_error = ""
        score_breakdown = {
            "correctness": round(reward * 0.7 / max(reward, 0.001), 4) if reward > 0 else 0.0,
            "keyword_bonus": 0.1,
            "efficiency_bonus": round(reward - min(reward * 0.7, 0.7), 4),
        }

        return SQLObservation(
            result=agent_rows,
            error="",
            reward=reward,
            done=done,
            message=message,
            schema_info=SCHEMA_INFO,
            task_description=self._current_task["description"] if not done else "Episode complete.",
            expected_columns=self._current_task["expected_columns"] if not done else [],
            step_count=self._state.step_count,
            score_breakdown=score_breakdown,
        )

    @property
    def state(self) -> SQLState:
        """Return current episode state."""
        return self._state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _advance_task(self) -> bool:
        """
        Move to the next task in the current difficulty tier.

        Returns True if advanced, False if no more tasks remain.
        """
        difficulty = self._state.current_difficulty
        ids = self._task_ids_by_difficulty.get(difficulty, [])
        self._task_index += 1
        if self._task_index < len(ids):
            next_id = ids[self._task_index]
            self._current_task = TASKS[next_id]
            self._state.current_task_id = next_id
            return True
        return False

    def close(self) -> None:
        """Close the SQLite connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
