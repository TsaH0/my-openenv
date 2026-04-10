"""
client.py - EnvClient for the SQL Query Learning Environment.

Provides a Python client that can connect to the environment server
over HTTP, compatible with the OpenEnv client interface.

Usage:
    from client import SQLEnvClient

    client = SQLEnvClient(base_url="http://localhost:7860")
    obs = client.reset(difficulty="easy")
    print(obs.task_description)

    action = SQLAction(query="SELECT name, email FROM customers WHERE country='USA'")
    obs = client.step(action)
    print(f"Reward: {obs.reward}")
    print(f"Result: {obs.result}")
"""

from __future__ import annotations

import json
import requests
from typing import Optional

from models import SQLAction, SQLObservation, SQLState


class SQLEnvClient:
    """
    HTTP client for the SQL Query Learning Environment.

    Compatible with the OpenEnv EnvClient interface pattern.
    Supports reset(), step(), and state() methods.
    """

    def __init__(self, base_url: str = "http://localhost:7860", session_id: str = "default") -> None:
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, difficulty: str = "easy", task_id: Optional[str] = None) -> SQLObservation:
        """
        Reset the environment for a new episode.

        Args:
            difficulty: Task tier - 'easy', 'medium', or 'hard'.
            task_id: Optional specific task to start with.

        Returns:
            Initial SQLObservation.
        """
        payload = {"difficulty": difficulty}
        if task_id:
            payload["task_id"] = task_id

        resp = self._session.post(
            f"{self.base_url}/reset",
            json=payload,
            params={"session_id": self.session_id},
        )
        resp.raise_for_status()
        return self._parse_observation(resp.json())

    def step(self, action: SQLAction) -> SQLObservation:
        """
        Execute a SQL query action.

        Args:
            action: SQLAction containing the query string.

        Returns:
            SQLObservation with graded results.
        """
        payload = {
            "query": action.query,
            "difficulty": action.difficulty or "easy",
        }
        if action.task_id:
            payload["task_id"] = action.task_id

        resp = self._session.post(
            f"{self.base_url}/step",
            json=payload,
            params={"session_id": self.session_id},
        )
        resp.raise_for_status()
        return self._parse_observation(resp.json())

    def state(self) -> SQLState:
        """
        Fetch current episode state.

        Returns:
            SQLState with episode metadata.
        """
        resp = self._session.get(
            f"{self.base_url}/state",
            params={"session_id": self.session_id},
        )
        resp.raise_for_status()
        data = resp.json()
        return SQLState(**data)

    def health(self) -> bool:
        """Return True if the server is reachable and healthy."""
        try:
            resp = self._session.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "SQLEnvClient":
        return self

    def __exit__(self, *args) -> None:
        self._session.close()

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_observation(data: dict) -> SQLObservation:
        return SQLObservation(
            result=data.get("result", []),
            error=data.get("error", ""),
            reward=float(data.get("reward", 0.0)),
            done=bool(data.get("done", False)),
            message=data.get("message", ""),
            schema_info=data.get("schema_info", ""),
            task_description=data.get("task_description", ""),
            expected_columns=data.get("expected_columns", []),
            step_count=int(data.get("step_count", 0)),
            score_breakdown=data.get("score_breakdown", {}),
        )
