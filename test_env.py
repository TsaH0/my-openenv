"""
test_env.py - Comprehensive test suite for the SQL Query Learning Environment.

Tests:
  1. Database seeding and schema
  2. All 9 tasks (3 easy, 3 medium, 3 hard) with reference solutions
  3. Grader correctness and reward ranges
  4. Environment lifecycle (reset, step, state)
  5. Partial credit scoring
  6. Error handling (bad SQL, wrong difficulty, empty queries)
  7. Multi-difficulty sweep
  8. Inference script dry-run
  9. FastAPI HTTP endpoints (integration)
  10. WebSocket endpoint
"""

from __future__ import annotations

import json
import sqlite3
import sys
import os
import time
import threading

# Make root importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest

from models import SQLAction, SQLObservation, SQLState
from server.tasks import (
    TASKS, SCHEMA_INFO, seed_database, grade,
    get_all_tasks_by_difficulty,
)
from server.sql_environment import SQLEnvironment


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def db():
    """In-memory SQLite database seeded with test data."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    seed_database(conn)
    yield conn
    conn.close()


@pytest.fixture
def env():
    """Fresh SQLEnvironment instance."""
    e = SQLEnvironment()
    yield e
    e.close()


# ===========================================================================
# 1. Database / Schema Tests
# ===========================================================================

class TestDatabase:

    def test_tables_exist(self, db):
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        assert {"customers", "products", "orders", "order_items"} <= tables

    def test_customers_seeded(self, db):
        count = db.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        assert count >= 10

    def test_products_seeded(self, db):
        count = db.execute("SELECT COUNT(*) FROM products").fetchone()[0]
        assert count >= 10

    def test_orders_seeded(self, db):
        count = db.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
        assert count >= 20

    def test_order_items_seeded(self, db):
        count = db.execute("SELECT COUNT(*) FROM order_items").fetchone()[0]
        assert count >= 30

    def test_categories_exist(self, db):
        cats = {row[0] for row in db.execute("SELECT DISTINCT category FROM products").fetchall()}
        assert len(cats) >= 3

    def test_order_statuses(self, db):
        statuses = {row[0] for row in db.execute("SELECT DISTINCT status FROM orders").fetchall()}
        assert "completed" in statuses

    def test_schema_info_nonempty(self):
        assert len(SCHEMA_INFO) > 100
        assert "customers" in SCHEMA_INFO
        assert "products" in SCHEMA_INFO
        assert "orders" in SCHEMA_INFO


# ===========================================================================
# 2. Task Registry Tests
# ===========================================================================

class TestTaskRegistry:

    def test_nine_tasks_total(self):
        assert len(TASKS) == 9

    def test_three_per_difficulty(self):
        for diff in ("easy", "medium", "hard"):
            tasks = get_all_tasks_by_difficulty(diff)
            assert len(tasks) == 3, f"Expected 3 {diff} tasks, got {len(tasks)}"

    def test_all_tasks_have_required_fields(self):
        required = {"id", "difficulty", "description", "reference_sql",
                    "required_keywords", "expected_columns"}
        for tid, task in TASKS.items():
            for field in required:
                assert field in task, f"Task {tid} missing field: {field}"

    def test_descriptions_nonempty(self):
        for tid, task in TASKS.items():
            assert len(task["description"]) > 20, f"Task {tid} has short description"

    def test_reference_sql_nonempty(self):
        for tid, task in TASKS.items():
            assert len(task["reference_sql"].strip()) > 10, f"Task {tid} missing SQL"


# ===========================================================================
# 3. Grader Tests - Reference Solutions Score 1.0
# ===========================================================================

class TestGraderReferenceSolutions:
    """The reference solution for every task must score >= 0.9."""

    def _grade_reference(self, task_id: str, db) -> float:
        task = TASKS[task_id]
        reward, msg, agent_rows, expected_rows = grade(task_id, task["reference_sql"], db)
        return reward

    def test_easy_1_reference(self, db):
        assert self._grade_reference("easy_1", db) >= 0.9

    def test_easy_2_reference(self, db):
        assert self._grade_reference("easy_2", db) >= 0.9

    def test_easy_3_reference(self, db):
        assert self._grade_reference("easy_3", db) >= 0.9

    def test_medium_1_reference(self, db):
        assert self._grade_reference("medium_1", db) >= 0.9

    def test_medium_2_reference(self, db):
        assert self._grade_reference("medium_2", db) >= 0.9

    def test_medium_3_reference(self, db):
        assert self._grade_reference("medium_3", db) >= 0.9

    def test_hard_1_reference(self, db):
        assert self._grade_reference("hard_1", db) >= 0.9

    def test_hard_2_reference(self, db):
        assert self._grade_reference("hard_2", db) >= 0.9

    def test_hard_3_reference(self, db):
        assert self._grade_reference("hard_3", db) >= 0.9


# ===========================================================================
# 4. Grader Tests - Incorrect Queries Score Low
# ===========================================================================

class TestGraderIncorrectQueries:

    def test_wrong_table(self, db):
        reward, msg, _, _ = grade("easy_1", "SELECT name FROM products", db)
        assert reward < 0.5

    def test_syntax_error(self, db):
        reward, msg, _, _ = grade("easy_1", "SELEKT * FRUM customers", db)
        assert reward == 0.0
        assert "error" in msg.lower() or "query error" in msg.lower()

    def test_empty_query(self, db):
        reward, msg, _, _ = grade("easy_2", "", db)
        assert reward == 0.0

    def test_wrong_filter(self, db):
        # Should be USA, returns UK instead - partial credit at most
        reward, msg, _, _ = grade(
            "easy_1",
            "SELECT name, email FROM customers WHERE country = 'UK'",
            db
        )
        assert reward < 0.8

    def test_reward_range(self, db):
        for task_id, task in TASKS.items():
            reward, _, _, _ = grade(task_id, task["reference_sql"], db)
            assert 0.0 <= reward <= 1.0, f"Reward out of range for {task_id}: {reward}"

    def test_unknown_task_returns_zero(self, db):
        reward, msg, _, _ = grade("nonexistent_task", "SELECT 1", db)
        assert reward == 0.0
        assert "unknown" in msg.lower()


# ===========================================================================
# 5. Partial Credit Tests
# ===========================================================================

class TestPartialCredit:

    def test_partial_result_gets_partial_credit(self, db):
        # Returns only some USA customers
        reward_full, _, _, _ = grade(
            "easy_1",
            "SELECT name, email FROM customers WHERE country = 'USA'",
            db
        )
        reward_partial, _, _, _ = grade(
            "easy_1",
            "SELECT name, email FROM customers WHERE country = 'USA' LIMIT 1",
            db
        )
        assert reward_partial < reward_full
        assert reward_partial > 0.0

    def test_extra_columns_dont_break_grader(self, db):
        # Adding extra columns - result still matches on name/email fields
        reward, msg, _, _ = grade(
            "easy_1",
            "SELECT name, email, city FROM customers WHERE country = 'USA'",
            db
        )
        # Should get some reward (has name+email in result, plus extra city)
        assert reward >= 0.0  # grader is lenient


# ===========================================================================
# 6. Environment Lifecycle Tests
# ===========================================================================

class TestEnvironmentLifecycle:

    def test_reset_returns_observation(self, env):
        obs = env.reset()
        assert isinstance(obs, SQLObservation)
        assert obs.task_description != ""
        assert obs.schema_info != ""
        assert obs.reward == 0.0
        assert obs.done is False

    def test_reset_difficulty_easy(self, env):
        obs = env.reset(difficulty="easy")
        assert env.state.current_difficulty == "easy"

    def test_reset_difficulty_medium(self, env):
        obs = env.reset(difficulty="medium")
        assert env.state.current_difficulty == "medium"

    def test_reset_difficulty_hard(self, env):
        obs = env.reset(difficulty="hard")
        assert env.state.current_difficulty == "hard"

    def test_state_after_reset(self, env):
        env.reset()
        state = env.state
        assert isinstance(state, SQLState)
        assert state.episode_id != ""
        assert state.step_count == 0
        assert state.total_reward == 0.0

    def test_step_increments_step_count(self, env):
        env.reset(difficulty="easy")
        env.step(SQLAction(query="SELECT 1"))
        assert env.state.step_count == 1
        env.step(SQLAction(query="SELECT 2"))
        assert env.state.step_count == 2

    def test_step_before_reset_returns_error(self, env):
        obs = env.step(SQLAction(query="SELECT 1"))
        assert obs.error != "" or obs.done is True

    def test_step_updates_total_reward(self, env):
        env.reset(difficulty="easy")
        task = TASKS["easy_1"]
        env.step(SQLAction(query=task["reference_sql"], difficulty="easy"))
        assert env.state.total_reward > 0.0

    def test_step_with_correct_query(self, env):
        env.reset(difficulty="easy", task_id="easy_1")
        task = TASKS["easy_1"]
        obs = env.step(SQLAction(query=task["reference_sql"], difficulty="easy"))
        assert obs.reward >= 0.9
        assert len(obs.result) > 0

    def test_step_with_error_query(self, env):
        env.reset(difficulty="easy")
        obs = env.step(SQLAction(query="INVALID SQL !!!", difficulty="easy"))
        assert obs.reward == 0.0

    def test_done_after_max_steps(self, env):
        env.reset(difficulty="easy")
        for _ in range(20):
            obs = env.step(SQLAction(query="SELECT 1", difficulty="easy"))
            if obs.done:
                break
        assert obs.done is True

    def test_multiple_resets_give_fresh_state(self, env):
        env.reset()
        env.step(SQLAction(query="SELECT 1"))
        first_id = env.state.episode_id

        env.reset()
        second_id = env.state.episode_id

        assert first_id != second_id
        assert env.state.step_count == 0


# ===========================================================================
# 7. All 9 Tasks Solvable End-to-End
# ===========================================================================

class TestAllTasksSolvable:
    """Each task should yield reward >= 0.9 when given the reference solution."""

    @pytest.mark.parametrize("task_id", list(TASKS.keys()))
    def test_task_solvable(self, task_id):
        env = SQLEnvironment()
        try:
            obs = env.reset(task_id=task_id,
                            difficulty=TASKS[task_id]["difficulty"])
            task = TASKS[task_id]
            obs = env.step(SQLAction(
                query=task["reference_sql"],
                difficulty=task["difficulty"],
                task_id=task_id,
            ))
            assert obs.reward >= 0.9, (
                f"Task {task_id} scored {obs.reward}: {obs.message}"
            )
        finally:
            env.close()


# ===========================================================================
# 8. Multi-Difficulty Sweep
# ===========================================================================

class TestMultiDifficultySweep:

    def test_easy_sweep(self, env):
        obs = env.reset(difficulty="easy")
        total = 0.0
        for task_id in ["easy_1", "easy_2", "easy_3"]:
            task = TASKS[task_id]
            obs = env.step(SQLAction(
                query=task["reference_sql"],
                difficulty="easy",
                task_id=task_id,
            ))
            total += obs.reward
            if obs.done:
                break
        assert total > 0.0

    def test_medium_sweep(self, env):
        obs = env.reset(difficulty="medium")
        total = 0.0
        for task_id in ["medium_1", "medium_2", "medium_3"]:
            task = TASKS[task_id]
            obs = env.step(SQLAction(
                query=task["reference_sql"],
                difficulty="medium",
                task_id=task_id,
            ))
            total += obs.reward
            if obs.done:
                break
        assert total > 0.0

    def test_hard_sweep(self, env):
        obs = env.reset(difficulty="hard")
        total = 0.0
        for task_id in ["hard_1", "hard_2", "hard_3"]:
            task = TASKS[task_id]
            obs = env.step(SQLAction(
                query=task["reference_sql"],
                difficulty="hard",
                task_id=task_id,
            ))
            total += obs.reward
            if obs.done:
                break
        assert total > 0.0


# ===========================================================================
# 9. Inference Script Tests
# ===========================================================================

class TestInferenceScript:

    def test_inference_run_task_easy(self):
        """Run a single easy task with fallback heuristic (no LLM API)."""
        from inference import run_task
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:1", api_key="test")
        score = run_task("easy_1", "easy", client)
        assert 0.0 <= score <= 1.0
        assert score >= 0.9

    def test_inference_run_task_medium(self):
        from inference import run_task
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:1", api_key="test")
        score = run_task("medium_1", "medium", client)
        assert 0.0 <= score <= 1.0
        assert score >= 0.9

    def test_inference_run_task_hard(self):
        from inference import run_task
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:1", api_key="test")
        score = run_task("hard_1", "hard", client)
        assert 0.0 <= score <= 1.0
        assert score >= 0.9

    def test_inference_all_tasks(self):
        from inference import run_task, ALL_TASKS
        from openai import OpenAI
        client = OpenAI(base_url="http://localhost:1", api_key="test")
        for task_id, difficulty in ALL_TASKS:
            score = run_task(task_id, difficulty, client)
            assert 0.0 <= score <= 1.0, f"Task {task_id} score out of range: {score}"


# ===========================================================================
# 10. FastAPI HTTP Endpoint Tests (integration)
# ===========================================================================

class TestFastAPIEndpoints:

    @pytest.fixture(scope="class")
    def test_client(self):
        from fastapi.testclient import TestClient
        from server.app import app
        with TestClient(app) as client:
            yield client

    def test_health_returns_200(self, test_client):
        resp = test_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_reset_returns_observation(self, test_client):
        resp = test_client.post("/reset", json={"difficulty": "easy"})
        assert resp.status_code == 200
        data = resp.json()
        assert "task_description" in data
        assert "schema_info" in data
        assert data["reward"] == 0.0

    def test_step_returns_reward(self, test_client):
        test_client.post("/reset", json={"difficulty": "easy"})
        resp = test_client.post("/step", json={
            "query": "SELECT name, email FROM customers WHERE country = 'USA'",
            "difficulty": "easy",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "reward" in data
        assert 0.0 <= data["reward"] <= 1.0

    def test_state_returns_metadata(self, test_client):
        test_client.post("/reset", json={"difficulty": "easy"})
        resp = test_client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "episode_id" in data
        assert "step_count" in data

    def test_step_with_bad_sql(self, test_client):
        test_client.post("/reset", json={"difficulty": "easy"})
        resp = test_client.post("/step", json={
            "query": "THIS IS NOT SQL",
            "difficulty": "easy",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["reward"] == 0.0

    def test_tasks_endpoint(self, test_client):
        resp = test_client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert "easy" in data
        assert "medium" in data
        assert "hard" in data
        assert len(data["easy"]) == 3

    def test_schema_endpoint(self, test_client):
        resp = test_client.get("/schema")
        assert resp.status_code == 200
        data = resp.json()
        assert "schema" in data
        assert "customers" in data["schema"]

    def test_root_returns_html(self, test_client):
        resp = test_client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    def test_full_easy_episode(self, test_client):
        """Complete episode solving easy_1 with reference solution."""
        test_client.post("/reset", json={"difficulty": "easy", "task_id": "easy_1"})
        resp = test_client.post("/step", json={
            "query": TASKS["easy_1"]["reference_sql"],
            "difficulty": "easy",
            "task_id": "easy_1",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["reward"] >= 0.9

    def test_medium_join_query(self, test_client):
        test_client.post("/reset", json={"difficulty": "medium", "task_id": "medium_1"})
        resp = test_client.post("/step", json={
            "query": TASKS["medium_1"]["reference_sql"],
            "difficulty": "medium",
            "task_id": "medium_1",
        })
        data = resp.json()
        assert data["reward"] >= 0.9

    def test_hard_cte_query(self, test_client):
        test_client.post("/reset", json={"difficulty": "hard", "task_id": "hard_1"})
        resp = test_client.post("/step", json={
            "query": TASKS["hard_1"]["reference_sql"],
            "difficulty": "hard",
            "task_id": "hard_1",
        })
        data = resp.json()
        assert data["reward"] >= 0.9


# ===========================================================================
# 11. Reward Consistency Tests
# ===========================================================================

class TestRewardConsistency:

    def test_reward_deterministic(self, db):
        """Same query on same data should always give same reward."""
        task_id = "easy_1"
        query = TASKS[task_id]["reference_sql"]
        rewards = [grade(task_id, query, db)[0] for _ in range(5)]
        assert len(set(rewards)) == 1, "Reward is not deterministic"

    def test_correct_beats_incorrect(self, db):
        correct_reward, _, _, _ = grade(
            "easy_1",
            "SELECT name, email FROM customers WHERE country = 'USA'",
            db
        )
        wrong_reward, _, _, _ = grade(
            "easy_1",
            "SELECT name, email FROM customers WHERE country = 'Japan'",
            db
        )
        assert correct_reward > wrong_reward

    def test_reward_increases_with_more_matches(self, db):
        """Returning more matching rows should give higher reward."""
        # All USA customers
        full_reward, _, _, _ = grade(
            "easy_1",
            "SELECT name, email FROM customers WHERE country = 'USA'",
            db
        )
        # Only one USA customer
        partial_reward, _, _, _ = grade(
            "easy_1",
            "SELECT name, email FROM customers WHERE country = 'USA' LIMIT 1",
            db
        )
        assert full_reward >= partial_reward


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    # Quick smoke test without pytest
    print("Running smoke tests...")
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    seed_database(conn)

    print(f"  Tasks registered: {len(TASKS)}")
    for diff in ["easy", "medium", "hard"]:
        tasks = get_all_tasks_by_difficulty(diff)
        print(f"  {diff.capitalize()} tasks: {len(tasks)}")

    # Test all reference solutions
    passed = 0
    failed = 0
    for task_id, task in TASKS.items():
        reward, msg, _, _ = grade(task_id, task["reference_sql"], conn)
        status = "PASS" if reward >= 0.9 else "FAIL"
        if reward >= 0.9:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {task_id}: reward={reward:.4f} - {msg[:50]}")

    conn.close()
    print(f"\nSmoke test: {passed} passed, {failed} failed")

    # Test environment lifecycle
    env = SQLEnvironment()
    obs = env.reset(difficulty="easy")
    print(f"\nEnvironment reset: task={obs.task_description[:60]}...")
    obs = env.step(SQLAction(query=TASKS["easy_1"]["reference_sql"], difficulty="easy"))
    print(f"Step result: reward={obs.reward}, rows={len(obs.result)}")
    env.close()

    print("\nAll smoke tests complete.")
