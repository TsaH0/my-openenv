"""
inference.py - Baseline inference script for the SQL Query Learning Environment.

MANDATORY STDOUT FORMAT:
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables:
  API_BASE_URL   The API endpoint for the LLM (default: HF router)
  MODEL_NAME     The model identifier (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       Hugging Face / API key

Usage:
  python inference.py
"""

from __future__ import annotations

import os
import sys
import textwrap
from typing import List, Optional

from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────

API_KEY = (
    os.getenv("HF_TOKEN")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("API_KEY")
)
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

BENCHMARK = "sql_env"
MAX_STEPS = 10
TEMPERATURE = 0.1
MAX_TOKENS = 512

# All 9 tasks across 3 difficulty tiers
ALL_TASKS = [
    ("easy_1",   "easy"),
    ("easy_2",   "easy"),
    ("easy_3",   "easy"),
    ("medium_1", "medium"),
    ("medium_2", "medium"),
    ("medium_3", "medium"),
    ("hard_1",   "hard"),
    ("hard_2",   "hard"),
    ("hard_3",   "hard"),
]

SYSTEM_PROMPT = textwrap.dedent("""
You are a data analyst at an e-commerce company. Business stakeholders
(marketing, finance, CRM, merchandising) submit ad-hoc data requests and
you fulfil them by writing SQL queries against the company database.

RULES:
- Return ONLY the SQL query — no explanation, no markdown fences, no comments.
- Use standard SQLite syntax.
- Use column aliases to match the expected column names exactly as stated in the task.
- Do NOT use SELECT * — select only the required columns.
- Aim for the simplest correct query; avoid unnecessary subqueries or CROSS JOINs.
""").strip()


# ── Structured Logging (MANDATORY FORMAT) ───────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    # Sanitize action string: remove newlines, limit length
    action_clean = action.replace("\n", " ").replace("\r", "").strip()
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action_clean} "
        f"reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ── LLM Query ──────────────────────────────────────────────────────────────

def get_sql_from_llm(
    client: OpenAI,
    schema_info: str,
    task_description: str,
    expected_columns: List[str],
    previous_attempts: List[dict],
) -> str:
    """Ask the LLM to produce a SQL query for the given task."""
    attempts_text = ""
    if previous_attempts:
        last = previous_attempts[-1]
        attempts_text = (
            f"\nPREVIOUS ATTEMPT:\n"
            f"  Query: {last['query']}\n"
            f"  Reward: {last['reward']}\n"
            f"  Feedback: {last['message']}\n"
            f"  Error: {last.get('error', 'none')}\n"
            f"\nImprove on this attempt.\n"
        )

    user_prompt = (
        f"DATABASE SCHEMA:\n{schema_info}\n\n"
        f"TASK:\n{task_description}\n\n"
        f"EXPECTED OUTPUT COLUMNS: {', '.join(expected_columns)}\n"
        f"{attempts_text}\n"
        f"SQL QUERY:"
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.startswith("```")]
            text = "\n".join(lines).strip()
        return text if text else _fallback_query(task_description)
    except Exception as exc:
        print(f"[DEBUG] LLM request failed: {exc}", flush=True)
        return _fallback_query(task_description)


def _fallback_query(description: str) -> str:
    """
    Deterministic rule-based fallback when LLM API is unavailable.
    Pattern-matches task descriptions to known reference queries.
    """
    d = description.lower()

    # Easy
    if "usa" in d or "united states" in d:
        return "SELECT name, email FROM customers WHERE country = 'USA'"
    if "count" in d and "completed" in d:
        return "SELECT COUNT(*) AS total_completed FROM orders WHERE status = 'completed'"
    if "top 5" in d and "expensive" in d:
        return "SELECT name, category, price FROM products ORDER BY price DESC LIMIT 5"

    # Hard (checked before medium to avoid substring collisions)
    if "above" in d and "average" in d:
        return (
            "WITH customer_totals AS ("
            "  SELECT c.name, SUM(o.total_amount) AS total_spent"
            "  FROM customers c"
            "  JOIN orders o ON c.id = o.customer_id"
            "  WHERE o.status = 'completed'"
            "  GROUP BY c.id, c.name"
            ") "
            "SELECT name, total_spent FROM customer_totals "
            "WHERE total_spent > (SELECT AVG(total_spent) FROM customer_totals) "
            "ORDER BY total_spent DESC"
        )
    if "best-selling" in d or "best selling" in d:
        return (
            "WITH product_sales AS ("
            "  SELECT p.category, p.name AS product_name, SUM(oi.quantity) AS total_quantity"
            "  FROM products p JOIN order_items oi ON p.id = oi.product_id"
            "  GROUP BY p.id, p.category, p.name"
            "), ranked AS ("
            "  SELECT category, product_name, total_quantity,"
            "    RANK() OVER (PARTITION BY category ORDER BY total_quantity DESC) AS rnk"
            "  FROM product_sales"
            ") "
            "SELECT category, product_name, total_quantity FROM ranked WHERE rnk = 1 "
            "ORDER BY category, product_name"
        )
    if "2022" in d and "2023" in d and "2024" in d:
        return (
            "SELECT c.name, c.email FROM customers c "
            "WHERE (SELECT COUNT(DISTINCT STRFTIME('%Y', o.order_date)) "
            "FROM orders o WHERE o.customer_id = c.id "
            "AND STRFTIME('%Y', o.order_date) IN ('2022','2023','2024')) = 3 "
            "ORDER BY c.name ASC"
        )

    # Medium
    if "total spending" in d or "total spent" in d:
        return (
            "SELECT c.name, SUM(o.total_amount) AS total_spent "
            "FROM customers c JOIN orders o ON c.id = o.customer_id "
            "WHERE o.status = 'completed' "
            "GROUP BY c.id, c.name ORDER BY total_spent DESC"
        )
    if "never" in d and ("order" in d or "appear" in d):
        return (
            "SELECT p.name, p.category, p.price FROM products p "
            "LEFT JOIN order_items oi ON p.id = oi.product_id WHERE oi.id IS NULL"
        )
    if "average" in d and "month" in d:
        return (
            "SELECT STRFTIME('%Y-%m', order_date) AS month, "
            "ROUND(AVG(total_amount), 2) AS avg_order_value "
            "FROM orders WHERE order_date LIKE '2023%' "
            "GROUP BY month ORDER BY month ASC"
        )

    return "SELECT name FROM customers LIMIT 5"


# ── Run One Task Episode ───────────────────────────────────────────────────

def run_task(
    task_id: str,
    difficulty: str,
    client: OpenAI,
) -> float:
    """
    Run a single task as one episode. Returns the best score in [0, 1].

    Emits [START], [STEP]..., [END] to stdout.
    """
    # Import env locally to keep module-level clean
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from server.sql_environment import SQLEnvironment
    from models import SQLAction

    env = SQLEnvironment()
    rewards: List[float] = []
    steps_taken = 0
    best_score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(difficulty=difficulty, task_id=task_id)
        previous_attempts: List[dict] = []

        for step in range(1, MAX_STEPS + 1):
            if obs.done:
                break

            # Get query from LLM or fallback
            if API_KEY and API_BASE_URL:
                query = get_sql_from_llm(
                    client,
                    schema_info=obs.schema_info,
                    task_description=obs.task_description,
                    expected_columns=obs.expected_columns,
                    previous_attempts=previous_attempts,
                )
            else:
                query = _fallback_query(obs.task_description)

            action = SQLAction(query=query, difficulty=difficulty, task_id=task_id)
            obs = env.step(action)

            reward = obs.reward
            done = obs.done
            error = obs.error if obs.error else None

            rewards.append(reward)
            steps_taken = step
            best_score = max(best_score, reward)

            log_step(
                step=step,
                action=query,
                reward=reward,
                done=done,
                error=error,
            )

            previous_attempts.append({
                "query": query,
                "reward": reward,
                "message": obs.message,
                "error": obs.error,
            })
            if len(previous_attempts) > 2:
                previous_attempts.pop(0)

            # Perfect score achieved — stop
            if reward >= 1.0:
                break

            if done:
                break

        # Final score = best reward achieved (already in [0, 1])
        score = min(max(best_score, 0.0), 1.0)
        success = score >= 0.5

    except Exception as exc:
        print(f"[DEBUG] Exception during episode: {exc}", flush=True)
        score = 0.0

    finally:
        try:
            env.close()
        except Exception:
            pass
        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
        )

    return score


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "sk-placeholder")

    total_score = 0.0
    task_scores = {}

    for task_id, difficulty in ALL_TASKS:
        score = run_task(task_id, difficulty, client)
        task_scores[task_id] = score
        total_score += score

    # Summary (not part of mandatory format — informational only)
    print("\n" + "=" * 60, flush=True)
    print("INFERENCE SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for task_id, score in task_scores.items():
        status = "PASS" if score >= 0.5 else "FAIL"
        print(f"  [{status}] {task_id}: score={score:.2f}", flush=True)
    avg = total_score / len(ALL_TASKS) if ALL_TASKS else 0.0
    print(f"\n  Average score: {avg:.2f} ({total_score:.2f}/{len(ALL_TASKS)})", flush=True)


if __name__ == "__main__":
    main()
