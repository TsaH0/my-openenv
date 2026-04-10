---
title: SQL Query Learning Environment
emoji: 🗄️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - sql
  - agent
license: mit
---

# SQL Query Learning Environment

An OpenEnv-compatible reinforcement learning environment where AI agents learn to write correct and efficient SQL queries against a realistic e-commerce database.

## Motivation

SQL querying is a fundamental real-world skill used by millions of analysts, engineers, and data scientists daily. This environment provides a structured, graded setting for training and evaluating LLM-based agents on SQL generation tasks of increasing complexity.

## Environment Overview

The environment hosts an in-memory SQLite database seeded with a realistic e-commerce schema:
- **customers** (20 rows) -- name, email, city, country
- **products** (20 rows) -- name, category, price, stock
- **orders** (30 rows) -- customer_id, date, total_amount, status
- **order_items** (50 rows) -- order_id, product_id, quantity, price

The agent receives the schema description and a task, then submits SQL queries. The environment grades each query against the reference solution and returns a reward in [0.0, 1.0].

## Action Space

| Field | Type | Description |
|---|---|---|
| `query` | `str` | The SQL query string to execute |
| `task_id` | `str \| None` | Optional task ID (defaults to current) |
| `difficulty` | `str` | `"easy"`, `"medium"`, or `"hard"` |

## Observation Space

| Field | Type | Description |
|---|---|---|
| `result` | `list[dict]` | Query result rows |
| `error` | `str` | Error message (empty if none) |
| `reward` | `float` | Step reward in [0.0, 1.0] |
| `done` | `bool` | Whether the episode is finished |
| `message` | `str` | Human-readable grader feedback |
| `schema_info` | `str` | Database schema description |
| `task_description` | `str` | Natural-language task objective |
| `expected_columns` | `list[str]` | Expected output column names |
| `step_count` | `int` | Steps taken so far |
| `score_breakdown` | `dict` | Sub-scores (correctness, efficiency) |

## Tasks (9 total, 3 difficulty tiers)

### Easy (SELECT, WHERE, ORDER BY)

| ID | Task | SQL Concepts |
|---|---|---|
| `easy_1` | Retrieve customers from the USA | WHERE filter |
| `easy_2` | Count completed orders | COUNT + WHERE |
| `easy_3` | Top 5 most expensive products | ORDER BY + LIMIT |

### Medium (JOIN, GROUP BY, aggregation)

| ID | Task | SQL Concepts |
|---|---|---|
| `medium_1` | Total spending per customer | JOIN + GROUP BY + SUM |
| `medium_2` | Products never ordered | LEFT JOIN + NULL check |
| `medium_3` | Average order value per month | STRFTIME + AVG + GROUP BY |

### Hard (subqueries, CTEs, window functions)

| ID | Task | SQL Concepts |
|---|---|---|
| `hard_1` | Customers with above-average spending | CTE + subquery |
| `hard_2` | Best-selling product per category | CTE + RANK() window function |
| `hard_3` | Customers active in 2022, 2023, and 2024 | Correlated subquery |

## Reward Function

```
reward = correctness * 0.7 + keyword_bonus * 0.1 + efficiency_bonus * 0.2
```

- **Correctness (0.0-1.0)**: Ratio of matched rows between agent and reference output. Partial credit is awarded for partially correct results.
- **Keyword bonus (0.1)**: Awarded if the query uses expected SQL constructs (e.g., JOIN, GROUP BY).
- **Efficiency bonus (0.0-0.2)**: Penalizes unnecessary SELECT *, excessive subqueries, CROSS JOINs.

Reward is capped at 1.0.

## Baseline Scores

Using the deterministic fallback heuristic (no LLM):

| Difficulty | Score |
|---|---|
| Easy (3 tasks) | 1.00 / 1.00 / 1.00 |
| Medium (3 tasks) | 1.00 / 1.00 / 1.00 |
| Hard (3 tasks) | 1.00 / 1.00 / 1.00 |

## Setup

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)

### Local Development

```bash
pip install -r requirements.txt

# Run the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# In a separate terminal, run inference
python inference.py
```

### Docker

```bash
docker build -t sql_env .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
  -e HF_TOKEN="your-token" \
  sql_env
```

### Tests

```bash
python -m pytest test_env.py -v
```

72 tests covering: database seeding, task registry, grader accuracy, partial credit, environment lifecycle, all 9 tasks solvable, multi-difficulty sweeps, inference script, FastAPI endpoints, reward consistency.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check (HTTP 200) |
| `POST` | `/reset` | Start new episode |
| `POST` | `/step` | Execute SQL action |
| `GET` | `/state` | Get episode state |
| `GET` | `/tasks` | List all tasks by difficulty |
| `GET` | `/schema` | Get database schema |
| `GET` | `/` | Interactive web UI |
| `WS` | `/ws` | WebSocket endpoint |

## Environment Variables

| Variable | Description |
|---|---|
| `API_BASE_URL` | LLM API endpoint |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | Hugging Face / API key |

## Project Structure

```
.
├── inference.py          # Baseline inference script (root, mandatory)
├── models.py             # Pydantic Action/Observation/State models
├── client.py             # HTTP EnvClient
├── openenv.yaml          # OpenEnv manifest
├── requirements.txt      # Python dependencies
├── Dockerfile            # HF Spaces compatible container
├── test_env.py           # 72-test suite
├── README.md             # This file
└── server/
    ├── app.py            # FastAPI server
    ├── sql_environment.py # Core Environment (reset/step/state)
    └── tasks.py          # 9 tasks, graders, SQLite seed data
```
