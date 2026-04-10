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
  - text-to-sql
license: mit
---

# SQL Query Learning Environment

An OpenEnv-compatible reinforcement learning environment for training and evaluating AI agents on real-world SQL query generation tasks of increasing complexity.

## Why This Environment Matters

**Text-to-SQL is one of the most commercially valuable agent capabilities** — yet current LLMs still fail on complex queries involving CTEs, window functions, and multi-table joins. Existing benchmarks (Spider, BIRD) are static evaluation sets. There is no standard RL environment for *training* agents to improve iteratively on SQL tasks through trial-and-error feedback.

This environment fills that gap:

- **Iterative learning**: agents receive reward signals at every step, not just at episode end
- **Partial credit**: a query returning 3 of 5 correct rows scores higher than one returning 0 — meaningful gradient signal for RL training
- **Difficulty curriculum**: 3 tiers (easy → medium → hard) enable curriculum learning strategies
- **Business-realistic tasks**: queries are framed as actual analyst/engineer requests, not academic puzzles
- **Zero external dependencies**: pure SQLite, runs on any 2 vCPU / 8 GB machine

## Environment Overview

The environment hosts an in-memory SQLite e-commerce database with 4 tables and deterministic seed data:

| Table | Rows | Key Columns |
|---|---|---|
| `customers` | 20 | id, name, email, city, country, created_at |
| `products` | 20 | id, name, category, price, stock |
| `orders` | 30 | id, customer_id, order_date, total_amount, status |
| `order_items` | 50 | id, order_id, product_id, quantity, unit_price |

At each step, the agent submits a SQL query. The grader executes it against the database, compares results to the reference solution, and returns a reward in `[0.0, 1.0]` with detailed feedback.

## Action Space

| Field | Type | Description |
|---|---|---|
| `query` | `str` | The SQL query string to execute |
| `task_id` | `str \| None` | Target a specific task (defaults to current) |
| `difficulty` | `str` | `"easy"`, `"medium"`, or `"hard"` |

## Observation Space

| Field | Type | Description |
|---|---|---|
| `result` | `list[dict]` | Rows returned by the agent's query |
| `error` | `str` | SQL error message (empty if none) |
| `reward` | `float` | Step reward in `[0.0, 1.0]` |
| `done` | `bool` | Whether the episode has ended |
| `message` | `str` | Grader feedback (e.g. "3/5 rows matched") |
| `schema_info` | `str` | Full database schema for the agent |
| `task_description` | `str` | Business-context task description |
| `expected_columns` | `list[str]` | Column names the answer must contain |
| `step_count` | `int` | Steps taken this episode |
| `score_breakdown` | `dict` | Sub-scores: correctness, keyword_bonus, efficiency |

## Tasks (9 total across 3 tiers)

### Easy — Basic filtering and aggregation

| ID | Business Scenario | SQL Concepts |
|---|---|---|
| `easy_1` | Marketing needs a US customer email list | `WHERE` country filter |
| `easy_2` | Finance wants a fulfillment count | `COUNT` + `WHERE` status |
| `easy_3` | Merchandising wants premium products for a catalogue | `ORDER BY` + `LIMIT` |

### Medium — Joins, grouping, and date logic

| ID | Business Scenario | SQL Concepts |
|---|---|---|
| `medium_1` | Identify top spenders for a loyalty program | `JOIN` + `GROUP BY` + `SUM` |
| `medium_2` | Find dead-stock products for clearance | `LEFT JOIN` + `NULL` check |
| `medium_3` | Build a monthly revenue trend report for 2023 | `STRFTIME` + `AVG` + `GROUP BY` |

### Hard — CTEs, window functions, correlated subqueries

| ID | Business Scenario | SQL Concepts |
|---|---|---|
| `hard_1` | VIP segment: customers above average lifetime value | `CTE` + scalar subquery |
| `hard_2` | Category hero: best-selling SKU per product category | `CTE` + `RANK()` window function |
| `hard_3` | Retained customers active across 2022, 2023, and 2024 | Multi-year correlated subquery |

## Reward Function

```
reward = correctness × 0.7 + keyword_bonus × 0.1 + efficiency_bonus × 0.2
```

| Component | Range | Description |
|---|---|---|
| `correctness` | 0.0–1.0 | Fraction of expected rows matched (supports partial credit and column-alias normalization) |
| `keyword_bonus` | 0.0–0.1 | Query uses expected SQL constructs (JOIN, GROUP BY, etc.) |
| `efficiency_bonus` | 0.0–0.2 | Penalises `SELECT *`, `CROSS JOIN`, and deeply nested subqueries |

**Why this reward design is good for RL:**
- Non-binary: agents receive signal even on partially correct queries
- Column aliases are normalised before comparison — an agent using `total` instead of `total_spent` is not punished for a cosmetic difference
- Efficiency signal discourages degenerate solutions (e.g. scanning all rows)
- Per-step rewards enable policy gradient methods without sparse returns

## Baseline Scores

Deterministic rule-based baseline (no LLM):

| Task | Score | Steps |
|---|---|---|
| easy_1 | 1.00 | 1 |
| easy_2 | 1.00 | 1 |
| easy_3 | 1.00 | 1 |
| medium_1 | 1.00 | 1 |
| medium_2 | 1.00 | 1 |
| medium_3 | 1.00 | 1 |
| hard_1 | 1.00 | 1 |
| hard_2 | 1.00 | 1 |
| hard_3 | 1.00 | 1 |

LLM agents (e.g. Qwen2.5-72B) are expected to score 0.7–1.0 on easy, 0.5–0.9 on medium, and 0.2–0.7 on hard tasks — leaving meaningful room for RL improvement.

## Setup

### Local Development

```bash
pip install -r requirements.txt

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Run inference (separate terminal)
python inference.py
```

### Docker

```bash
docker build -t sql_env .
docker run -p 7860:7860 \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
  -e HF_TOKEN="your-hf-token" \
  sql_env
```

### Run Tests

```bash
python -m pytest test_env.py -v
# 72 tests: database, graders, partial credit, lifecycle, all 9 tasks, FastAPI endpoints
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check — returns HTTP 200 |
| `POST` | `/reset` | Start new episode |
| `POST` | `/step` | Execute SQL action |
| `GET` | `/state` | Current episode state |
| `GET` | `/tasks` | List all tasks by difficulty |
| `GET` | `/schema` | Database schema |
| `GET` | `/` | Interactive web UI |
| `WS` | `/ws` | WebSocket endpoint |

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | *(required)* | Hugging Face / API key |

## Project Structure

```
.
├── inference.py           # Baseline inference script (mandatory, root)
├── models.py              # Pydantic Action / Observation / State models
├── client.py              # HTTP EnvClient
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Package metadata and dependencies
├── requirements.txt       # pip dependencies
├── Dockerfile             # HF Spaces-compatible container
├── test_env.py            # 72-test suite
└── server/
    ├── app.py             # FastAPI server (reset / step / state / ws)
    ├── sql_environment.py # Core environment logic
    └── tasks.py           # 9 tasks, graders, SQLite seed data
```
