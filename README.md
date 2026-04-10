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

An OpenEnv-compatible reinforcement learning environment that simulates the day-to-day work of a **data analyst** fulfilling ad-hoc SQL requests from business stakeholders (marketing, finance, CRM, merchandising). Agents learn to translate natural-language business requirements into correct, efficient SQL queries — a high-value real-world task.

## Why This Environment Matters

**Text-to-SQL is one of the most commercially valuable analyst skills** — data teams spend significant time writing bespoke queries for stakeholder requests, and current LLMs still fail on complex queries involving CTEs, window functions, and multi-table joins. Existing benchmarks (Spider, BIRD) are static evaluation sets. There is no standard RL environment for *training* agents to improve iteratively on SQL tasks through trial-and-error feedback.

This environment fills that gap:

- **Real-world task**: agents play the role of a data analyst, fulfilling concrete business requests — not solving academic puzzles
- **Iterative learning**: reward signal at every step, not just at episode end
- **Partial credit with exploit protection**: Jaccard-based row matching gives meaningful gradient signal while penalising both missing rows *and* extra rows — returning the entire table does not score 1.0
- **Difficulty curriculum**: 3 tiers (easy → medium → hard) enable curriculum learning strategies
- **Business-realistic framing**: all 9 tasks are modelled on real analyst/engineer requests with named stakeholder teams
- **Zero external dependencies**: pure SQLite, runs on any 2 vCPU / 8 GB machine

## Environment Overview

The environment hosts an in-memory SQLite e-commerce database with 4 tables and deterministic seed data:

| Table | Rows | Key Columns |
|---|---|---|
| `customers` | 20 | id, name, email, city, country, created_at |
| `products` | 20 | id, name, category, price, stock |
| `orders` | 30 | id, customer_id, order_date, total_amount, status |
| `order_items` | 50 | id, order_id, product_id, quantity, unit_price |

At each step the agent submits a SQL query. The grader executes it, compares results to the reference solution, and returns a reward in `[0.0, 1.0]` with detailed feedback.

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
| `schema_info` | `str` | Full database schema and role context |
| `task_description` | `str` | Business-context task description |
| `expected_columns` | `list[str]` | Column names the answer must contain |
| `step_count` | `int` | Steps taken this episode |
| `score_breakdown` | `dict` | Sub-scores: correctness, keyword_bonus, efficiency |

## Tasks (9 total across 3 tiers)

All tasks simulate real stakeholder requests submitted to a data analyst:

### Easy — Basic filtering and aggregation

| ID | Stakeholder Request | SQL Concepts |
|---|---|---|
| `easy_1` | Marketing: US customer email list for a promotional campaign | `WHERE` country filter |
| `easy_2` | Finance: daily fulfilment count report | `COUNT` + `WHERE` status |
| `easy_3` | Merchandising: top-5 premium products for a catalogue | `ORDER BY` + `LIMIT` |

### Medium — Joins, grouping, and date logic

| ID | Stakeholder Request | SQL Concepts |
|---|---|---|
| `medium_1` | Loyalty: total spend per customer for VIP tier selection | `JOIN` + `GROUP BY` + `SUM` |
| `medium_2` | Inventory: dead-stock products that have never been ordered | `LEFT JOIN` + `NULL` check |
| `medium_3` | Finance: monthly average order value trend for 2023 | `STRFTIME` + `AVG` + `GROUP BY` |

### Hard — CTEs, window functions, correlated subqueries

| ID | Stakeholder Request | SQL Concepts |
|---|---|---|
| `hard_1` | CRM: customers above average lifetime value for premium tier | `CTE` + scalar subquery |
| `hard_2` | Category mgmt: best-selling SKU per product category (ties allowed) | `CTE` + `RANK()` window function |
| `hard_3` | Retention: customers active in all three years 2022, 2023, and 2024 | Correlated subquery + `COUNT DISTINCT` |

## Reward Function

```
reward = correctness × 0.7 + keyword_bonus × 0.1 + efficiency_bonus × 0.2
```

| Component | Range | Description |
|---|---|---|
| `correctness` | 0.0–1.0 | **Jaccard multiset similarity** — penalises both missing rows (low recall) and extra rows (low precision). An agent cannot score 1.0 by returning the entire table. |
| `keyword_bonus` | 0.0–0.1 | Query uses expected SQL constructs (JOIN, GROUP BY, etc.) |
| `efficiency_bonus` | 0.0–0.2 | Penalises `SELECT *`, `CROSS JOIN`, and deeply nested subqueries |

**Why this reward design is good for RL:**
- Non-binary: agents receive gradient signal even on partially correct queries
- Exploit-resistant: Jaccard scoring penalises both over-fetching and under-fetching
- Column aliases normalised: `total` instead of `total_spent` is not penalised if values match
- Efficiency signal discourages degenerate solutions
- Per-step rewards enable policy gradient methods without sparse returns

**Grader variety (not always the same score):**

| Input | easy_1 reward |
|---|---|
| Correct `WHERE country = 'USA'` | 1.00 |
| Return all 20 customers (exploit attempt) | 0.34 |
| Empty result set | 0.30 |
| SQL syntax error | 0.00 |

## Baseline Scores

Deterministic rule-based baseline (no LLM) — achieved in 1 step per task:

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

LLM agents (e.g. Qwen2.5-72B, Nemotron) are expected to score **0.7–1.0 on easy**, **0.5–0.9 on medium**, and **0.2–0.7 on hard** — leaving meaningful room for RL improvement.

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
| `HF_TOKEN` / `OPENAI_API_KEY` | *(required)* | Hugging Face / API key |

The inference script accepts `HF_TOKEN`, `OPENAI_API_KEY`, or `API_KEY` (checked in that order).

## Project Structure

```
.
├── inference.py           # Baseline inference script (mandatory, root)
├── models.py              # Pydantic Action / Observation / State models
├── client.py              # HTTP EnvClient
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Package metadata and dependencies
├── uv.lock                # uv lockfile for multi-mode deployment
├── requirements.txt       # pip dependencies
├── Dockerfile             # HF Spaces-compatible container
├── test_env.py            # 72-test suite
└── server/
    ├── app.py             # FastAPI server (reset / step / state / ws)
    ├── sql_environment.py # Core environment logic
    └── tasks.py           # 9 tasks, graders, SQLite seed data
```

## Judging Criteria Compliance

| Criterion | Status |
|---|---|
| Simulates real human task (data analyst SQL workflow) | ✓ |
| OpenEnv typed models (Action, Observation, State) | ✓ |
| `step()` / `reset()` / `state()` endpoints | ✓ |
| `openenv.yaml` with metadata | ✓ |
| 3+ tasks with programmatic graders | ✓ (9 tasks) |
| Easy → medium → hard difficulty range | ✓ |
| Deterministic graders with clear success/failure | ✓ |
| Partial-credit reward over full trajectory | ✓ |
| Exploit-resistant grader (Jaccard, no table-dump cheat) | ✓ |
| Baseline inference script with OpenAI client | ✓ |
| `HF_TOKEN` / `OPENAI_API_KEY` credential support | ✓ |
| `[START]` / `[STEP]` / `[END]` stdout log format | ✓ |
| Dockerfile builds (non-root, port 7860, HEALTHCHECK) | ✓ |
| `pyproject.toml` + `uv.lock` for multi-mode deployment | ✓ |
| Runs on 2 vCPU / 8 GB, inference < 20 min | ✓ |
