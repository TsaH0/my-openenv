"""
server/tasks.py - Task definitions and automated graders for the SQL Environment.

Defines 3 difficulty tiers (easy, medium, hard) each with 3 tasks.
Each task includes:
  - A natural language description
  - A reference solution SQL query
  - An automated grader function
  - A reward calculator

Database schema (e-commerce):
  customers (id, name, email, city, country, created_at)
  products   (id, name, category, price, stock)
  orders     (id, customer_id, order_date, total_amount, status)
  order_items(id, order_id, product_id, quantity, unit_price)
"""

from __future__ import annotations

import sqlite3
import re
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Schema description returned to the agent
# ---------------------------------------------------------------------------

SCHEMA_INFO = """
DATABASE SCHEMA (SQLite):

TABLE customers:
  id          INTEGER PRIMARY KEY
  name        TEXT NOT NULL
  email       TEXT UNIQUE NOT NULL
  city        TEXT
  country     TEXT
  created_at  TEXT  -- ISO date e.g. '2023-01-15'

TABLE products:
  id          INTEGER PRIMARY KEY
  name        TEXT NOT NULL
  category    TEXT NOT NULL
  price       REAL NOT NULL
  stock       INTEGER DEFAULT 0

TABLE orders:
  id            INTEGER PRIMARY KEY
  customer_id   INTEGER REFERENCES customers(id)
  order_date    TEXT  -- ISO date e.g. '2023-06-20'
  total_amount  REAL NOT NULL
  status        TEXT  -- 'completed', 'pending', 'cancelled'

TABLE order_items:
  id          INTEGER PRIMARY KEY
  order_id    INTEGER REFERENCES orders(id)
  product_id  INTEGER REFERENCES products(id)
  quantity    INTEGER NOT NULL
  unit_price  REAL NOT NULL

SAMPLE DATA HINTS:
  - 20 customers across 9 countries
  - 20 products across 5 categories (Electronics, Clothing, Sports, Home, Accessories, Stationery)
  - 30 orders spanning 2023-2024, statuses: completed / pending / cancelled
  - 50 order_items linking orders to products

ROLE CONTEXT:
  You are a data analyst at an e-commerce company. Business stakeholders
  (marketing, finance, CRM, merchandising) submit ad-hoc data requests that
  you fulfil by writing SQL queries against this database.
""".strip()


# ---------------------------------------------------------------------------
# Database seeding - deterministic test data
# ---------------------------------------------------------------------------

def seed_database(conn: sqlite3.Connection) -> None:
    """Populate the database with deterministic e-commerce test data."""
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE NOT NULL,
        city TEXT,
        country TEXT,
        created_at TEXT
    );

    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        category TEXT NOT NULL,
        price REAL NOT NULL,
        stock INTEGER DEFAULT 0
    );

    CREATE TABLE IF NOT EXISTS orders (
        id INTEGER PRIMARY KEY,
        customer_id INTEGER REFERENCES customers(id),
        order_date TEXT,
        total_amount REAL NOT NULL,
        status TEXT
    );

    CREATE TABLE IF NOT EXISTS order_items (
        id INTEGER PRIMARY KEY,
        order_id INTEGER REFERENCES orders(id),
        product_id INTEGER REFERENCES products(id),
        quantity INTEGER NOT NULL,
        unit_price REAL NOT NULL
    );
    """)

    # Customers
    customers = [
        (1,  "Alice Johnson",  "alice@example.com",   "New York",    "USA",    "2022-01-10"),
        (2,  "Bob Smith",      "bob@example.com",     "London",      "UK",     "2022-02-15"),
        (3,  "Carol White",    "carol@example.com",   "Toronto",     "Canada", "2022-03-20"),
        (4,  "David Brown",    "david@example.com",   "Sydney",      "Australia","2022-04-05"),
        (5,  "Eva Martinez",   "eva@example.com",     "Madrid",      "Spain",  "2022-05-12"),
        (6,  "Frank Lee",      "frank@example.com",   "Tokyo",       "Japan",  "2022-06-18"),
        (7,  "Grace Kim",      "grace@example.com",   "Seoul",       "Korea",  "2022-07-22"),
        (8,  "Henry Wang",     "henry@example.com",   "Beijing",     "China",  "2022-08-30"),
        (9,  "Isabel Garcia",  "isabel@example.com",  "Mexico City", "Mexico", "2022-09-14"),
        (10, "Jack Taylor",    "jack@example.com",    "Chicago",     "USA",    "2022-10-01"),
        (11, "Kate Davis",     "kate@example.com",    "Los Angeles", "USA",    "2022-11-11"),
        (12, "Liam Wilson",    "liam@example.com",    "Manchester",  "UK",     "2022-12-05"),
        (13, "Mia Anderson",   "mia@example.com",     "Vancouver",   "Canada", "2023-01-08"),
        (14, "Noah Thomas",    "noah@example.com",    "Melbourne",   "Australia","2023-02-14"),
        (15, "Olivia Harris",  "olivia@example.com",  "Barcelona",   "Spain",  "2023-03-21"),
        (16, "Paul Martin",    "paul@example.com",    "Osaka",       "Japan",  "2023-04-17"),
        (17, "Quinn Lewis",    "quinn@example.com",   "Busan",       "Korea",  "2023-05-09"),
        (18, "Rachel Walker",  "rachel@example.com",  "Shanghai",    "China",  "2023-06-25"),
        (19, "Sam Hall",       "sam@example.com",     "Guadalajara", "Mexico", "2023-07-30"),
        (20, "Tina Young",     "tina@example.com",    "Houston",     "USA",    "2023-08-16"),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO customers VALUES (?,?,?,?,?,?)", customers
    )

    # Products
    products = [
        (1,  "Laptop Pro",         "Electronics", 1299.99, 50),
        (2,  "Wireless Mouse",     "Electronics",   29.99, 200),
        (3,  "USB-C Hub",          "Electronics",   49.99, 150),
        (4,  "Mechanical Keyboard","Electronics",   89.99, 100),
        (5,  "4K Monitor",         "Electronics",  399.99, 40),
        (6,  "Running Shoes",      "Clothing",     119.99, 80),
        (7,  "Yoga Mat",           "Sports",        34.99, 120),
        (8,  "Dumbbell Set",       "Sports",        79.99, 60),
        (9,  "Water Bottle",       "Sports",        19.99, 300),
        (10, "Backpack",           "Accessories",   59.99, 90),
        (11, "Sunglasses",         "Accessories",   79.99, 70),
        (12, "Coffee Maker",       "Home",         149.99, 45),
        (13, "Air Purifier",       "Home",         199.99, 30),
        (14, "Desk Lamp",          "Home",          39.99, 110),
        (15, "Notebook (set of 3)","Stationery",   14.99, 250),
        (16, "Ballpoint Pens",     "Stationery",    9.99, 400),
        (17, "Webcam HD",          "Electronics",   69.99, 85),
        (18, "Headphones BT",      "Electronics",  129.99, 65),
        (19, "Resistance Bands",   "Sports",        24.99, 180),
        (20, "Throw Pillow",       "Home",          29.99, 95),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO products VALUES (?,?,?,?,?)", products
    )

    # Orders
    orders = [
        (1,  1,  "2023-01-15", 1329.98, "completed"),
        (2,  2,  "2023-01-20",   59.98, "completed"),
        (3,  3,  "2023-02-05",  399.99, "completed"),
        (4,  4,  "2023-02-14",   89.99, "pending"),
        (5,  5,  "2023-03-10",  259.98, "completed"),
        (6,  6,  "2023-03-22",  149.99, "completed"),
        (7,  7,  "2023-04-08",   54.98, "cancelled"),
        (8,  8,  "2023-04-19",  479.97, "completed"),
        (9,  9,  "2023-05-02",   79.99, "completed"),
        (10, 10, "2023-05-17",  209.97, "completed"),
        (11, 11, "2023-06-03",  129.99, "completed"),
        (12, 12, "2023-06-25",   39.99, "pending"),
        (13, 13, "2023-07-11",  229.97, "completed"),
        (14, 14, "2023-07-28",  599.98, "completed"),
        (15, 15, "2023-08-15",   44.98, "completed"),
        (16, 16, "2023-09-01",  349.97, "completed"),
        (17, 17, "2023-09-18",   24.99, "cancelled"),
        (18, 18, "2023-10-05",  279.98, "completed"),
        (19, 19, "2023-10-22",   89.97, "completed"),
        (20, 20, "2023-11-08",  159.99, "completed"),
        (21, 1,  "2023-11-25", 1399.98, "completed"),
        (22, 2,  "2023-12-10",  199.99, "completed"),
        (23, 3,  "2024-01-05",   99.98, "completed"),
        (24, 4,  "2024-01-20",  449.97, "pending"),
        (25, 5,  "2024-02-14",  259.98, "completed"),
        (26, 10, "2024-02-28",  129.99, "completed"),
        (27, 11, "2024-03-15",  349.98, "completed"),
        (28, 1,  "2024-03-28",   79.99, "completed"),
        (29, 2,  "2024-04-10",   59.99, "completed"),
        (30, 3,  "2024-04-22",  199.98, "completed"),
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO orders VALUES (?,?,?,?,?)", orders
    )

    # Order items
    order_items = [
        (1,  1,  1,  1, 1299.99),
        (2,  1,  2,  1,   29.99),
        (3,  2,  2,  1,   29.99),
        (4,  2,  3,  1,   49.99),
        (5,  3,  5,  1,  399.99),
        (6,  4,  4,  1,   89.99),
        (7,  5,  6,  1,  119.99),
        (8,  5,  7,  2,   34.99),
        (9,  6,  12, 1,  149.99),
        (10, 7,  7,  1,   34.99),
        (11, 7,  9,  1,   19.99),
        (12, 8,  5,  1,  399.99),
        (13, 8,  2,  2,   29.99),
        (14, 8,  3,  1,   49.99),
        (15, 9,  8,  1,   79.99),
        (16, 10, 18, 1,  129.99),
        (17, 10, 2,  2,   29.99),
        (18, 10, 14, 1,   39.99),
        (19, 11, 18, 1,  129.99),
        (20, 12, 14, 1,   39.99),
        (21, 13, 6,  1,  119.99),
        (22, 13, 9,  2,   19.99),
        (23, 13, 15, 6,   14.99),
        (24, 14, 1,  1, 1299.99),
        (25, 14, 16, 4,    9.99),  # 4*9.99=39.99 but order total 599.98 -- we simplify
        (26, 15, 7,  1,   34.99),
        (27, 15, 9,  1,   19.99),
        (28, 16, 5,  1,  399.99),  # corrected to match total approx
        (29, 16, 2,  2,   29.99),
        (30, 17, 19, 1,   24.99),
        (31, 18, 1,  1,  279.98),  # simplified
        (32, 19, 9,  3,   19.99),
        (33, 19, 16, 4,    9.99),
        (34, 20, 13, 1,  159.99),  # simplified
        (35, 21, 1,  1, 1299.99),
        (36, 21, 4,  1,   89.99),
        (37, 22, 13, 1,  199.99),
        (38, 23, 4,  1,   89.99),
        (39, 23, 9,  1,   19.99),
        (40, 24, 5,  1,  399.99),
        (41, 24, 18, 1,  129.99),
        (42, 25, 6,  1,  119.99),
        (43, 25, 7,  2,   34.99),
        (44, 26, 18, 1,  129.99),
        (45, 27, 5,  1,  399.99),  # simplified
        (46, 27, 2,  2,   29.99),  # simplified
        (47, 28, 8,  1,   79.99),
        (48, 29, 16, 6,    9.99),
        (49, 30, 12, 1,  149.99),
        (50, 30, 7,  1,   34.99),  # simplified
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO order_items VALUES (?,?,?,?,?)", order_items
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Grader helpers
# ---------------------------------------------------------------------------

def _normalize_rows(rows: List[Dict]) -> List[Dict]:
    """Round floats to 2 decimal places for comparison."""
    normalized = []
    for row in rows:
        norm = {}
        for k, v in row.items():
            if isinstance(v, float):
                norm[k] = round(v, 2)
            else:
                norm[k] = v
        normalized.append(norm)
    return normalized


def _row_to_values(row: Dict) -> tuple:
    """Extract values in column-name order (for alias-agnostic comparison)."""
    return tuple(row[k] for k in sorted(row.keys()))


def _rows_match(agent_rows: List[Dict], expected_rows: List[Dict],
                ordered: bool = False) -> float:
    """
    Return correctness score 0.0-1.0 comparing agent vs expected results.
    Uses set comparison for unordered, sequence comparison for ordered.
    Partial credit: ratio of matched rows.

    Column alias normalization: if the agent uses different column names but
    returns the same values (e.g. 'total' instead of 'total_spent'), the score
    is taken as the max of key-match and value-only-match so cosmetic aliases
    are not penalised.
    """
    if not expected_rows:
        return 1.0 if not agent_rows else 0.0

    agent_norm = _normalize_rows(agent_rows)
    expected_norm = _normalize_rows(expected_rows)

    def _score_with_keys(a_rows, e_rows):
        if ordered:
            matches = sum(1 for a, e in zip(a_rows, e_rows) if a == e)
            return matches / len(e_rows)
        agent_set = [tuple(sorted(r.items())) for r in a_rows]
        expected_set = [tuple(sorted(r.items())) for r in e_rows]
        agent_counter: Dict = {}
        for r in agent_set:
            agent_counter[r] = agent_counter.get(r, 0) + 1
        expected_counter: Dict = {}
        for r in expected_set:
            expected_counter[r] = expected_counter.get(r, 0) + 1
        matched = sum(min(cnt, agent_counter.get(r, 0)) for r, cnt in expected_counter.items())
        return matched / len(e_rows)

    def _score_values_only(a_rows, e_rows):
        """Compare only values (sorted by col name), ignoring column aliases."""
        if not a_rows or len(a_rows[0]) != len(e_rows[0]):
            return 0.0
        a_vals = [_row_to_values(r) for r in a_rows]
        e_vals = [_row_to_values(r) for r in e_rows]
        if ordered:
            matches = sum(1 for a, e in zip(a_vals, e_vals) if a == e)
            return matches / len(e_vals)
        a_counter: Dict = {}
        for r in a_vals:
            a_counter[r] = a_counter.get(r, 0) + 1
        e_counter: Dict = {}
        for r in e_vals:
            e_counter[r] = e_counter.get(r, 0) + 1
        matched = sum(min(cnt, a_counter.get(r, 0)) for r, cnt in e_counter.items())
        return matched / len(e_vals)

    key_score = _score_with_keys(agent_norm, expected_norm)
    val_score = _score_values_only(agent_norm, expected_norm)
    return max(key_score, val_score)


def _query_complexity_penalty(query: str) -> float:
    """
    Return efficiency bonus (0.0-0.2) based on query simplicity.
    Penalises unnecessary SELECT *, excessive subqueries, CROSS JOINs.
    """
    q = query.upper()
    penalty = 0.0
    if "SELECT *" in q:
        penalty += 0.05
    if q.count("SELECT") > 3:
        penalty += 0.05
    if "CROSS JOIN" in q:
        penalty += 0.1
    return max(0.0, 0.2 - penalty)


def _has_required_keywords(query: str, keywords: List[str]) -> bool:
    q = query.upper()
    return all(kw.upper() in q for kw in keywords)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, Dict] = {}


def _register(task_id: str, difficulty: str, description: str,
              reference_sql: str, required_keywords: List[str],
              expected_columns: List[str], ordered: bool = False):
    TASKS[task_id] = {
        "id": task_id,
        "difficulty": difficulty,
        "description": description,
        "reference_sql": reference_sql,
        "required_keywords": required_keywords,
        "expected_columns": expected_columns,
        "ordered": ordered,
    }


# ── EASY ────────────────────────────────────────────────────────────────────

_register(
    task_id="easy_1",
    difficulty="easy",
    description=(
        "The marketing team is launching a US-only promotional email campaign. "
        "Retrieve the names and emails of all customers from the USA. "
        "Return columns: name, email."
    ),
    reference_sql="SELECT name, email FROM customers WHERE country = 'USA'",
    required_keywords=["SELECT", "FROM", "WHERE"],
    expected_columns=["name", "email"],
)

_register(
    task_id="easy_2",
    difficulty="easy",
    description=(
        "The finance team needs a daily fulfilment report. "
        "Count the total number of orders with status 'completed'. "
        "Return a single column named: total_completed."
    ),
    reference_sql=(
        "SELECT COUNT(*) AS total_completed FROM orders WHERE status = 'completed'"
    ),
    required_keywords=["SELECT", "COUNT", "WHERE"],
    expected_columns=["total_completed"],
)

_register(
    task_id="easy_3",
    difficulty="easy",
    description=(
        "The merchandising team is building a premium product catalogue. "
        "List the top 5 most expensive products by price, highest first. "
        "Return columns: name, category, price."
    ),
    reference_sql=(
        "SELECT name, category, price FROM products "
        "ORDER BY price DESC LIMIT 5"
    ),
    required_keywords=["SELECT", "ORDER BY", "LIMIT"],
    expected_columns=["name", "category", "price"],
    ordered=True,
)

# ── MEDIUM ───────────────────────────────────────────────────────────────────

_register(
    task_id="medium_1",
    difficulty="medium",
    description=(
        "The loyalty team wants to identify top spenders for a VIP rewards programme. "
        "For each customer, calculate their total spending across all completed orders. "
        "Include only customers who have placed at least one completed order. "
        "Return columns: name, total_spent. Order by total_spent descending."
    ),
    reference_sql="""
        SELECT c.name, SUM(o.total_amount) AS total_spent
        FROM customers c
        JOIN orders o ON c.id = o.customer_id
        WHERE o.status = 'completed'
        GROUP BY c.id, c.name
        ORDER BY total_spent DESC
    """.strip(),
    required_keywords=["SELECT", "JOIN", "GROUP BY", "SUM"],
    expected_columns=["name", "total_spent"],
    ordered=True,
)

_register(
    task_id="medium_2",
    difficulty="medium",
    description=(
        "The inventory team needs to identify dead-stock items for a clearance sale. "
        "Find all products that have NEVER appeared in any order. "
        "Return columns: name, category, price."
    ),
    reference_sql="""
        SELECT p.name, p.category, p.price
        FROM products p
        LEFT JOIN order_items oi ON p.id = oi.product_id
        WHERE oi.id IS NULL
    """.strip(),
    required_keywords=["SELECT", "LEFT JOIN", "WHERE"],
    expected_columns=["name", "category", "price"],
)

_register(
    task_id="medium_3",
    difficulty="medium",
    description=(
        "The finance team is building a monthly revenue trend report for 2023. "
        "Calculate the average order value for each month in 2023. "
        "Format the month as 'YYYY-MM'. "
        "Return columns: month, avg_order_value. Order by month ascending."
    ),
    reference_sql="""
        SELECT STRFTIME('%Y-%m', order_date) AS month,
               ROUND(AVG(total_amount), 2)   AS avg_order_value
        FROM orders
        WHERE order_date LIKE '2023%'
        GROUP BY month
        ORDER BY month ASC
    """.strip(),
    required_keywords=["SELECT", "AVG", "GROUP BY"],
    expected_columns=["month", "avg_order_value"],
    ordered=True,
)

# ── HARD ──────────────────────────────────────────────────────────────────────

_register(
    task_id="hard_1",
    difficulty="hard",
    description=(
        "The CRM team wants to segment high-value customers for a premium tier. "
        "Find all customers whose total completed spending is above the average "
        "total spending across all customers who have made at least one completed order. "
        "Use a CTE for clarity. "
        "Return columns: name, total_spent. Order by total_spent descending."
    ),
    reference_sql="""
        WITH customer_totals AS (
            SELECT c.name, SUM(o.total_amount) AS total_spent
            FROM customers c
            JOIN orders o ON c.id = o.customer_id
            WHERE o.status = 'completed'
            GROUP BY c.id, c.name
        )
        SELECT name, total_spent
        FROM customer_totals
        WHERE total_spent > (SELECT AVG(total_spent) FROM customer_totals)
        ORDER BY total_spent DESC
    """.strip(),
    required_keywords=["SELECT", "GROUP BY", "AVG"],
    expected_columns=["name", "total_spent"],
    ordered=True,
)

_register(
    task_id="hard_2",
    difficulty="hard",
    description=(
        "The category management team needs a 'category hero' SKU report for the homepage. "
        "For each product category, find the best-selling product by total quantity sold. "
        "If two products tie, return both (use RANK, not ROW_NUMBER). "
        "Return columns: category, product_name, total_quantity."
    ),
    reference_sql="""
        WITH product_sales AS (
            SELECT p.category,
                   p.name AS product_name,
                   SUM(oi.quantity) AS total_quantity
            FROM products p
            JOIN order_items oi ON p.id = oi.product_id
            GROUP BY p.id, p.category, p.name
        ),
        ranked AS (
            SELECT category, product_name, total_quantity,
                   RANK() OVER (PARTITION BY category ORDER BY total_quantity DESC) AS rnk
            FROM product_sales
        )
        SELECT category, product_name, total_quantity
        FROM ranked
        WHERE rnk = 1
        ORDER BY category, product_name
    """.strip(),
    required_keywords=["SELECT", "GROUP BY", "JOIN"],
    expected_columns=["category", "product_name", "total_quantity"],
)

_register(
    task_id="hard_3",
    difficulty="hard",
    description=(
        "The retention team wants to reward customers who have been active every "
        "year since the platform launched. Find customers who placed at least one "
        "order in each of the years 2022, 2023, AND 2024 (all three). "
        "Return columns: name, email. Order by name ascending."
    ),
    reference_sql="""
        SELECT c.name, c.email
        FROM customers c
        WHERE (
            SELECT COUNT(DISTINCT STRFTIME('%Y', o.order_date))
            FROM orders o
            WHERE o.customer_id = c.id
              AND STRFTIME('%Y', o.order_date) IN ('2022', '2023', '2024')
        ) = 3
        ORDER BY c.name ASC
    """.strip(),
    required_keywords=["SELECT", "WHERE"],
    expected_columns=["name", "email"],
    ordered=True,
)


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

def grade(
    task_id: str,
    agent_query: str,
    conn: sqlite3.Connection,
) -> Tuple[float, str, List[Dict], List[Dict]]:
    """
    Grade an agent's SQL query against the reference solution.

    Returns:
        (reward, message, agent_rows, expected_rows)
        reward: float 0.0-1.0
        message: human-readable feedback
        agent_rows: what the agent's query returned
        expected_rows: what the reference returned
    """
    task = TASKS.get(task_id)
    if task is None:
        return 0.0, f"Unknown task_id: {task_id}", [], []

    # Run reference solution
    try:
        cur = conn.execute(task["reference_sql"])
        cols = [d[0] for d in cur.description]
        expected_rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as e:
        return 0.0, f"Reference SQL failed (bug): {e}", [], []

    # Run agent query
    try:
        cur = conn.execute(agent_query)
        cols = [d[0] for d in cur.description]
        agent_rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as e:
        return 0.0, f"Query error: {e}", [], expected_rows

    # Score correctness
    correctness = _rows_match(agent_rows, expected_rows, task["ordered"])

    # Keyword bonus (checks structural correctness)
    kw_bonus = 0.1 if _has_required_keywords(agent_query, task["required_keywords"]) else 0.0

    # Efficiency bonus
    efficiency = _query_complexity_penalty(agent_query)

    # Total reward (capped at 1.0)
    reward = min(1.0, correctness * 0.7 + kw_bonus + efficiency)

    # Build message
    if correctness >= 0.99:
        msg = "Correct! Full marks for result accuracy."
    elif correctness >= 0.5:
        msg = f"Partially correct ({correctness:.0%} rows matched)."
    else:
        msg = f"Incorrect results ({correctness:.0%} rows matched)."

    if reward == 1.0:
        msg += " Perfect score!"

    return round(reward, 4), msg, agent_rows, expected_rows


def get_task_by_difficulty(difficulty: str) -> Optional[Dict]:
    """Return the first task matching the given difficulty."""
    for task in TASKS.values():
        if task["difficulty"] == difficulty:
            return task
    return None


def get_all_tasks_by_difficulty(difficulty: str) -> List[Dict]:
    return [t for t in TASKS.values() if t["difficulty"] == difficulty]
