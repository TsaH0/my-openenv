"""
server/app.py - FastAPI server exposing the SQL Environment over HTTP and WebSocket.

Endpoints:
  POST /reset             - Start a new episode
  POST /step              - Execute a SQL action
  GET  /state             - Get current environment state
  GET  /health            - Health check (returns HTTP 200)
  WebSocket /ws           - Real-time bidirectional communication

Compatible with the OpenEnv client interface.

Usage:
  uvicorn server.app:app --host 0.0.0.0 --port 7860
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any as _Any
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import SQLAction, SQLObservation, SQLState
from server.sql_environment import SQLEnvironment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic request/response models (for FastAPI schema generation)
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    difficulty: str = "easy"
    task_id: Optional[str] = None


class StepRequest(BaseModel):
    query: str
    task_id: Optional[str] = None
    difficulty: str = "easy"


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

# Map episode_id -> SQLEnvironment instance
_sessions: Dict[str, SQLEnvironment] = {}
_default_session_id = "default"


def _get_or_create_session(session_id: str = _default_session_id) -> SQLEnvironment:
    if session_id not in _sessions:
        _sessions[session_id] = SQLEnvironment()
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise default session. Shutdown: close all sessions."""
    env = _get_or_create_session(_default_session_id)
    env.reset()
    logger.info("SQL Environment server started. Default session ready.")
    yield
    for env in _sessions.values():
        env.close()
    logger.info("SQL Environment server stopped.")


app = FastAPI(
    title="SQL Query Learning Environment",
    description=(
        "An OpenEnv-compatible reinforcement learning environment where "
        "agents learn to write correct and efficient SQL queries."
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


# ---------------------------------------------------------------------------
# HTTP Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check - required by OpenEnv deployment validation."""
    return JSONResponse({"status": "ok", "environment": "sql_env", "version": "1.0.0"})


@app.post("/reset")
async def reset(
    request: Optional[ResetRequest] = Body(default=None),
    session_id: str = _default_session_id,
):
    """
    Initialize a new episode.

    Returns:
        SQLObservation as JSON with schema_info and first task description.
    """
    env = _get_or_create_session(session_id)
    difficulty = request.difficulty if request else "easy"
    task_id = request.task_id if request else None
    obs: SQLObservation = env.reset(difficulty=difficulty, task_id=task_id)
    return _obs_to_dict(obs)


@app.post("/step")
async def step(request: StepRequest, session_id: str = _default_session_id):
    """
    Execute a SQL query action.

    Returns:
        SQLObservation as JSON with result rows, reward, and feedback.
    """
    env = _get_or_create_session(session_id)
    action = SQLAction(
        query=request.query,
        task_id=request.task_id,
        difficulty=request.difficulty,
    )
    obs: SQLObservation = env.step(action)
    return _obs_to_dict(obs)


@app.get("/state")
async def get_state(session_id: str = _default_session_id):
    """
    Return current episode state metadata.

    Returns:
        SQLState as JSON with episode_id, step_count, total_reward, etc.
    """
    env = _get_or_create_session(session_id)
    return env.state.model_dump()


@app.get("/tasks")
async def list_tasks():
    """List all available tasks grouped by difficulty."""
    from server.tasks import TASKS
    grouped: Dict[str, list] = {"easy": [], "medium": [], "hard": []}
    for task in TASKS.values():
        entry = {
            "id": task["id"],
            "description": task["description"],
            "expected_columns": task["expected_columns"],
        }
        grouped[task["difficulty"]].append(entry)
    return grouped


@app.get("/schema")
async def get_schema():
    """Return the database schema description."""
    from server.tasks import SCHEMA_INFO
    return {"schema": SCHEMA_INFO}


@app.get("/", response_class=HTMLResponse)
async def root():
    """Simple web interface for manual testing."""
    return HTMLResponse(content=_web_ui_html(), status_code=200)


# ---------------------------------------------------------------------------
# WebSocket Endpoint
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time agent interaction.

    Message format (JSON):
      { "action": "reset", "difficulty": "easy" }
      { "action": "step",  "query": "SELECT ...", "difficulty": "easy" }
      { "action": "state" }

    Response format (JSON):
      { "type": "observation"|"state"|"error", "data": {...} }
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    env = SQLEnvironment()
    _sessions[session_id] = env
    logger.info(f"WebSocket session opened: {session_id}")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "data": {"message": "Invalid JSON"},
                }))
                continue

            action_type = msg.get("action", "")

            if action_type == "reset":
                obs = env.reset(
                    difficulty=msg.get("difficulty", "easy"),
                    task_id=msg.get("task_id"),
                )
                await websocket.send_text(json.dumps({
                    "type": "observation",
                    "data": _obs_to_dict(obs),
                }))

            elif action_type == "step":
                query = msg.get("query", "")
                if not query:
                    await websocket.send_text(json.dumps({
                        "type": "error",
                        "data": {"message": "Missing 'query' field"},
                    }))
                    continue
                action = SQLAction(
                    query=query,
                    task_id=msg.get("task_id"),
                    difficulty=msg.get("difficulty", "easy"),
                )
                obs = env.step(action)
                await websocket.send_text(json.dumps({
                    "type": "observation",
                    "data": _obs_to_dict(obs),
                }))

            elif action_type == "state":
                await websocket.send_text(json.dumps({
                    "type": "state",
                    "data": env.state.model_dump(),
                }))

            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "data": {"message": f"Unknown action: {action_type}"},
                }))

    except WebSocketDisconnect:
        logger.info(f"WebSocket session closed: {session_id}")
    finally:
        env.close()
        _sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs_to_dict(obs: SQLObservation) -> Dict[str, Any]:
    return obs.model_dump()


def _web_ui_html() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>SQL Query Learning Environment</title>
  <style>
    body { font-family: monospace; max-width: 900px; margin: 40px auto; padding: 0 20px; background: #1e1e2e; color: #cdd6f4; }
    h1 { color: #89b4fa; }
    h2 { color: #a6e3a1; font-size: 1em; margin-top: 20px; }
    textarea { width: 100%; height: 120px; background: #313244; color: #cdd6f4; border: 1px solid #585b70; padding: 8px; font-family: monospace; font-size: 13px; }
    button { background: #89b4fa; color: #1e1e2e; border: none; padding: 8px 16px; cursor: pointer; margin: 4px; font-weight: bold; }
    button:hover { background: #74c7ec; }
    select { background: #313244; color: #cdd6f4; border: 1px solid #585b70; padding: 6px; }
    pre { background: #313244; padding: 12px; overflow: auto; font-size: 12px; max-height: 400px; border: 1px solid #585b70; }
    .reward { color: #a6e3a1; font-size: 1.2em; font-weight: bold; }
    .error  { color: #f38ba8; }
    .task   { background: #2a273f; padding: 10px; border-left: 3px solid #cba6f7; margin: 10px 0; }
  </style>
</head>
<body>
  <h1>SQL Query Learning Environment</h1>
  <p>An OpenEnv RL environment for learning SQL against an e-commerce database.</p>

  <h2>1. Start Episode</h2>
  Difficulty:
  <select id="difficulty">
    <option value="easy">Easy</option>
    <option value="medium">Medium</option>
    <option value="hard">Hard</option>
  </select>
  <button onclick="doReset()">Reset / New Episode</button>

  <div id="taskBox" class="task" style="display:none">
    <b>Task:</b> <span id="taskDesc"></span><br>
    <b>Expected columns:</b> <span id="taskCols"></span>
  </div>

  <h2>2. Submit SQL Query</h2>
  <textarea id="query" placeholder="SELECT name, email FROM customers WHERE country = 'USA'"></textarea>
  <br>
  <button onclick="doStep()">Execute Query</button>
  <button onclick="doState()">Get State</button>

  <h2>3. Result</h2>
  <div class="reward">Reward: <span id="reward">-</span></div>
  <div id="message"></div>
  <pre id="output">Response will appear here...</pre>

  <script>
    const base = window.location.origin;

    async function doReset() {
      const diff = document.getElementById('difficulty').value;
      const res = await fetch(base + '/reset', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ difficulty: diff })
      });
      const data = await res.json();
      document.getElementById('output').textContent = JSON.stringify(data, null, 2);
      document.getElementById('reward').textContent = data.reward ?? '-';
      document.getElementById('message').textContent = data.message ?? '';
      document.getElementById('taskDesc').textContent = data.task_description ?? '';
      document.getElementById('taskCols').textContent = (data.expected_columns ?? []).join(', ');
      document.getElementById('taskBox').style.display = 'block';
    }

    async function doStep() {
      const query = document.getElementById('query').value;
      const diff  = document.getElementById('difficulty').value;
      const res = await fetch(base + '/step', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ query, difficulty: diff })
      });
      const data = await res.json();
      document.getElementById('output').textContent = JSON.stringify(data, null, 2);
      document.getElementById('reward').textContent = data.reward ?? '-';
      document.getElementById('message').textContent = data.message ?? '';
      if (data.task_description) {
        document.getElementById('taskDesc').textContent = data.task_description;
        document.getElementById('taskCols').textContent = (data.expected_columns ?? []).join(', ');
      }
    }

    async function doState() {
      const res = await fetch(base + '/state');
      const data = await res.json();
      document.getElementById('output').textContent = JSON.stringify(data, null, 2);
    }
  </script>
</body>
</html>
""".strip()


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
