"""FastAPI + WebSocket dashboard server."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from harness.dashboard.state import DashboardState

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Context Gathering Harness — Dashboard")

# Global state reference (set by the runner before starting the server)
_state: DashboardState | None = None


def set_state(state: DashboardState) -> None:
    """Inject the shared experiment state into the server module."""
    global _state
    _state = state


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the dashboard HTML."""
    index_path = STATIC_DIR / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """Push state snapshots to connected dashboard clients every second."""
    await ws.accept()
    try:
        while True:
            if _state is not None:
                snapshot = _state.snapshot()
                await ws.send_json(snapshot)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass


def start_dashboard_server(
    state: DashboardState,
    port: int = 8765,
    host: str = "127.0.0.1",
) -> threading.Thread:
    """Start the dashboard server in a background daemon thread.

    Returns the thread handle.
    """
    import uvicorn

    set_state(state)

    def _run():
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="warning",
        )

    thread = threading.Thread(target=_run, daemon=True, name="dashboard-server")
    thread.start()
    return thread
