"""FastAPI application for the MaterialForge Environment.

Endpoints provided by OpenEnv create_app:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET  /state: Get current environment state
    - GET  /schema: Get action/observation schemas
    - GET  /health: Health check
    - WS   /ws: WebSocket endpoint for persistent sessions
    - GET  /docs: Swagger UI

Custom additions:
    - GET  /ui → serves the HTML/CSS/JS visualisation dashboard
"""

import os
from pathlib import Path

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError("openenv is required. Install dependencies with 'uv sync'") from e

try:
    from ..models import MaterialForgeAction, MaterialForgeObservation
    from ..environment.rubrics import HeuristicRewardRubric
    from .material_forge_env_environment import MaterialForgeEnvironment
except ImportError:
    from models import MaterialForgeAction, MaterialForgeObservation
    from environment.rubrics import HeuristicRewardRubric
    from server.material_forge_env_environment import MaterialForgeEnvironment


def _env_factory():
    """Factory that creates a new environment instance with the heuristic rubric."""
    return MaterialForgeEnvironment(rubric=HeuristicRewardRubric())


app = create_app(
    _env_factory,
    MaterialForgeAction,
    MaterialForgeObservation,
    env_name="material_forge_env",
    max_concurrent_envs=4,
)

if not hasattr(app, "add_api_route"):
    from fastapi import FastAPI

    _app = FastAPI()
    _app.add_api_route("/health", lambda: {"status": "ok"}, methods=["GET"])
    for route in app.routes:
        _app.routes.append(route)
    app = _app

# ── Professional Routing & Static Mounting ──────────────────────────────
# Matches Gridmind-RL pattern: /playground, /api (docs), /
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    from fastapi.responses import FileResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles

    # Serve the whole static folder at /playground
    app.mount("/playground", StaticFiles(directory=str(_STATIC_DIR), html=True), name="playground")

    # API Alias: /api → /docs
    @app.get("/api", include_in_schema=False)
    def api_docs_redirect():
        return RedirectResponse(url="/docs")

    # Convenience redirect: root → /playground
    @app.get("/", include_in_schema=False)
    def root_redirect():
        return FileResponse(str(_STATIC_DIR / "index.html"))


def main():
    """Entry point for direct execution via uv run or python -m.

    Usage:
        uv run server
        uv run server --port 8000
    """
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
