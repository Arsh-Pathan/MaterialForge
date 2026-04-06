"""FastAPI application for the MaterialForge Environment.

Endpoints provided by OpenEnv create_app:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - GET /health: Health check
    - WS /ws: WebSocket endpoint for persistent sessions
    - GET /docs: Swagger UI
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError("openenv is required. Install dependencies with 'uv sync'") from e

import os

try:
    from ..models import MaterialForgeAction, MaterialForgeObservation
    from ..environment.rubrics import HeuristicRewardRubric
    from .material_forge_env_environment import MaterialForgeEnvironment
    from .gradio_ui import build_gradio_frontend
except ImportError:
    from models import MaterialForgeAction, MaterialForgeObservation
    from environment.rubrics import HeuristicRewardRubric
    from server.material_forge_env_environment import MaterialForgeEnvironment
    from server.gradio_ui import build_gradio_frontend


def _env_factory():
    """Factory that creates a new environment instance with the heuristic rubric."""
    return MaterialForgeEnvironment(rubric=HeuristicRewardRubric())


# Enable Gradio web interface when ENABLE_WEB_INTERFACE=true or on HF Spaces
_enable_web = os.getenv("ENABLE_WEB_INTERFACE", "").lower() in ("true", "1", "yes")
_on_hf_spaces = os.getenv("SPACE_ID") is not None

_create_kwargs = dict(
    env_name="material_forge_env",
    max_concurrent_envs=4,
)

if _enable_web or _on_hf_spaces:
    _create_kwargs["gradio_builder"] = build_gradio_frontend

app = create_app(
    _env_factory,
    MaterialForgeAction,
    MaterialForgeObservation,
    **_create_kwargs,
)

if not hasattr(app, "add_api_route"):
    from fastapi import FastAPI

    _app = FastAPI()
    _app.add_api_route("/health", lambda: {"status": "ok"}, methods=["GET"])
    for route in app.routes:
        _app.routes.append(route)
    app = _app


def main():
    """Entry point for direct execution via uv run or python -m.

    Usage:
        uv run server
        uv run server --port 7860
    """
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
