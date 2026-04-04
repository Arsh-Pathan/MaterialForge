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
    raise ImportError(
        "openenv is required. Install dependencies with 'uv sync'"
    ) from e

try:
    from ..models import MaterialForgeAction, MaterialForgeObservation
    from ..rubrics import HeuristicRewardRubric
    from .material_forge_env_environment import MaterialForgeEnvironment
except ImportError:
    from models import MaterialForgeAction, MaterialForgeObservation
    from rubrics import HeuristicRewardRubric
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
