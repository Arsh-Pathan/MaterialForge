"""
MaterialForge Environment Server
================================
Main FastAPI application entry point for the MaterialForge OpenEnv environment.
It provides a high-performance REST and WebSocket interface for RL agents.

Core Endpoints:
- /reset: Re-initialize the crystal design workspace.
- /step: Execute an atomic placement or replacement action.
- /playground: Interactive web dashboard for human visualization.
"""

import os
from pathlib import Path

# OpenEnv runtime integration: connects our logic to the standardized agent protocol.
try:
    from openenv.core.env_server.http_server import create_fastapi_app
except ImportError as e:
    raise ImportError("OpenEnv-Core is required. Please install via 'uv sync'.") from e

# Internal Module Imports (Flat Repository Structure)
from models import MaterialForgeAction, MaterialForgeObservation
from environment.rubrics import HeuristicRewardRubric
from server.material_forge_env_environment import MaterialForgeEnvironment


# Factory function to instantiate the environment with the specific physics rubric.
def _env_factory():
    """Environment instance factory for the FastAPI server wrapper."""
    return MaterialForgeEnvironment(rubric=HeuristicRewardRubric())


# Initializes the FastAPI app that agents connect to for /reset and /step.
app = create_fastapi_app(
    _env_factory,
    MaterialForgeAction,
    MaterialForgeObservation,
    max_concurrent_envs=4,
)

# ── Dynamic Route Customization ──────────────────────────────────────────

# Ensures the health check route exists for container orchestration (Docker/Spaces).
if not hasattr(app, "add_api_route"):
    from fastapi import FastAPI
    _app = FastAPI()
    _app.add_api_route("/health", lambda: {"status": "ok"}, methods=["GET"])
    for route in app.routes:
        _app.routes.append(route)
    app = _app

# ── Static Asset Management (Laboratory Dashboard) ──────────────────────────

# Mounts the 'Discovery Lab' UI so judges can visualize the agent's progress.
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    from fastapi.responses import FileResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles

    # Mount the static directory at /playground to serve the interactive UI.
    app.mount("/playground", StaticFiles(directory=str(_STATIC_DIR), html=True), name="playground")

    # Handle legacy system links without redirects to avoid HF proxy breakage.
    @app.get("/api", include_in_schema=False)
    def api_docs_redirect():
        return RedirectResponse(url="/docs")

    @app.get("/web", include_in_schema=False)
    def web_legacy_redirect():
        return FileResponse(str(_STATIC_DIR / "index.html"))

    @app.get("/playground", include_in_schema=False)
    def serve_playground():
        return FileResponse(str(_STATIC_DIR / "index.html"))

    @app.get("/", include_in_schema=False)
    def serve_dashboard():
        """Default landing page."""
        return FileResponse(str(_STATIC_DIR / "index.html"))

    # Health probe for automated deployment checks.
    @app.get("/health")
    async def health():
        return {"status": "healthy"}


    # Reorder routes so custom UI logic takes precedence over default OpenEnv paths.
    custom_routes = []
    for i in range(len(app.routes) - 1, -1, -1):
        r = app.routes[i]
        if hasattr(r, "path") and r.path in ("/", "/web", "/playground", "/api", "/health"):
            custom_routes.append(app.routes.pop(i))
        elif hasattr(r, "path") and r.path.startswith("/playground"):
            custom_routes.append(app.routes.pop(i))
    
    for r in reversed(custom_routes):
        app.routes.insert(0, r)


# CLI Entry point: starts the uvicorn server for local development or Docker.
def main():
    """Server entry point for CLI and container launch."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="MaterialForge Environment Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", 8000)), help="Binding port")
    
    args = parser.parse_args()
    
    print(f"\n🚀 MaterialForge Laboratory starting on http://{args.host}:{args.port}")
    print(f"🔬 Visual Playground: http://{args.host}:{args.port}/playground\n")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
