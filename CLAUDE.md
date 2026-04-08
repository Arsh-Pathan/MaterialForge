# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MaterialForge is an OpenEnv-compliant RL environment where an AI agent designs atomic crystal structures on an 8×8 lattice grid to match target material properties (hardness, conductivity, thermal resistance, elasticity). Built for the Meta PyTorch OpenEnv Hackathon (Round 1 deadline: April 8, 2026).

## Commands

All commands run from `material_forge_env/` directory. Dependencies are managed by `uv`.

```bash
# Install dependencies
uv sync

# Run server (development)
uv run server                          # http://localhost:8000
uv run server --port 7860              # HF Spaces port

# Run server (alternative)
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Run tests
uv run pytest
uv run pytest tests/test_lattice.py -k "test_place"   # single test

# Build & run Docker
docker build -t material-forge-env:latest server/
docker run -p 8000:8000 material-forge-env:latest

# Deploy to Hugging Face Spaces
openenv push

# Validate OpenEnv compliance
openenv validate
```

## Architecture

The project follows the OpenEnv 3-component pattern: **Models → Environment → Server**.

### Data Flow

```
Client.step(MaterialForgeAction)
  → WebSocket → FastAPI (app.py)
  → MaterialForgeEnvironment.step(action)
    → lattice.py: apply action (place/replace/remove atom)
    → physics.py: estimate_properties(lattice) → {hardness, conductivity, ...}
    → physics.py: classify_phase(lattice) → "crystalline"|"polycrystalline"|"amorphous"
    → rubrics.py: _apply_rubric(action, obs) → reward float
  → MaterialForgeObservation (grid, properties, phase, reward, done)
  → WebSocket → StepResult[MaterialForgeObservation]
```

### Key Modules (inside `material_forge_env/`)

| Module | Role |
|--------|------|
| `models.py` | `MaterialForgeAction` (place/replace/remove + row/col/atom) and `MaterialForgeObservation` (grid, target, current_properties, phase, cost, score_breakdown) |
| `server/material_forge_env_environment.py` | `MaterialForgeEnvironment(Environment)` — reset/step/state implementation |
| `server/app.py` | `create_app(MaterialForgeEnvironment, Action, Observation)` → FastAPI with HTTP + WebSocket |
| `client.py` | `MaterialForgeEnv(EnvClient)` — WebSocket client with serialize/parse methods |
| `config.py` | Constants: GRID_SIZE, MAX_STEPS, ATOM_TYPES dict, PROPERTY_NAMES, difficulty presets |
| `lattice.py` | `Lattice` class — grid manipulation, neighbor queries, cost calculation |
| `physics.py` | Heuristic property estimation + phase classification + stability scoring |
| `scenarios.py` | Target profile generators (easy/medium/hard + predefined archetypes) |
| `rubrics.py` | `HeuristicRewardRubric(Rubric)` — computes reward via property matching + stability + phase |

### OpenEnv Framework Patterns

- `Environment.__init__(transform, rubric)` — pass rubric in constructor
- `Observation` has `extra="forbid"` — all custom fields must be explicitly declared
- `_apply_rubric(action, obs)` / `_apply_rubric_async(action, obs)` — built-in reward helpers
- `_reset_rubric()` — call in reset() to clear rubric trajectory state
- `Rubric.forward(action, obs) -> float` — implement for custom reward logic
- `WeightedSum(rubrics, weights)` — weights must sum to 1.0
- `create_app` takes a class (factory), not an instance — one Environment per WebSocket session
- `SUPPORTS_CONCURRENT_SESSIONS = True` — set only when state is session-isolated

### Reward Formula

```
reward = 0.50 × property_match + 0.25 × stability + 0.15 × lattice_quality + 0.10 × phase_bonus − cost_penalty
```

### LLM Integration

All LLM features gracefully degrade without `HF_TOKEN`. When available:
- `rubrics.py`: WeightedSum(heuristic=0.7, llm_judge=0.3)
- `hint_engine.py`: contextual hints via HF Inference API
- `scenarios.py`: LLM-generated creative scenarios with `llm_easy/medium/hard` prefix

## Server Endpoints

Provided by OpenEnv `create_app`: `POST /reset`, `POST /step`, `GET /state`, `GET /schema`, `GET /health`, `WS /ws`, `GET /docs` (Swagger).
