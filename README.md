# MaterialForge

**An OpenEnv environment for AI-driven atomic crystal structure design.**

MaterialForge is a reinforcement learning environment where an AI agent designs atomic crystal structures on an 8x8 lattice grid to match target material properties (hardness, conductivity, thermal resistance, elasticity). Built with the [OpenEnv](https://github.com/meta-pytorch/openenv) framework for the Meta PyTorch OpenEnv Hackathon.

**Live Demo:** [huggingface.co/spaces/ArshPathan/material_forge_env](https://huggingface.co/spaces/ArshPathan/material_forge_env)

## The Problem

In materials science, the physical properties of a material depend heavily on atomic arrangement — carbon as diamond is extremely hard, while carbon as graphite is soft and conductive. Discovering new materials requires exploring a massive design space of possible atomic configurations.

MaterialForge frames this as an optimization task: given a target material specification, an AI agent iteratively constructs an atomic structure that satisfies the requirements.

## How It Works

```
Episode start → Agent receives target properties + empty 8x8 grid
     ↓
Agent places/replaces/removes atoms (A=Metal, B=Conductor, C=Ceramic, P=Polymer)
     ↓
Physics engine estimates properties from structure (bonding, density, symmetry)
     ↓
Reward signal based on property match, stability, quality, phase, cost
     ↓
Repeat until target met or step limit reached
```

### Atom Types

| Symbol | Name | Cost | Primary Strength |
|--------|------|------|------------------|
| **A** | Metal | 8 | Hardness |
| **B** | Conductor | 6 | Conductivity |
| **C** | Ceramic | 4 | Thermal Resistance |
| **P** | Polymer | 2 | Elasticity |

### Reward Formula

```
reward = 0.50 * property_match + 0.25 * stability + 0.15 * lattice_quality + 0.10 * phase_bonus - cost_penalty
```

## Quick Start

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager

### Install & Run

```bash
cd material_forge_env
uv sync
uv run server              # starts on http://localhost:8000
uv run server --port 7860  # or specify a port
```

### Connect with Python

```python
from material_forge_env import MaterialForgeAction, MaterialForgeEnv

with MaterialForgeEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Target: {result.observation.target}")

    # Build a structure
    actions = [
        MaterialForgeAction(action_type="place", row=0, col=0, atom="A"),
        MaterialForgeAction(action_type="place", row=0, col=1, atom="A"),
        MaterialForgeAction(action_type="place", row=1, col=0, atom="B"),
        MaterialForgeAction(action_type="place", row=1, col=1, atom="C"),
    ]

    for action in actions:
        result = env.step(action)
        print(f"Step {result.observation.step_number}: "
              f"reward={result.reward:.4f}, phase={result.observation.phase}")
```

### REST API

```bash
# Reset with a named scenario
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy", "scenario_name": "diamond-like"}'

# Place an atom
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "place", "row": 3, "col": 3, "atom": "A"}'
```

Interactive API docs at [localhost:8000/docs](http://localhost:8000/docs).

## Deployment

### Docker

```bash
docker build -t material-forge-env:latest material_forge_env/server/
docker run -p 8000:8000 material-forge-env:latest
```

### Hugging Face Spaces

```bash
cd material_forge_env
uv run openenv validate   # check everything is correct
uv run openenv push       # deploy to HF Spaces
```

## Project Structure

```
MaterialForge/
├── README.md
├── CLAUDE.md                 # Claude Code project guidance
├── IDEA.md                   # Original concept document
├── PLAN.md                   # Implementation plan
└── material_forge_env/       # OpenEnv environment package
    ├── config.py             # Constants: grid size, atom types, difficulty presets
    ├── models.py             # MaterialForgeAction & MaterialForgeObservation (Pydantic)
    ├── lattice.py            # 8x8 grid engine: place, replace, remove, neighbors
    ├── physics.py            # Property estimation, phase classification, stability
    ├── rubrics.py            # HeuristicRewardRubric — reward computation
    ├── scenarios.py          # Target profile generators (5 named + random)
    ├── client.py             # MaterialForgeEnv WebSocket client
    ├── openenv.yaml          # OpenEnv manifest
    ├── pyproject.toml        # Package metadata & dependencies
    ├── inference.py              # Inference script for hackathon submission
    ├── Dockerfile                # Container image
    └── server/
        ├── app.py                            # FastAPI application (HTTP + WS)
        └── material_forge_env_environment.py # MaterialForgeEnvironment(Environment)
```

## Environment Details

### Action Space

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `"place"` \| `"replace"` \| `"remove"` | Lattice operation |
| `row` | `0-7` | Grid row |
| `col` | `0-7` | Grid column |
| `atom` | `"A"` \| `"B"` \| `"C"` \| `"P"` | Atom type (required for place/replace) |

### Observation Space

| Field | Description |
|-------|-------------|
| `grid` | 8x8 list of strings (`"."` = empty, `"A"`/`"B"`/`"C"`/`"P"` = atoms) |
| `target` | Dict of target property values |
| `current_properties` | Dict of estimated properties (0-100 scale) |
| `phase` | `"crystalline"` \| `"polycrystalline"` \| `"amorphous"` |
| `total_cost` / `cost_budget` | Atom cost tracking |
| `step_number` / `max_steps` | Episode progress |
| `score_breakdown` | Stability, lattice quality, action validity |
| `reward` | Scalar reward (0.0 - 1.0) |
| `done` | Episode termination flag |

### Difficulty Presets

| Level | Tolerance | Budget | Steps |
|-------|-----------|--------|-------|
| Easy | 20 | 120 | 64 |
| Medium | 10 | 80 | 50 |
| Hard | 5 | 60 | 40 |

### Named Scenarios

| Name | Hardness | Conductivity | Thermal Res. | Elasticity |
|------|----------|--------------|--------------|------------|
| diamond-like | 90 | 30 | 60 | 10 |
| conductor | 25 | 90 | 20 | 30 |
| heat-shield | 50 | 10 | 90 | 15 |
| flexible-polymer | 15 | 20 | 25 | 85 |
| balanced-alloy | 55 | 50 | 50 | 45 |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment |
| `/step` | POST | Execute an action |
| `/state` | GET | Current state |
| `/schema` | GET | Action/observation schemas |
| `/health` | GET | Health check |
| `/ws` | WS | WebSocket (persistent sessions) |
| `/docs` | GET | Swagger UI |

## Architecture

MaterialForge follows the OpenEnv 3-component pattern: **Models -> Environment -> Server**.

```
Client.step(MaterialForgeAction)
  -> WebSocket -> FastAPI (app.py)
  -> MaterialForgeEnvironment.step(action)
    -> lattice.py: apply action (place/replace/remove atom)
    -> physics.py: estimate_properties(lattice)
    -> physics.py: classify_phase(lattice)
    -> rubrics.py: _apply_rubric(action, obs) -> reward
  -> MaterialForgeObservation
  -> WebSocket -> StepResult
```

## License

BSD-style license. See file headers for details.
