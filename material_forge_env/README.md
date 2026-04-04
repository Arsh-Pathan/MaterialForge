---
title: MaterialForge
emoji: 🔬
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - materials-science
  - crystal-structure
---

# MaterialForge

**AI-Driven Atomic Crystal Structure Design**

MaterialForge is an OpenEnv reinforcement learning environment where an AI agent designs atomic crystal structures on an 8x8 lattice grid to match target material properties.

Given a target specification (hardness, conductivity, thermal resistance, elasticity), the agent places, replaces, and removes atoms to construct a crystal structure that satisfies the requirements within a cost budget.

## Try It

Use the **Playground** tab above to interact with the environment, or connect programmatically via the API.

### Python Client

```python
from material_forge_env import MaterialForgeAction, MaterialForgeEnv

with MaterialForgeEnv(base_url="https://ArshPathan-material-forge-env.hf.space") as env:
    result = env.reset()
    print(f"Target: {result.observation.target}")
    print(f"Budget: {result.observation.cost_budget}")

    # Place a metal atom at (0, 0)
    result = env.step(MaterialForgeAction(action_type="place", row=0, col=0, atom="A"))
    print(f"Properties: {result.observation.current_properties}")
    print(f"Reward: {result.reward:.4f}")
```

### REST API

```bash
# Reset environment
curl -X POST https://ArshPathan-material-forge-env.hf.space/reset

# Place an atom
curl -X POST https://ArshPathan-material-forge-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "place", "row": 0, "col": 0, "atom": "A"}'

# Health check
curl https://ArshPathan-material-forge-env.hf.space/health
```

Full API docs: [/docs](https://ArshPathan-material-forge-env.hf.space/docs)

## Environment Details

### Atom Types

| Symbol | Name | Cost | Primary Strength |
|--------|------|------|------------------|
| **A** | Metal | 8 | Hardness (0.85) |
| **B** | Conductor | 6 | Conductivity (0.90) |
| **C** | Ceramic | 4 | Thermal Resistance (0.85) |
| **P** | Polymer | 2 | Elasticity (0.85) |

### Action Space

```
action_type: "place" | "replace" | "remove"
row: 0-7
col: 0-7
atom: "A" | "B" | "C" | "P"  (required for place/replace)
```

### Observation Space

| Field | Description |
|-------|-------------|
| `grid` | 8x8 lattice state (atom symbols or `"."` for empty) |
| `target` | Target property values (hardness, conductivity, thermal_resistance, elasticity) |
| `current_properties` | Estimated properties of the current structure (0-100 scale) |
| `phase` | Crystal phase: `"crystalline"`, `"polycrystalline"`, or `"amorphous"` |
| `total_cost` / `cost_budget` | Current spend vs. allowed budget |
| `step_number` / `max_steps` | Episode progress |
| `score_breakdown` | Component-wise reward breakdown |
| `reward` | Scalar reward signal (0.0 - 1.0) |
| `done` | Whether the episode has ended |

### Reward Formula

```
reward = 0.50 x property_match
       + 0.25 x stability
       + 0.15 x lattice_quality
       + 0.10 x phase_bonus
       - cost_penalty
```

### Difficulty Presets

| Difficulty | Tolerance | Cost Budget | Max Steps |
|------------|-----------|-------------|-----------|
| Easy | 20 | 120 | 64 |
| Medium | 10 | 80 | 50 |
| Hard | 5 | 60 | 40 |

### Named Scenarios

- **diamond-like** — High hardness (90), low elasticity
- **conductor** — High conductivity (90), low thermal resistance
- **heat-shield** — High thermal resistance (90), low conductivity
- **flexible-polymer** — High elasticity (85), low hardness
- **balanced-alloy** — Balanced properties (~50 each)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Reset environment (accepts `difficulty`, `scenario_name`) |
| `/step` | POST | Execute an action |
| `/state` | GET | Get current environment state |
| `/schema` | GET | Action/observation JSON schemas |
| `/health` | GET | Health check |
| `/ws` | WS | WebSocket for persistent sessions |
| `/docs` | GET | Swagger UI |

## Links

- [GitHub Repository](https://github.com/Arsh-Pathan/MaterialForge)
- Built with [OpenEnv](https://github.com/meta-pytorch/openenv) for the Meta PyTorch OpenEnv Hackathon
