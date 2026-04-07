---
title: MaterialForge
emoji: 🔬
colorFrom: red
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - materials-science
  - crystal-structure
---

# 🔬 MaterialForge — AI-Driven Atomic Crystal Structure Design

---
MaterialForge is an OpenEnv reinforcement learning environment where an AI agent designs atomic crystal structures on an 8x8 lattice grid to match target material properties.

Given a target specification (hardness, conductivity, thermal resistance, elasticity), the agent places, replaces, and removes atoms to construct a crystal structure that satisfies the requirements within a cost budget.

## Hugging Face Space Deployment

This Space is built from OpenEnv environment `material_forge_env`.

- **Space URL**: [huggingface.co/spaces/ArshPathan/material_forge_env](https://huggingface.co/spaces/ArshPathan/material_forge_env)
- **Hub Tag**: `openenv`
- **SDK**: Docker (FastAPI)

### Connecting from Code

Use the `MaterialForgeEnv` client to connect to this environment programmatically.

```python
from material_forge_env import MaterialForgeEnv

# Connect to the Hugging Face Space
env = MaterialForgeEnv(base_url="https://ArshPathan-material-forge-env.hf.space")

# Alternatively, connect to a local server
# env = MaterialForgeEnv(base_url="http://localhost:7860")

with env:
    result = env.reset()
    print(f"Target Properties: {result.observation.target}")
```

## Quick Start

### Python Client Example

```python
from material_forge_env import MaterialForgeAction, MaterialForgeEnv

with MaterialForgeEnv(base_url="https://ArshPathan-material-forge-env.hf.space") as env:
    # Reset to a fresh episode
    result = env.reset()
    
    # Place a metal atom (A) at row 0, col 0
    action = MaterialForgeAction(action_type="place", row=0, col=0, atom="A")
    result = env.step(action)
    
    print(f"Current Properties: {result.observation.current_properties}")
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
```

Full API documentation available at [/docs](https://ArshPathan-material-forge-env.hf.space/docs).

## Environment Details

### Atom Types

| Symbol | Name | Cost | Primary Strength |
|--------|------|------|------------------|
| **A** | Metal | 8 | Hardness (0.85) |
| **B** | Conductor | 6 | Conductivity (0.90) |
| **C** | Ceramic | 4 | Thermal Resistance (0.85) |
| **P** | Polymer | 2 | Elasticity (0.85) |

### Action Space

```python
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
| `current_properties` | Estimated properties of current structure (0-100) |
| `phase` | Crystal phase: `"crystalline"`, `"polycrystalline"`, or `"amorphous"` |
| `total_cost` | Current atom cost spend |
| `reward` | Scalar reward signal (0.0 - 1.0) |
| `done` | Whether requirements met or budget exceeded |

### Reward Formula

```
reward = 0.50 x property_match
       + 0.25 x stability
       + 0.15 x lattice_quality
       + 0.10 x phase_bonus
       - cost_penalty
```

## Scenarios & Difficulty

MaterialForge supports named scenarios (e.g., `diamond-like`, `conductor`, `heat-shield`) and three difficulty levels:

| Difficulty | Tolerance | Cost Budget | Max Steps |
|------------|-----------|-------------|-----------|
| Easy | 20 | 120 | 64 |
| Medium | 10 | 80 | 50 |
| Hard | 5 | 60 | 40 |

## Links

- [GitHub Repository](https://github.com/Arsh-Pathan/MaterialForge)
- Built with [OpenEnv](https://github.com/meta-pytorch/openenv) for the Meta PyTorch OpenEnv Hackathon
