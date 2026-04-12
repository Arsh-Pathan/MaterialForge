---
title: MaterialForge
emoji: 🔬
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# MaterialForge

**Inverse Autonomous Synthesis of Crystalline Atomic Lattices**

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://openenv.org/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11+-3776ab)](https://www.python.org/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🚀 Live Discovery Lab

| Feature | URL |
|--|-----|
| **Environment API** | https://huggingface.co/spaces/ArshPathan/material_forge_env |
| **Interactive Dashboard** | https://huggingface.co/spaces/ArshPathan/material_forge_env/playground |

**Quick health check:**
```bash
curl https://huggingface.co/spaces/ArshPathan/material_forge_env/health
```

---

## Overview

MaterialForge is a high-fidelity reinforcement learning environment for the autonomous discovery and optimization of crystalline structures. Based on an 8&times;8 atomic lattice, agents must position different atomic species to engineer specific macro-physical properties such as **Hardness**, **Conductivity**, **Thermal Resistance**, and **Elasticity**.

**Key Challenges:**
- **Property Matching**: Synthesize materials that match randomly generated physical specifications.
- **Structural Integrity**: Optimize for Gibbs stability and coordinate bonding neighbor density.
- **Lattice Order**: Achieve high-symmetry crystalline phases (from Amorphous to Monocrystalline).
---

## 🌎 Scientific Utility & Real-world Relevance

**MaterialForge** models a genuine scientific challenge: **Inverse Molecular Design**. While simplified to an 8x8 lattice, it simulates critical concepts used in actual materia-science research:

- **Percolation Pathways**: Crucial for designing next-gen battery electrolytes and organic electronics.
- **Coordination Chemistry**: Modeling how local atomic environments dictate global structural stability.
- **Phase Transition Engineering**: Agents must navigate the complex trade-offs between entropy (amorphicity) and order (crystallinity).

By providing a dense reward signal and a clear physics-based rubric, MaterialForge serves as a robust baseline for evaluating RL agents' ability to perform structured, scientific reasoning under budget constraints.

---

## Architecture

MaterialForge follows a strictly decoupled architecture, separating the core physics simulation from the interactive interface.

```
Agent (python/inference.py)
    → HTTP POST /step, /reset
    ↓
FastAPI Server (server/app.py) → Port 7860
    ↓
Physics Engine (environment/material_forge_env_environment.py) 
    + Discovery Logic (environment/rubrics.py)
    ↓
Discovery Lab Dashboard (server/static/index.html) → /playground
```

**Design Philosophy:**
- **OpenEnv Compliance**: Fully standardized REST/WebSocket API.
- **Analytical Precision**: BFS-based percolation path identification and symmetry analysis.
- **Professional Aesthetics**: High-fidelity, real-time visualization dashboard.

---

## Environment Specification

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `grid` | list[list] | 8x8 atomic lattice containing species (A, B, C, P) or empty (.) |
| `current_properties` | dict | hardness, conductivity, thermal_resistance, elasticity |
| `target` | dict | Target physical property values to match |
| `total_cost` | float | Cumulative cost of the current discovery session |
| `cost_budget` | int | Maximum allowed discovery resources |
| `step_number` | int | Current iteration [0-50] |

### Action Space

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | string | `place`, `remove`, or `replace` |
| `row`, `col` | int | Target coordinates [0-7] |
| `atom` | string | Species to place: **A** (Metal), **B** (Conductor), **C** (Ceramic), **P** (Polymer) |

---

## Reward System

The environment provides a dense reward signal composed of four distinct scientific components:

1.  **Property Matching (50%)**: Measures the Euclidean distance between current and target physical vectors.
2.  **Structural Stability (25%)**: Awarded based on coordination numbers and neighbor density.
3.  **Lattice Quality (15%)**: Bonus for achieving clear crystalline phases and symmetry.
4.  **Efficiency Bonus (10%)**: Rewarded for meeting targets while significantly under-budget.

**Normalization:**
All rewards are normalized to the `[0.0, 1.0]` range for standardized agent evaluation.

---

## Output Format

MaterialForge emits machine-parsed STDOUT in accordance with OpenEnv evaluation standards:

`[START] task=<scenario> env=material_forge_env model=<model_name>`  
`[STEP] step=<n> action=<json> reward=<0.00> done=<bool> error=<msg>`  
`[END] success=<bool> steps=<n> score=<final_score> rewards=<list>`

---

## Tasks & Difficuly

| Scenario | Focus | Baseline Score |
|----------|-------|----------------|
| **Basic Synthesis** (Easy) | General balancing | **0.892** |
| **Diamond-like** (Medium) | Hardness + Thermal | **0.876** |
| **Superconductor Analogue** (Hard) | Conductivity Pathways | **0.814** |

---

## Quickstart

### Docker (Recommended)
```bash
docker build -t material-forge .
docker run -p 7860:7860 material-forge
```

### Local Development
```bash
# Install dependencies
uv sync

# Start the environment server
uv run server

# Run the inference agent (requires HF_TOKEN)
set HF_TOKEN=your_token
uv run python inference.py
```

---

## Project Structure
```
MaterialForge/
+-- environment/               # Core Physics & Discovery Logic
+-- server/                    # FastAPI Server & Static Assets
+-- scripts/                   # Evaluation & Automation scripts
+-- tests/                     # Unit tests & Benchmark suites
+-- outputs/                   # Execution logs & Benchmark results
+-- models.py                  # OpenEnv Data Models
+-- inference.py               # Main Agent entry point
+-- openenv.yaml               # Discovery task definitions
+-- Dockerfile                 # Space deployment config
+-- pyproject.toml             # Dependency management
+-- uv.lock                    # Deterministic lockfile
+-- client.py                  # Environment API client
```

---

<div align="center">
  <p>Built for the <b>Meta PyTorch OpenEnv Hackathon</b> by Arsh Pathan</p>
</div>
