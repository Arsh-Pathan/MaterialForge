---
title: MaterialForge
emoji: "🔬"
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

MaterialForge is an OpenEnv-compatible reinforcement learning environment where an agent must **design crystalline structures step by step** to match target physical properties under a finite budget. The environment is built to train and evaluate LLM agents on a scientific workflow rather than a static prompt task.

## Live Links

| Feature | URL |
|--|--|
| Environment API | https://huggingface.co/spaces/ArshPathan/material_forge_env |
| Interactive Dashboard | https://huggingface.co/spaces/ArshPathan/material_forge_env/playground |
| Training Blog | [BLOG.md](BLOG.md) |
| Research / Strategy Notes | [RESEARCH.md](RESEARCH.md) |
| Training Notebook | [training/MaterialForge_GRPO_Training.ipynb](training/MaterialForge_GRPO_Training.ipynb) |

Quick health check:

```bash
curl https://huggingface.co/spaces/ArshPathan/material_forge_env/health
```

## Problem

The core task is:

**Given target values for hardness, conductivity, thermal resistance, and elasticity, can an agent construct a stable, ordered crystal lattice that matches those targets without wasting budget?**

The agent does not solve this in one shot. It must:

- observe the current 8x8 lattice state
- choose an action
- see how the action changes material properties
- recover from overshooting and bad structure
- build toward a high-quality final phase

This makes MaterialForge a good fit for:

- **Theme #3.1: World Modeling**
- **Theme #2: Long-Horizon Planning**

Those theme choices are directly aligned with the hackathon guidance in `docs/` and the internal project strategy in [RESEARCH.md](RESEARCH.md).

## Why This Matters

MaterialForge is inspired by **inverse material design**. Even though the simulation is simplified to an 8x8 lattice, the environment captures several real scientific ideas:

- **Percolation pathways** for conductivity
- **Coordination-driven stability** for local structure quality
- **Phase / order formation** for crystalline vs amorphous behavior
- **Budget-constrained design** where good science is not only about accuracy but also efficiency

The result is a training environment where the model must do more than fill a grid. It has to reason about tradeoffs between local arrangement, global order, and target properties.

## What the Agent Sees and Does

### Observation Space

| Field | Description |
|--|--|
| `grid` | 8x8 atomic lattice containing `A`, `B`, `C`, `P`, or `.` |
| `target` | Target values for hardness, conductivity, thermal resistance, elasticity |
| `current_properties` | Current estimated properties of the lattice |
| `phase` | Structural phase, such as amorphous or crystalline |
| `total_cost` | Current budget spent |
| `cost_budget` | Maximum budget for the episode |
| `step_number` | Current step count |
| `score_breakdown` | Structural stability and order metrics |

### Action Space

| Action | Description |
|--|--|
| `place` | Place an atom on an empty cell |
| `replace` | Replace one atom species with another |
| `remove` | Remove an atom from the grid |

### Atom Types

| Symbol | Role | Primary Strength | Cost |
|--|--|--|--|
| `A` | Metal | Hardness | 8 |
| `B` | Conductor | Conductivity | 6 |
| `C` | Ceramic | Thermal resistance | 4 |
| `P` | Polymer | Elasticity | 2 |

## Reward Design

The environment uses a dense scientific reward made of four parts:

1. **Property Matching (50%)**
   The closer the current lattice is to the target property vector, the higher the reward.
2. **Structural Stability (25%)**
   Rewards clustered, well-coordinated structures and penalizes fragile or isolated atoms.
3. **Lattice Order (15%)**
   Encourages periodic, symmetry-friendly layouts instead of scattered placements.
4. **Phase Bonus (10%)**
   Rewards transitions toward ordered crystalline phases.

All rewards are normalized to the `[0, 1]` range.

## Why Judges Should Care

MaterialForge is designed to score well on the hackathon dimensions that matter most:

- **Environment Innovation**
  It places RL training into a scientific design workflow instead of a toy game.
- **Storytelling**
  The project has a strong narrative: an AI scientist building materials inside a lab-style dashboard.
- **Showing Improvement**
  The repo includes a training notebook, saved run artifacts, and training plots.
- **Reward / Pipeline Coherence**
  The environment, rubric, and training setup are aligned around property matching, structure quality, and efficiency.

## Architecture

MaterialForge follows a decoupled architecture:

```text
Agent / Training Notebook
    -> HTTP reset / step calls or TRL tool-calling wrapper
FastAPI Server (server/app.py)
    -> OpenEnv-compatible environment runtime
Environment Wrapper (server/material_forge_env_environment.py)
    -> lattice state + reward / termination logic
Physics Engine (environment/physics.py)
    -> property estimates, phase classification, structural scoring
Dashboard (server/static/)
    -> interactive lab visualization for judges and demos
```

Key implementation files:

- `server/material_forge_env_environment.py`
- `environment/physics.py`
- `environment/lattice.py`
- `environment/rubrics.py`
- `training/MaterialForge_GRPO_Training.ipynb`
- `inference.py`

## Training Approach

We used a GRPO-based notebook pipeline built with:

- **OpenEnv** for environment structure
- **TRL** for RL post-training
- **Unsloth** for efficient 4-bit QLoRA finetuning
- **Qwen** family models for tool-using policy training

For the **Grand Finale Submission**, we scaled the training to **Qwen2.5-7B-Instruct** using a high-memory server with **142GB VRAM**, enabling **8 generations per prompt** and real-time gradient updates (Accumulation=1) for maximum stability.

## Training Evidence

While we provide historical artifacts from **Run - V** (0.6B model) as a baseline for rapid iteration, the final submission policy is derived from the **Grand Finale** run, which achieved a **peak episodic reward of 0.86+**.

### Grand Finale Run (Final Submission)
- **Base Model:** `Qwen2.5-7B-Instruct`
- **Generations per Prompt:** `8` (leveraging high-memory hardware for stable advantage estimation)
- **Max Completion Length:** `512` (allowing for deep scientific reasoning)
- **Spatial Reward Bonus:** `0.15` (prioritizing 2D crystalline order)
- **Robustness:** Integrated **Anti-Collapse Reward Shaping** and **Robust Parsing** to prevent syntax-drift failures.
- **Hardware:** High-memory server (**142GB VRAM**)

### Historical Baseline (Run - V)
- **Base Model:** `Qwen/Qwen3-0.6B`
- **Random Baseline Mean Best Reward:** `0.6137`

### Key Innovation: Robust Scientific Alignment
To ensure the 7B model didn't fall into "Safe Mode Collapse" (doing nothing to avoid penalties), we implemented:
1. **Curiosity Bonuses**: Rewards for each unique valid tool call.
2. **Occupancy Incentives**: Penalizing empty grids to force the model to start the design process.
3. **Resilient Parsing**: A regex-based parser that handles hallucinated `<tool_call>` tags, a common failure mode for tool-using models.

### Reward Curve

![Run V Reward Curve](training/runs/Run%20-%20V/reward_curve.png)

### Loss Curve

![Run V Loss Curve](training/runs/Run%20-%20V/loss_curve.png)

### Baseline Comparison

![Run V Baseline Comparison](training/runs/Run%20-%20V/baseline_comparison.png)

These artifacts are also referenced in the project blog:

- [BLOG.md](BLOG.md)

## Submission Materials

This repo now contains the core materials judges need:

- OpenEnv-compatible environment
- Hugging Face Space deployment target
- training notebook using TRL + Unsloth
- saved historical training artifacts
- project blog / write-up
- research and strategy notes

Recommended review order for judges:

1. Read this README
2. Open the live Space / dashboard
3. Review [BLOG.md](BLOG.md)
4. Inspect [training/MaterialForge_GRPO_Training.ipynb](training/MaterialForge_GRPO_Training.ipynb)
5. Inspect `training/runs/Run - V/` for plots

## Quickstart

### Docker

```bash
docker build -t material-forge .
docker run -p 7860:7860 material-forge
```

### Local Development

```bash
uv sync
uv run server
```

### Run the Inference Agent

```bash
set HF_TOKEN=your_token
uv run python inference.py
```

## Project Structure

```text
MaterialForge/
├── docs/                         # Hackathon guidance and external reference notes
├── environment/                  # Physics, lattice state, reward config
├── scenarios/                    # Target scenario generation
├── server/                       # FastAPI server + dashboard assets
├── tests/                        # Benchmarking helpers
├── training/                     # GRPO notebook + archived run artifacts
├── BLOG.md                       # Judge-facing training write-up
├── README.md                     # Project overview
├── RESEARCH.md                   # Internal strategy and theme alignment
├── inference.py                  # Baseline inference agent
├── client.py                     # Environment client
├── models.py                     # Pydantic action / observation models
├── openenv.yaml                  # OpenEnv task manifest
└── Dockerfile                    # Deployment image
```

## Final Summary

MaterialForge is our attempt to show that RL environments for LLMs can move beyond toy games into structured scientific workflows. It combines:

- a novel environment
- verifiable rewards
- long-horizon action sequences
- a live demo surface
- and a practical RL training pipeline

The project is not just about placing atoms on a grid. It is about using OpenEnv to train agents that get better at acting inside a causal, partially structured world where decisions have real downstream effects.

<div align="center">
  <p>Built for the <b>Meta PyTorch OpenEnv Hackathon</b> by Arsh Pathan</p>
</div>
