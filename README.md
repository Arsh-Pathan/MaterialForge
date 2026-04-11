---
title: MaterialForge
emoji: 🔬
colorFrom: red
colorTo: blue
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - materials-science
  - crystal-structure
---

<div align="center">
  <img src="server/static/hero.png" width="100%" alt="MaterialForge Hero">
  <h1>🔬 MaterialForge</h1>
  <p><b>An Advanced Reinforcement Learning Sandbox for Atomic Crystal Engineering</b></p>

[![Framework: OpenEnv](https://img.shields.io/badge/Framework-OpenEnv-blueviolet?style=for-the-badge)](https://github.com/meta-pytorch/openenv)
[![License: BSD](https://img.shields.io/badge/License-BSD-green?style=for-the-badge)](LICENSE)
[![Status: Production](https://img.shields.io/badge/Status-Production-orange?style=for-the-badge)](#)

</div>

---

## 🏛️ Executive Summary

**MaterialForge** is a high-fidelity environment designed for the discovery of optimal crystal structures. It simulates the structure-property relationship of materials using advanced physics heuristics, challenging AI agents to evolve a lattice grid from an amorphous state into highly ordered crystalline structures that meet target thermal and electrical specifications.

---

## 🏗️ System Architecture

MaterialForge follows a decoupled architectural pattern, ensuring that pure-physics logic is separated from the interactive server and the RL-loop orchestration.

```mermaid
graph TD
    subgraph "Agent Workspace"
        Agent[LLM Agent]
    end

    subgraph "MaterialForge Environment"
        API[FastAPI / WebSocket Interface]
        Engine[Lattice Management Engine]
        Physics[Advanced Physics Heuristics]
        Rubric[Multi-Objective Reward System]
    end

    subgraph "Observations & State"
        State[(8x8 Atomic Grid)]
        Props[Property Vector]
        Phase[Phase Classification]
    end

    Agent -->|MaterialForgeAction| API
    API --> Engine
    Engine -->|Compute| Physics
    Physics -->|Properties| Rubric
    Rubric -->|Reward + Observation| API
    API -->|Feedback| Agent
    Engine --> State
    Physics --> Props
    Physics --> Phase
```

---

## 🧪 Scientific Foundations

The environment implements three primary physical models to simulate material behavior at a heuristic level.

### 1. Percolation & Conductivity

Conductivity is determined by the **Percolation Threshold**. The engine identifies connected pathways of Conductive Atoms (Species B) across the lattice.

```mermaid
flowchart LR
    Start([Action Applied]) --> Search[BFS Pathway Search]
    Search --> Connectivity{Continuous Path?}
    Connectivity -- "Span X-Axis" --> MultB[+4.0 Multiplier]
    Connectivity -- "Span Y-Axis" --> MultC[+2.5 Multiplier]
    Connectivity -- "No Path" --> Low[Base Value Only]
    MultB --> Result[Conductivity Property Result]
    MultC --> Result
    Low --> Result
```

### 2. Structural Stability (Gibbs Approach)

Stability is derived from local coordination numbers and mirror-plane symmetry.

- **Coordination bonding**: Rewards atoms with a higher local neighbor density.
- **Mirror Symmetry**: Symmetry across central axes stabilizes the lattice against thermal stress.

### 3. Lattice Order (Entropy)

Measures the **Positional Entropy** of atoms. Highly ordered crystalline structures (Symmetric and Homogeneous) yield the highest "Lattice Order Index".

---

## 📊 Interaction Model

### Action Space

Agents interact via discrete operations on the 8x8 lattice:

| Action    | Description                                     | Scientific Intent                   |
| :-------- | :---------------------------------------------- | :---------------------------------- |
| `place`   | Inserts an atom into an empty cell.             | Material Growth                     |
| `replace` | Swaps an existing atom for a different species. | Lattice Refinement                  |
| `remove`  | Clears a cell.                                  | Defect Management / Budget Recovery |

### Observation Space

The environment returns a rich state-vector containing:

- **Grid Snapshot**: Full 2D array representation.
- **Property Vector**: Current [Hardness, Conductivity, Thermal, Elasticity].
- **Score Breakdown**: Granular feedback on Stability and Order.

---

## 🏆 Scoring Rubric

The reward signal $R$ is calculated to incentivize scientific accuracy while enforcing material efficiency.

```mermaid
pie title Reward Distribution
    "Property Alignment" : 40
    "Structural Stability" : 30
    "Lattice Order" : 20
    "Phase Bonus" : 10
```

> [!IMPORTANT]
> **Quadratic Cost Pressure**: Every action incurs an atomic cost. If the total cost exceeds the budget, a **quadratic penalty** is applied: $Penalty = (Cost_{total} - Budget)^2$. This forces agents to build efficiently rather than filling the grid blindly.

---

## 🛠️ Developer Integration

### Quick Launch

```bash
# Install dependencies
uv sync

# Start the environment server
uv run server
```

### Interactive Playground

Access the high-fidelity laboratory dashboard at:
`http://localhost:8000/playground`

---

<div align="center">
  <p>Built with ❤️ by <b>Arsh Pathan</b> for the Meta PyTorch OpenEnv Hackathon</p>
  <a href="https://huggingface.co/spaces/ArshPathan/material_forge_env"><b>Launch Production Discovery Lab</b></a>
</div>
