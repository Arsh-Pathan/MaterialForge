---
title: MaterialForge
emoji: 🔬
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - materials-science
  - crystal-structure
---

<div align="center">
  <h1>🧬 MaterialForge</h1>
  <p><b>Advanced Reinforcement Learning Environment for Atomic Lattice Synthesis</b></p>

[![Framework: OpenEnv](https://img.shields.io/badge/Framework-OpenEnv-blueviolet?style=for-the-badge)](https://github.com/meta-pytorch/openenv)
[![Performance: 0.831](https://img.shields.io/badge/Performance-0.831_Avg_Score-emerald?style=for-the-badge)](#)
[![Status: Platinum](https://img.shields.io/badge/Status-Evaluation_Ready-orange?style=for-the-badge)](#)

</div>

---

## 🔬 Project Overview

**MaterialForge** is a high-fidelity reinforcement learning sandbox designed for the autonomous discovery and optimization of crystalline structures. Built on the **OpenEnv** framework, it challenges agents to manipulate an 8x8 atomic lattice to achieve targeted material properties through precise structural engineering.

By simulating real-world physics heuristics—such as percolation thresholds for conductivity and coordinate bonding for structural stability—MaterialForge provides a robust environment for evaluating the decision-making capabilities of both traditional and LLM-based agents.

---

## 🏗️ Design & Architecture

MaterialForge implements a strictly decoupled architecture, ensuring that the physical simulation logic is separated from the server interface and agent interactive loops.

### System Flow
```mermaid
graph TD
    subgraph "Intelligent Agent"
        Agent[LLM / Heuristic Controller]
    end

    subgraph "MaterialForge Core"
        API[FastAPI / WebSocket Interface]
        Engine[Lattice Management Engine]
        Physics[Physical Heuristic Engine]
        Rubric[Reward Rubric Module]
    end

    subgraph "Environmental State"
        State[(8x8 Tensor Grid)]
        Props[Property Alignment Vector]
    end

    Agent -->|MaterialForgeAction| API
    API --> Engine
    Engine -->|State Update| Physics
    Physics -->|Scientific Feedback| Rubric
    Rubric -->|Reward Signal| API
    Engine --> State
    Physics --> Props
```

---

## 🧪 Scientific Framework

The environment evaluates crystalline feasibility across three primary physical pillars:

### 1. Atom Species Catalog
| Species | Designation | Physical Properties |
| :--- | :--- | :--- |
| **A** | Transition Metal | High hardness, low thermal resistance. |
| **B** | Conductive Agent | Essential for percolation pathway formation. |
| **C** | Structural Ceramic | Excellent thermal shielding, highly stable. |
| **P** | Organic Polymer | Lightweight, high elasticity, budget-efficient. |

### 2. Physical Heuristics
*   **Percolation Conductivity**: Identifies continuous pathways of Species B across the grid using BFS cluster analysis. Spanning pathways yield a +4.0x property multiplier.
*   **Gibbs Stability**: Stability is awarded based on local coordination numbers (neighbor density) and mirror-plane symmetry.
*   **Lattice Entropy**: Measures the positional order of atoms. High symmetry structures yield a lower entropy and a higher Order Index.

### 3. Reward Function $R$
Rewards are calculated as a weighted sum of property matching and structural integrity, tempered by a **Quadratic Cost Penalty** to enforce material efficiency:
$$R = (w_{match} \cdot \text{Score}) - (Cost - Budget)^2$$

---

## 📊 Benchmarking & Performance

We conducted a large-scale evaluation of the MaterialForge agent across **100 randomized crystalline seeds**.

| Evaluation Metric | Baseline (Heuristic) | Augmented (LLM) | Rating |
| :--- | :--- | :--- | :--- |
| **Mean Reward** | **0.831** | 0.804 | 🌟 EXCELLENT |
| **Success Rate** | 100% | 100% | ✅ PASS |
| **Max Score** | 0.876 | 0.864 | 🏆 PLATINUM |
| **Efficiency** | 18.4 steps | 21.2 steps | ⚡ OPTIMAL |

---

## 🚀 Installation & Usage

### 🔬 [Discovery Lab Interactive UI](https://huggingface.co/spaces/ArshPathan/material_forge_env)

### Local Dev Setup
```bash
# Initialize
git clone https://github.com/Arsh-Pathan/MaterialForge.git
uv sync

# Run the 100-trial analytical suite
python benchmark.py --trials 100 --mode llm
```

---

<div align="center">
  <p>Built for the <b>Meta PyTorch OpenEnv Hackathon</b> by Arsh Pathan</p>
</div>
