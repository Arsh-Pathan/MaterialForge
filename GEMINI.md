# MaterialForge - AI-Driven Crystal Design Environment

MaterialForge is a high-fidelity reinforcement learning environment for the autonomous discovery and optimization of crystalline structures. It is part of the **Meta PyTorch OpenEnv Hackathon**.

## Project Overview

- **Core Objective:** Design 8x8 atomic lattices to match specific physical properties (Hardness, Conductivity, Thermal Resistance, Elasticity).
- **Architecture:** 
    - **Backend:** FastAPI server (`server/app.py`) providing an OpenEnv-compatible REST API.
    - **Physics Engine:** Heuristic-based simulation of material properties and phase classification (`environment/`).
    - **Frontend:** Interactive visualization dashboard (`server/static/`).
    - **Agent:** Heuristic-guided LLM agent for solving discovery tasks (`inference.py`).
- **Technologies:** Python 3.11+, FastAPI, `openenv-core`, Docker, `uv` (dependency manager).

## Key Components

- `environment/`: Core logic including lattice management (`lattice.py`), physics heuristics (`physics.py`), and reward rubrics (`rubrics.py`).
- `server/`: FastAPI application (`app.py`) and static assets for the "Discovery Lab" dashboard.
- `models.py`: Pydantic models for OpenEnv API compliance (Actions and Observations).
- `inference.py`: Baseline agent implementation using a combination of local lookahead heuristics and LLM-based decision making.
- `openenv.yaml`: Environment specification and task definitions for the OpenEnv benchmark.

## Building and Running

### Prerequisites
- [uv](https://github.com/astral-sh/uv) for dependency management.
- Docker (optional, for containerized execution).

### Commands

| Task | Command |
|------|---------|
| **Install Dependencies** | `uv sync` |
| **Start Environment Server** | `uv run server` (or `python -m server.app`) |
| **Run Inference Agent** | `uv run python inference.py` (Requires `HF_TOKEN`) |
| **Run Unit Tests** | `uv run pytest` |
| **Validate Submission** | `./scripts/validate-submission.sh <HF_SPACE_URL>` |
| **Docker Build** | `docker build -t material-forge .` |
| **Docker Run** | `docker run -p 7860:7860 material-forge` |

## Development Conventions

- **OpenEnv Compliance:** All environment interactions must adhere to the OpenEnv API standards (Actions/Observations).
- **Machine-Readable Logs:** `inference.py` emits standardized `[START]`, `[STEP]`, and `[END]` tokens for evaluation tracking.
- **Physics-First Design:** The environment rewards structural stability and crystalline order in addition to property matching.
- **Code Style:** Follows standard Python type-hinting and Pydantic for data validation.

## Environment Specification

- **Action Space:** Discrete actions (`place`, `remove`, `replace`) on an 8x8 grid.
- **Observation Space:** 8x8 grid state, current/target properties, cost budget, and lattice phase.
- **Atom Species:**
    - **A (Metal):** High Hardness, high cost.
    - **B (Conductor):** High Conductivity.
    - **C (Ceramic):** High Thermal Resistance.
    - **P (Polymer):** High Elasticity, low cost.
