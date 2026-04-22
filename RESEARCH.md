# MaterialForge: Operational Research & Hackathon Strategy
> **Event:** Meta PyTorch OpenEnv Hackathon × Scaler School of Technology (Grand Finale)
> **Dates:** April 25–26, 2026 | Bangalore, India
> **Theme Focus:** World Modeling (#3.1) & Long-Horizon Planning (#2)

---

## 1. Hackathon Context: The Grand Finale
This document serves as the internal reference for the MaterialForge project as it enters the in-person finale at Scaler School of Technology. The project is designed to showcase the power of **OpenEnv** in simulating complex, professional workflows—specifically **Inverse Molecular Design**.

### Core Problem Statement
How can an AI agent autonomously discover stable, high-performance crystalline structures given a set of target physical properties and a finite resource budget?

---

## 2. Strategic Theme Alignment

### Theme #3.1: World Modeling (Professional Tasks)
MaterialForge is a high-fidelity simulation of a scientific workflow.
- **Nuanced Interaction:** The agent does not just "fill a grid"; it interacts with a physics engine (`environment/physics.py`) that models percolation pathways, coordination chemistry, and phase transitions.
- **Causal Reasoning:** The reward signal encourages agents to understand that stability (coordination number) and order (crystallinity) are prerequisites for achieving high-tier material properties.

### Theme #2: (Super) Long-Horizon Planning
- **Multi-step Reasoning:** Designing a "Diamond-like" or "Superconductor Analogue" requires 50-80 coordinated steps.
- **Sparse Reward Recovery:** While the reward is dense, the ultimate goal (achieving a 'Crystalline' phase) is a delayed milestone that requires consistent, structured planning.
- **Budget Management:** Agents must track state over extended trajectories to ensure they don't exhaust the `cost_budget` before reaching the target symmetry.

---

## 3. Project Analysis & Architecture

### Repository Structure
| Directory | Role | Key Files |
|-----------|------|-----------|
| `environment/` | **Physics Engine** | `physics.py` (heuristics), `lattice.py` (state), `rubrics.py` (rewards) |
| `server/` | **Infrastructure** | `app.py` (FastAPI), `material_forge_env_environment.py` (OpenEnv wrapper) |
| `scenarios/` | **Task Logic** | `scenarios.py` (Target property generation) |
| `scripts/` | **Automation** | `validate-submission.sh` (Compliance check) |
| `tests/` | **Validation** | `benchmark.py` (Baseline tracking) |

### Core Logic Breakdown
1. **Lattice Engine:** An 8x8 grid where actions (`place`, `replace`, `remove`) mutate the atomic ensemble.
2. **Physics Heuristics:** Calculates 4 properties (Hardness, Conductivity, Thermal Resistance, Elasticity) based on atom fractions, clustering bonuses, and BFS-based percolation checks.
3. **Phase Classification:** Categorizes the structure as *Amorphous*, *Polycrystalline*, or *Crystalline* based on 2x2 repeating pattern density and symmetry.
4. **Reward Model:** A composite score that weights property accuracy (50%) against structural integrity (stability + order + phase) (50%), minus efficiency penalties.

---

## 4. Execution Flow
1. **Initialization:** The environment server starts via `uvicorn`.
2. **Reset:** The agent calls `/reset`, generating a `target` property vector (e.g., Hardness: 90, Conductivity: 10).
3. **Inference Loop:**
   - Agent observes the `grid` and `current_properties`.
   - Agent simulates potential moves using a local copy of the physics engine (Lookahead).
   - Agent selects an action to minimize the "property gap" while maximizing "stability".
4. **Step:** Action is sent to the server; the server returns the updated state and a dense reward.
5. **Termination:** Episode ends when targets are met within `tolerance` and the phase is `Crystalline`, or `max_steps` is reached.

---

## 5. Submission Requirements & Judging Criteria

### Minimum Requirements Checklist
- [x] **OpenEnv Usage:** Fully compliant with `Environment` and `EnvClient` base classes.
- [ ] **Training Script:** (Planned) Implementation of a GRPO training loop using **HF TRL** or **Unsloth**.
- [ ] **Evidence of Training:** (Planned) Reward curves and before/after behavior comparison.
- [x] **Hosting:** Environment is ready for deployment to **Hugging Face Spaces**.
- [ ] **Mini-Blog/Video:** (To be created) 2-minute walkthrough of the "Discovery Lab".

### Judging Matrix (Targeting High Scores)
1. **Environment Innovation (40%):** MaterialForge introduces a novel scientific domain to RL, moving beyond simple games into professional discovery workflows.
2. **Storytelling (30%):** The "Discovery Lab" dashboard provides an engaging narrative of an AI "scientist" in a lab.
3. **Showing Improvement (20%):** We will demonstrate that a trained agent discovers "Crystalline" phases significantly faster than the heuristic baseline.
4. **Pipeline Coherence (10%):** Our reward logic is grounded in physical principles (Gibbs stability, Bragg order).

---

## 6. Development Roadmap (Bangalore Finale)
1. **Phase 1: Self-Improvement Strategy.** Implement a training pipeline using **Unsloth** to fine-tune a model on successful discovery trajectories.
2. **Phase 2: Multi-Agent Extension.** (Optional) Introduce a "Critic" agent that suggests structural modifications to the "Worker" agent.
3. **Phase 3: Final Validation.** Run the `validate-submission.sh` script to ensure 100% compliance before the 3-minute pitch.

---
*Document updated for Finale Preparation: April 22, 2026*
