# MaterialForge — Implementation Plan

> AI-Driven Atomic Structure Discovery Environment for the Meta PyTorch OpenEnv Hackathon

---

## 1. Project Summary

MaterialForge is an OpenEnv-compliant RL environment where an AI agent designs atomic crystal structures to match target material properties (hardness, conductivity, thermal resistance, elasticity). The agent places/replaces/removes atoms on a lattice grid, and the environment scores each configuration against the target using physics-inspired heuristics + optional LLM judging.

**Deadline:** April 8, 2026 (Round 1 submission)

---

## 2. Architecture Overview

```
material_forge_env/
├── __init__.py                          # Public exports
├── models.py                            # Action, Observation, State Pydantic models
├── client.py                            # EnvClient (WebSocket client)
├── config.py                            # Constants: atom types, grid size, property ranges
├── lattice.py                           # Core lattice data structure + manipulation
├── physics.py                           # Property estimation from atomic structure
├── scenarios.py                         # Target profile generators (easy/medium/hard)
├── rubrics.py                           # Reward rubrics (heuristic + optional LLM)
├── llm_utils.py                         # Shared LLM client factory (graceful degradation)
├── hint_engine.py                       # Optional LLM hint system for struggling agents
├── openenv.yaml
├── pyproject.toml
├── README.md
└── server/
    ├── __init__.py
    ├── app.py                           # FastAPI + create_app
    ├── material_forge_env_environment.py # Environment implementation
    ├── Dockerfile
    └── requirements.txt
```

---

## 3. Core Data Models

### 3.1 Atom Types & Properties

| Atom | Symbol | Description         | Cost | Hardness | Conductivity | Thermal Res | Elasticity |
|------|--------|---------------------|------|----------|--------------|-------------|------------|
| A    | `A`    | Strong metal        | 8    | High     | Medium       | High        | Low        |
| B    | `B`    | Conductive element  | 6    | Low      | High         | Low         | Medium     |
| C    | `C`    | Insulating ceramic  | 4    | Medium   | Low          | High        | Low        |
| P    | `P`    | Polymer-like        | 2    | Low      | Low          | Medium      | High       |
| `.`  | empty  | Vacant site         | 0    | —        | —            | —           | —          |

### 3.2 Action Model (`MaterialForgeAction`)

```python
class MaterialForgeAction(Action):
    action_type: Literal["place", "replace", "remove"]
    row: int          # 0-indexed grid position
    col: int          # 0-indexed grid position
    atom: Optional[Literal["A", "B", "C", "P"]] = None  # required for place/replace
```

### 3.3 Observation Model (`MaterialForgeObservation`)

```python
class MaterialForgeObservation(Observation):
    grid: List[List[str]]              # Current lattice state (e.g. 8x8)
    target: Dict[str, float]           # Target properties {hardness: 90, ...}
    current_properties: Dict[str, float]  # Estimated properties of current structure
    phase: str                         # "crystalline" | "polycrystalline" | "amorphous"
    total_cost: float                  # Sum of atom costs
    cost_budget: float                 # Max allowed cost
    step_number: int                   # Current step in episode
    max_steps: int                     # Episode length limit
    score_breakdown: Dict[str, float]  # Per-component reward breakdown
    hint: Optional[str] = None         # LLM hint (when available)
    # Inherited: done, reward, metadata
```

### 3.4 State

Uses the default OpenEnv `State` with `episode_id` and `step_count`.

---

## 4. Core Modules

### 4.1 `config.py` — Constants & Configuration

- `GRID_SIZE = 8` (8×8 lattice)
- `MAX_STEPS = 50` per episode
- `ATOM_TYPES` dict with per-atom property contributions and costs
- `PROPERTY_NAMES = ["hardness", "conductivity", "thermal_resistance", "elasticity"]`
- `COST_BUDGET_DEFAULT = 80`
- Difficulty presets: easy (wide tolerance), medium, hard (tight tolerance)

### 4.2 `lattice.py` — Lattice Engine

Core data structure representing the material unit cell.

**Class `Lattice`:**
- `__init__(size: int)` — create empty grid
- `place(row, col, atom)` → bool (validates bounds + empty)
- `replace(row, col, atom)` → bool (validates bounds + occupied)
- `remove(row, col)` → bool (validates bounds + occupied)
- `get_grid() → List[List[str]]`
- `count_atoms() → Dict[str, int]`
- `total_cost() → float`
- `get_neighbors(row, col) → List[str]` — 8-connected neighbors
- `clone() → Lattice` — deep copy for rollback
- `from_grid(grid: List[List[str]]) → Lattice` — deserialize

### 4.3 `physics.py` — Property Estimation

Heuristic-based physics engine. No real MD simulation, but plausible.

**Functions:**
- `estimate_properties(lattice: Lattice) → Dict[str, float]`
  - **Hardness:** f(metal density, bonding density, packing ratio, crystal order)
  - **Conductivity:** f(conductive atom fraction, connectivity paths, percolation)
  - **Thermal Resistance:** f(insulator fraction, structural order, density)
  - **Elasticity:** f(polymer fraction, amorphous ratio, void fraction)
- `classify_phase(lattice: Lattice) → str`
  - Checks for repeating patterns → "crystalline"
  - Checks for regional order → "polycrystalline"
  - Default → "amorphous"
- `compute_stability(lattice: Lattice) → float`
  - Penalizes isolated atoms (no neighbors)
  - Rewards dense bonding clusters
  - Checks structural symmetry

**Property calculation strategy:**
Each property is a weighted sum of atomic contributions + structural bonuses:
```
property = base_contribution(atom_counts) + neighbor_bonus + phase_bonus + symmetry_bonus
```
All properties normalized to 0–100 scale.

### 4.4 `scenarios.py` — Target Profile Generator

**Functions:**
- `generate_scenario(difficulty: str) → Dict`
  - Returns: `{target: {hardness, conductivity, ...}, cost_budget, tolerance, max_steps}`
  - `"easy"`: large tolerance (±20), generous budget, fewer properties to match
  - `"medium"`: moderate tolerance (±10), standard budget
  - `"hard"`: tight tolerance (±5), tight budget, all properties must match
- `generate_llm_scenario()` — optional LLM-generated creative scenario with narrative
- Predefined classic scenarios: "diamond-like", "conductor", "heat-shield", "flexible-polymer"

### 4.5 `rubrics.py` — Reward System

**`HeuristicRewardRubric(Rubric)`:**
```python
def forward(action, observation) -> float:
    property_match = 1 - mean(|current[p] - target[p]| / 100 for p in properties)
    stability = compute_stability(lattice)
    phase_bonus = phase_alignment_score(current_phase, target_phase)
    cost_penalty = max(0, (total_cost - budget) / budget)

    reward = (0.50 * property_match
            + 0.25 * stability
            + 0.15 * lattice_quality
            + 0.10 * phase_bonus
            - cost_penalty)
    return clamp(reward, 0.0, 1.0)
```

**`MaterialDesignLLMJudge(LLMJudge)`:** (optional, 30% weight when available)
- Prompt: "Given target {target} and structure {grid}, rate quality 0-10"
- Uses HF Inference API via `HF_TOKEN`
- Falls back to heuristic-only if unavailable

**Composite Rubric (via `WeightedSum`):**
- With LLM: `WeightedSum([HeuristicRubric, LLMJudge], [0.7, 0.3])`
- Without LLM: `HeuristicRubric` only

### 4.6 `llm_utils.py` — LLM Client

- Reads `HF_TOKEN` from environment
- Creates `OpenAIClient` pointing to HF Inference API
- Returns `None` gracefully when no token available
- All LLM features degrade gracefully (heuristic-only mode)

### 4.7 `hint_engine.py` — Agent Hints

- Rate-limited: max 3 hints per episode
- Only triggers when agent is struggling (reward declining for 5+ steps)
- Provides textual guidance: "Try adding more metal atoms in the center for hardness"
- Returned in `observation.hint` field

---

## 5. Environment Logic (`MaterialForgeEnvironment`)

### 5.1 `reset()`
1. Generate scenario (random difficulty or specified)
2. Initialize empty 8×8 lattice
3. Reset rubric state via `self._reset_rubric()`
4. Return initial observation with empty grid + target properties

### 5.2 `step(action)`
1. Validate action (bounds, atom type, action_type logic)
2. Apply action to lattice (place/replace/remove)
3. Estimate current properties via `physics.estimate_properties()`
4. Classify structural phase via `physics.classify_phase()`
5. Compute reward via `self._apply_rubric(action, observation)`
6. Check termination: `done = (step >= max_steps) or (all properties within tolerance)`
7. Optionally generate hint if agent is struggling
8. Return observation

### 5.3 Termination Conditions
- **Success:** All target properties within tolerance → bonus reward
- **Budget exceeded:** Cost > budget → penalty, episode continues (agent can remove atoms)
- **Max steps:** Episode ends, final reward reflects best configuration
- **Perfect match:** Early termination with maximum reward

---

## 6. Client (`MaterialForgeEnv`)

Update the existing scaffold client to:
- Serialize `MaterialForgeAction` properly
- Parse `MaterialForgeObservation` with grid, properties, phase, etc.
- Handle the new observation fields in `_parse_result`

---

## 7. Deployment

### 7.1 Local Development
```bash
cd material_forge_env
uv sync
uv run server  # http://localhost:8000
```

### 7.2 Docker
```bash
cd material_forge_env/server
docker build -t material-forge-env:latest .
docker run -p 8000:8000 material-forge-env:latest
```

### 7.3 Hugging Face Spaces
```bash
cd material_forge_env
openenv push  # Deploys to HF Spaces with Gradio UI
```

---

## 8. Implementation Order

### Phase 1: Core Engine (Priority)
1. **`config.py`** — Define atom types, grid constants, property definitions
2. **`lattice.py`** — Grid data structure with place/replace/remove operations
3. **`physics.py`** — Property estimation heuristics + phase classification
4. **`models.py`** — Rewrite Action/Observation with full MaterialForge fields
5. **`scenarios.py`** — Target profile generator with difficulty levels

### Phase 2: Environment Integration
6. **`rubrics.py`** — HeuristicRewardRubric
7. **`material_forge_env_environment.py`** — Wire up reset/step with lattice + physics + rubric
8. **`client.py`** — Update client serialization/parsing
9. **`server/app.py`** — Update create_app with rubric

### Phase 3: LLM Features (Optional Enhancement)
10. **`llm_utils.py`** — HF Inference API client
11. **`rubrics.py`** — Add LLMJudge + WeightedSum composite
12. **`hint_engine.py`** — Contextual hint generation
13. **`scenarios.py`** — LLM-generated creative scenarios

### Phase 4: Polish & Submission
14. **`README.md`** — Full documentation with examples
15. **Demo script** — `demo.py` showing agent interaction loop
16. **Tests** — Validate environment correctness
17. **Docker build** — Verify containerized deployment
18. **HF Spaces** — Deploy and get demo link

---

## 9. Evaluation Criteria Alignment

| Criteria | How We Address It |
|----------|-------------------|
| **Correctness** | Physics heuristics produce plausible material properties; validated with known material archetypes |
| **OpenEnv Compliance** | Standard Environment base class, Action/Observation models, create_app, rubric system |
| **Task Design Quality** | Rich action space (place/replace/remove × 4 atoms × 64 positions), meaningful search problem |
| **Reward Logic** | Multi-component reward: property match + stability + lattice quality + phase alignment - cost |
| **Code Quality** | Modular design, type hints, docstrings, separated concerns |

---

## 10. Key Design Decisions

1. **2D grid (not 3D)** — Keeps complexity manageable for Round 1; 3D is a Phase 2 extension
2. **Heuristic physics** — No real DFT/MD simulation; plausible approximations that create a meaningful search landscape
3. **4 atom types** — Enough variety for interesting compositions without overwhelming the action space
4. **8×8 grid** — 64 positions × 5 options (4 atoms + empty) = rich enough search space
5. **50 max steps** — Allows iterative refinement without infinite episodes
6. **LLM features optional** — Everything works without HF_TOKEN; LLM adds richness when available
7. **WeightedSum rubric** — Clean separation of heuristic (70%) and LLM (30%) scoring

---

## 11. Submission Checklist

- [ ] Public GitHub repository
- [ ] Complete `README.md` with setup instructions
- [ ] `requirements.txt` in server/
- [ ] `pyproject.toml` with all dependencies
- [ ] Demo script (`demo.py`)
- [ ] Working Docker build
- [ ] Hugging Face Spaces demo link
- [ ] OpenEnv validation passes (`openenv validate`)
