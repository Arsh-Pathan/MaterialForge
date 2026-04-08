"""
Inference Script - MaterialForge
================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL    The LiteLLM proxy endpoint injected by the validator.
    API_KEY         The LiteLLM proxy API key injected by the validator.
    MODEL_NAME      The model identifier to use for inference.
    LOCAL_IMAGE_NAME The name of the local Docker image for the environment
                     (used by from_docker_image()).

- The inference script must be named `inference.py` and placed in the root directory of the project.
- Participants must use the OpenAI client for all LLM calls with:
    base_url=os.environ["API_BASE_URL"]
    api_key=os.environ["API_KEY"]

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import asyncio
import json
import os
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional

from openai import OpenAI

try:
    from material_forge_env import MaterialForgeAction, MaterialForgeEnv
except ImportError:
    from client import MaterialForgeEnv
    from models import MaterialForgeAction

try:
    from material_forge_env.environment.config import ATOM_SYMBOLS, ATOM_TYPES, EMPTY, PROPERTY_NAMES
    from material_forge_env.environment.lattice import Lattice
    from material_forge_env.environment.physics import classify_phase, estimate_properties
except ImportError:
    from environment.config import ATOM_SYMBOLS, ATOM_TYPES, EMPTY, PROPERTY_NAMES
    from environment.lattice import Lattice
    from environment.physics import classify_phase, estimate_properties


MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "material-forge-env:latest")
SPACE_URL = os.getenv("SPACE_URL")
BENCHMARK = "material_forge_env"

TASKS = [
    {"name": "diamond-like", "difficulty": "medium", "seed": 42},
    {"name": "conductor", "difficulty": "medium", "seed": 43},
    {"name": "heat-shield", "difficulty": "medium", "seed": 44},
]

TEMPERATURE = 0.1
MAX_TOKENS = 160
SUCCESS_THRESHOLD = 0.3
PROXY_PROBE_MAX_TOKENS = 8
MIN_TASK_SCORE = 0.001
MAX_TASK_SCORE = 0.999

PROPERTY_TO_ATOM = {
    "hardness": "A",
    "conductivity": "B",
    "thermal_resistance": "C",
    "elasticity": "P",
}

ATOM_TO_PRIMARY_PROPERTY = {
    "A": "hardness",
    "B": "conductivity",
    "C": "thermal_resistance",
    "P": "elasticity",
}


@dataclass(frozen=True)
class ActionCandidate:
    """A simple, human-readable candidate action for the LLM baseline."""

    action: MaterialForgeAction
    summary: str
    state_key: str
    gap_score: float
    budget_after: float
    phase: str


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def normalize_task_score(raw_score: float) -> float:
    """Clamp task scores into the validator-required open interval (0, 1)."""
    return min(max(raw_score, MIN_TASK_SCORE), MAX_TASK_SCORE)


SYSTEM_PROMPT = """You are a baseline agent for a crystal-design OpenEnv benchmark.

Pick exactly one action from the candidate list.
Use these rules:
- First reduce the biggest positive property deficits.
- Prefer place actions that grow a coherent structure.
- Avoid immediately undoing the previous move.
- Use replace or remove only when a property is overshooting or the structure is already large enough.

Return JSON only:
{"candidate_id": 1}"""


def grid_to_key(grid: List[List[str]]) -> str:
    """Create a compact string representation of the lattice state."""
    return "|".join("".join(row) for row in grid)


def action_to_key(action: MaterialForgeAction) -> str:
    """Create a stable identifier for an action."""
    atom = action.atom if action.atom is not None else "-"
    return f"{action.action_type}:{action.row}:{action.col}:{atom}"


def compute_gaps(target: Dict[str, float], current: Dict[str, float]) -> Dict[str, float]:
    """Return signed target-minus-current gaps."""
    return {prop: target[prop] - current.get(prop, 0.0) for prop in PROPERTY_NAMES}


def mean_abs_gap(gaps: Dict[str, float]) -> float:
    """Normalize average absolute gap to a 0-1 scale."""
    return sum(abs(gap) for gap in gaps.values()) / (len(gaps) * 100.0)


def parse_llm_choice(text: str, num_candidates: int) -> Optional[int]:
    """Extract candidate_id from an LLM JSON response."""
    if not text:
        return None

    cleaned = (
        text.strip()
        .replace("Here is the JSON:", "")
        .replace("Here is the JSON", "")
        .replace("Here's the JSON:", "")
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    start = cleaned.find("{")
    end = cleaned.rfind("}") + 1
    if start == -1 or end == 0:
        return None

    try:
        payload = json.loads(cleaned[start:end])
        candidate_id = int(payload["candidate_id"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None

    if 1 <= candidate_id <= num_candidates:
        return candidate_id - 1
    return None


def action_str(action: MaterialForgeAction) -> str:
    """Format action for [STEP] logging."""
    if action.atom is not None:
        return f"{action.action_type}({action.row},{action.col},{action.atom})"
    return f"{action.action_type}({action.row},{action.col})"


def apply_action_to_grid(grid: List[List[str]], action: MaterialForgeAction) -> Optional[List[List[str]]]:
    """Return the next grid for a candidate action, or None if invalid."""
    next_grid = [row[:] for row in grid]
    current_cell = next_grid[action.row][action.col]

    if action.action_type == "place":
        if action.atom is None or current_cell != EMPTY:
            return None
        next_grid[action.row][action.col] = action.atom
        return next_grid

    if action.action_type == "replace":
        if action.atom is None or current_cell == EMPTY or current_cell == action.atom:
            return None
        next_grid[action.row][action.col] = action.atom
        return next_grid

    if action.action_type == "remove":
        if current_cell == EMPTY:
            return None
        next_grid[action.row][action.col] = EMPTY
        return next_grid

    return None


def neighbor_stats(grid: List[List[str]], row: int, col: int, atom: str) -> tuple[int, int]:
    """Return same-type and occupied-neighbor counts around a cell."""
    same_type = 0
    occupied = 0
    size = len(grid)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr = row + dr
            nc = col + dc
            if 0 <= nr < size and 0 <= nc < size:
                cell = grid[nr][nc]
                if cell != EMPTY:
                    occupied += 1
                if cell == atom:
                    same_type += 1
    return same_type, occupied


def sort_empty_cells_for_atom(grid: List[List[str]], atom: str) -> List[tuple[int, int]]:
    """Rank empty cells for placing a given atom."""
    size = len(grid)
    center = (size - 1) / 2.0
    scored: List[tuple[float, int, int]] = []

    for row in range(size):
        for col in range(size):
            if grid[row][col] != EMPTY:
                continue
            same_type, occupied = neighbor_stats(grid, row, col, atom)
            center_distance = abs(row - center) + abs(col - center)
            edge_penalty = 0.2 if row in (0, size - 1) or col in (0, size - 1) else 0.0
            score = same_type * 3.0 + occupied * 1.2 - center_distance * 0.08 - edge_penalty
            scored.append((score, row, col))

    scored.sort(reverse=True)
    return [(row, col) for _, row, col in scored]


def count_atoms(grid: List[List[str]]) -> Dict[str, int]:
    """Count placed atom types in the current grid."""
    counts = {atom: 0 for atom in ATOM_SYMBOLS}
    for row in grid:
        for cell in row:
            if cell in counts:
                counts[cell] += 1
    return counts


def atom_count(grid: List[List[str]]) -> int:
    """Return the number of occupied cells."""
    return sum(1 for row in grid for cell in row if cell != EMPTY)


def evaluate_candidate_state(obs, action: MaterialForgeAction) -> Optional[ActionCandidate]:
    """Evaluate a candidate by the resulting property gap and budget usage."""
    next_grid = apply_action_to_grid(obs.grid, action)
    if next_grid is None:
        return None

    lattice = Lattice.from_grid(next_grid)
    properties = estimate_properties(lattice)
    gaps = compute_gaps(obs.target, properties)
    budget_after = lattice.total_cost()
    phase = classify_phase(lattice)
    primary_gap = gaps.get(ATOM_TO_PRIMARY_PROPERTY.get(action.atom, ""), 0.0) if action.atom else 0.0
    over_budget = max(budget_after - obs.cost_budget, 0.0) / max(obs.cost_budget, 1.0)
    overshoot = sum(max(-gap, 0.0) for gap in gaps.values()) / 100.0
    gap_score = mean_abs_gap(gaps)
    structure_bonus = 0.02 if phase in {"crystalline", "polycrystalline"} else 0.0
    placement_penalty = 0.0
    if action.action_type == "place" and budget_after > obs.cost_budget * 0.95:
        placement_penalty += 0.04
    if action.action_type == "place" and over_budget > 0.0:
        placement_penalty += 0.08
    if action.action_type == "remove" and budget_after < obs.cost_budget * 0.65 and gap_score > 0.12:
        placement_penalty += 0.03

    heuristic_score = (
        -gap_score
        - 0.90 * over_budget
        - 0.25 * overshoot
        - placement_penalty
        + structure_bonus
        + (0.01 if action.action_type == "replace" and primary_gap > 0 else 0.0)
        + (0.01 if action.action_type == "remove" and over_budget > 0 else 0.0)
    )
    atom_name = action.atom if action.atom is not None else "none"
    summary = (
        f"{action.action_type} {atom_name} at ({action.row},{action.col}); "
        f"gap={gap_score:.3f}, budget={budget_after:.0f}/{obs.cost_budget:.0f}, "
        f"phase={phase}, overshoot={overshoot:.3f}"
    )
    return ActionCandidate(
        action=action,
        summary=summary,
        state_key=grid_to_key(next_grid),
        gap_score=round(-heuristic_score, 4),
        budget_after=budget_after,
        phase=phase,
    )


def build_candidates(
    obs,
    recent_action_keys: List[str],
    seen_state_counts: Dict[str, int],
    max_candidates: int = 8,
) -> List[ActionCandidate]:
    """Build and rank a small candidate set without using the reward function."""
    gaps = compute_gaps(obs.target, obs.current_properties)
    occupied_cells = [
        (row, col, obs.grid[row][col])
        for row in range(8)
        for col in range(8)
        if obs.grid[row][col] != EMPTY
    ]
    total_atoms = atom_count(obs.grid)
    remaining_budget = obs.cost_budget - obs.total_cost

    deficit_props = [prop for prop in PROPERTY_NAMES if gaps[prop] > 0]
    deficit_props.sort(key=lambda prop: gaps[prop], reverse=True)
    overshot_props = [prop for prop in PROPERTY_NAMES if gaps[prop] < 0]
    overshot_props.sort(key=lambda prop: gaps[prop])
    max_overshoot = max((abs(gaps[prop]) for prop in overshot_props), default=0.0)
    top_needed_props = set(deficit_props[:2])

    prioritized_atoms = [PROPERTY_TO_ATOM[prop] for prop in deficit_props[:3]]
    if not prioritized_atoms:
        prioritized_atoms = list(ATOM_SYMBOLS)

    candidates: List[ActionCandidate] = []
    used_actions = set()

    def push_candidate(action: MaterialForgeAction) -> None:
        nonlocal candidates
        action_key = action_to_key(action)
        if action_key in used_actions:
            return
        if recent_action_keys and action_key == recent_action_keys[-1]:
            return

        evaluated = evaluate_candidate_state(obs, action)
        if evaluated is None:
            return

        if seen_state_counts.get(evaluated.state_key, 0) >= 1:
            return

        candidates.append(evaluated)
        used_actions.add(action_key)

    place_atoms = prioritized_atoms[:]
    if remaining_budget < 8:
        place_atoms = [atom for atom in place_atoms if ATOM_TYPES[atom]["cost"] <= remaining_budget + 2]
    if not place_atoms:
        place_atoms = [atom for atom in ATOM_SYMBOLS if ATOM_TYPES[atom]["cost"] <= max(remaining_budget + 2, 4)]

    allow_place = obs.total_cost < obs.cost_budget * 0.9 or total_atoms < 6
    allow_replace = total_atoms >= 6 and (
        obs.total_cost >= obs.cost_budget * 0.75 or max_overshoot >= 8.0
    )
    allow_remove = total_atoms >= 8 and (
        obs.total_cost >= obs.cost_budget or max_overshoot >= 12.0
    )

    if allow_place:
        for atom in place_atoms:
            limit = 3 if total_atoms < 12 else 2
            for row, col in sort_empty_cells_for_atom(obs.grid, atom)[:limit]:
                push_candidate(MaterialForgeAction(action_type="place", row=row, col=col, atom=atom))

    replacement_atoms = prioritized_atoms[:3] or list(ATOM_SYMBOLS)
    if allow_replace:
        for row, col, current_atom in occupied_cells:
            current_primary = ATOM_TO_PRIMARY_PROPERTY.get(current_atom)
            if obs.total_cost < obs.cost_budget and current_primary in top_needed_props:
                continue
            for replacement_atom in replacement_atoms:
                if replacement_atom == current_atom:
                    continue
                push_candidate(
                    MaterialForgeAction(
                        action_type="replace",
                        row=row,
                        col=col,
                        atom=replacement_atom,
                    )
                )

    if allow_remove:
        for row, col, _ in occupied_cells:
            push_candidate(MaterialForgeAction(action_type="remove", row=row, col=col, atom=None))

    if not candidates:
        for atom in ATOM_SYMBOLS:
            empty_cells = sort_empty_cells_for_atom(obs.grid, atom)
            if not empty_cells:
                continue
            row, col = empty_cells[0]
            push_candidate(MaterialForgeAction(action_type="place", row=row, col=col, atom=atom))
            if candidates:
                break

    candidates.sort(
        key=lambda candidate: (
            candidate.gap_score,
            candidate.budget_after,
            0 if candidate.phase == "crystalline" else 1 if candidate.phase == "polycrystalline" else 2,
        )
    )
    return candidates[:max_candidates]


def format_observation(obs, candidates: List[ActionCandidate]) -> str:
    """Format the observation and candidate list for the LLM."""
    gaps = compute_gaps(obs.target, obs.current_properties)
    target_str = ", ".join(f"{prop}: {obs.target[prop]:.1f}" for prop in PROPERTY_NAMES)
    current_str = ", ".join(
        f"{prop}: {obs.current_properties.get(prop, 0.0):.1f}" for prop in PROPERTY_NAMES
    )
    gap_str = ", ".join(f"{prop}: {gaps[prop]:+.1f}" for prop in PROPERTY_NAMES)
    grid_str = "\n".join(" ".join(row) for row in obs.grid)
    candidate_lines = "\n".join(
        f"{idx}. {action_str(candidate.action)} | {candidate.summary}"
        for idx, candidate in enumerate(candidates, start=1)
    )

    return textwrap.dedent(
        f"""\
        Step {obs.step_number}/{obs.max_steps}
        Cost: {obs.total_cost:.0f}/{obs.cost_budget:.0f}
        Phase: {obs.phase}

        Target:  {target_str}
        Current: {current_str}
        Gaps:    {gap_str}

        Candidate actions:
        {candidate_lines}

        Grid:
        {grid_str}

        Return JSON only: {{"candidate_id": 1}}"""
    )


async def choose_action_with_llm(
    client: OpenAI,
    obs,
    candidates: List[ActionCandidate],
) -> MaterialForgeAction:
    """Use the OpenAI client to choose from the generated candidate list."""
    if not candidates:
        return MaterialForgeAction(action_type="remove", row=0, col=0, atom=None)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_observation(obs, candidates)},
    ]

    for _ in range(3):
        try:
            completion = await asyncio.to_thread(
                client.chat.completions.create,
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            content = completion.choices[0].message.content or ""
            candidate_idx = parse_llm_choice(content, len(candidates))
            if candidate_idx is not None:
                return candidates[candidate_idx].action
        except Exception as exc:
            print(f"[DEBUG] LLM request failed: {exc}", flush=True)

    return candidates[0].action


async def warm_up_llm_proxy(client: OpenAI) -> None:
    """Force one request through the injected LiteLLM proxy before tasks start."""
    await asyncio.to_thread(
        client.chat.completions.create,
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": '{"candidate_id": 1}'},
        ],
        temperature=0.0,
        max_tokens=PROXY_PROBE_MAX_TOKENS,
        stream=False,
    )


async def run_task(env: MaterialForgeEnv, client: OpenAI, task: Dict) -> float:
    """Run one benchmark episode and return the best reward reached."""
    rewards: List[float] = []
    steps_taken = 0
    success = False
    seen_state_counts: Dict[str, int] = {}
    recent_action_keys: List[str] = []

    log_start(task=task["name"], env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(
            seed=task["seed"],
            difficulty=task["difficulty"],
            scenario_name=task["name"],
        )
        obs = result.observation
        seen_state_counts[grid_to_key(obs.grid)] = 1

        while not result.done and steps_taken < obs.max_steps:
            candidates = build_candidates(obs, recent_action_keys, seen_state_counts)
            action = await choose_action_with_llm(client, obs, candidates)

            result = await env.step(action)
            obs = result.observation
            reward = result.reward if result.reward is not None else 0.0
            steps_taken += 1
            rewards.append(reward)

            recent_action_keys.append(action_to_key(action))
            if len(recent_action_keys) > 6:
                recent_action_keys = recent_action_keys[-6:]
            state_key = grid_to_key(obs.grid)
            seen_state_counts[state_key] = seen_state_counts.get(state_key, 0) + 1

            error = None
            if obs.metadata and not obs.metadata.get("action_success", True):
                error = "invalid_action"

            log_step(
                step=steps_taken,
                action=action_str(action),
                reward=reward,
                done=result.done,
                error=error,
            )

    except Exception as exc:
        print(f"[DEBUG] Task {task['name']} error: {exc}", flush=True)

    score = max(rewards) if rewards else 0.0
    score = normalize_task_score(score)
    success = score >= SUCCESS_THRESHOLD
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


async def main() -> None:
    """Run the benchmark tasks using a simple LLM baseline policy."""
    try:
        client = OpenAI(
            base_url=os.environ["API_BASE_URL"],
            api_key=os.environ["API_KEY"],
        )
    except KeyError as exc:
        print(f"[DEBUG] {exc}", flush=True)
        for task in TASKS:
            log_start(task=task["name"], env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=MIN_TASK_SCORE, rewards=[])
        return

    try:
        await warm_up_llm_proxy(client)
    except Exception as exc:
        print(f"[DEBUG] LLM proxy warm-up failed: {exc}", flush=True)

    try:
        if SPACE_URL:
            env = MaterialForgeEnv(base_url=SPACE_URL)
            await env.connect()

        elif os.getenv("USE_LOCALHOST"):
            env = MaterialForgeEnv(base_url="http://localhost:8000")
            await env.connect()

        else:
            env = await MaterialForgeEnv.from_docker_image(IMAGE_NAME)

    except Exception as e:
        print(f"[DEBUG] Failed to start environment: {e}", flush=True)

        # Emit minimal structured output so validator can parse results
        for task in TASKS:
            log_start(task=task["name"], env=BENCHMARK, model=MODEL_NAME)
            log_end(success=False, steps=0, score=MIN_TASK_SCORE, rewards=[])

        return

    try:
        scores = []
        for task in TASKS:
            scores.append(await run_task(env, client, task))

        avg_score = sum(scores) / len(scores) if scores else 0.0
        print(
            f"\n[SUMMARY] tasks={len(TASKS)} avg_score={avg_score:.3f} scores={','.join(f'{score:.3f}' for score in scores)}",
            flush=True,
        )
    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close() error: {exc}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
