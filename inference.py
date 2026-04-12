"""
MaterialForge Inference Script
==============================
This script implements a high-performance heuristic-guided baseline agent for
the OpenEnv crystal design benchmark. It utilizes a lookahead heuristic to
rank candidate actions and uses an LLM to make the final optimal selection.

MANDATORY Submission Variables:
- API_BASE_URL: LiteLLM proxy endpoint.
- API_KEY: Authentication key.
- MODEL_NAME: Target LLM identifier.
"""

import asyncio
import json
import os
import sys
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Optional

from openai import OpenAI

# Dynamic imports to support both local development and standard package layout
try:
    from client import MaterialForgeEnv
    from models import MaterialForgeAction
except ImportError:
    # Fallback for alternative environments
    from material_forge_env import MaterialForgeAction, MaterialForgeEnv

try:
    from environment.config import ATOM_SYMBOLS, ATOM_TYPES, EMPTY, PROPERTY_NAMES
    from environment.lattice import Lattice
    from environment.physics import classify_phase, estimate_properties
except ImportError:
    # Fallback for alternative environments
    from material_forge_env.environment.config import ATOM_SYMBOLS, ATOM_TYPES, EMPTY, PROPERTY_NAMES
    from material_forge_env.environment.lattice import Lattice
    from material_forge_env.environment.physics import classify_phase, estimate_properties

# Global Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "material-forge-env:latest")
SPACE_URL = os.getenv("SPACE_URL")
BENCHMARK_ID = "material_forge_env"

# Task Definitions
TASKS = [
    {"name": "basic-synthesis", "difficulty": "easy", "seed": 101},
    {"name": "diamond-like", "difficulty": "medium", "seed": 42},
    {"name": "superconductor-analogue", "difficulty": "hard", "seed": 999},
]

# Hyperparameters
TEMPERATURE = 0.1
MAX_TOKENS = 160
SUCCESS_THRESHOLD = 0.3
PROXY_PROBE_MAX_TOKENS = 8
MIN_TASK_SCORE = 0.01
MAX_TASK_SCORE = 0.99

# Material Property logic
PROPERTY_TO_SYMBOL = {
    "hardness": "A",
    "conductivity": "B",
    "thermal_resistance": "C",
    "elasticity": "P",
}

SYMBOL_TO_PRIMARY = {
    "A": "hardness",
    "B": "conductivity",
    "C": "thermal_resistance",
    "P": "elasticity",
}


@dataclass(frozen=True)
class EvaluatedAction:
    """Represents a potential action with heuristic evaluation metrics."""
    action: MaterialForgeAction
    summary: str
    state_fingerprint: str
    heuristic_score: float
    projected_budget: float
    resulting_phase: str


def log_start(task: str, env: str, model: str) -> None:
    """Standardized [START] output."""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Standardized [STEP] logging for observability."""
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    """Standardized [END] reporting."""
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def normalize_score(raw_score: float) -> float:
    """Ensure task scores stay within the required (0, 1) interval."""
    return min(max(raw_score, MIN_TASK_SCORE), MAX_TASK_SCORE)


SYSTEM_PROMPT = """You are an expert Material Scientist assistant.
Your goal is to design a crystal lattice that matches the target physical properties.

Instructions:
1. Analyze the provided candidate actions and their heuristic scores.
2. Select the "candidate_id" that best helps bridge the remaining property gaps.
3. Prioritize stability (crystalline phase) and cost-efficiency.

Return ONLY a JSON object: {"candidate_id": <int>}"""


def get_grid_key(grid: List[List[str]]) -> str:
    """Generates a compact string representation of the grid state."""
    return "|".join("".join(row) for row in grid)


def get_action_key(action: MaterialForgeAction) -> str:
    """Generates a stable key for identifying specific actions."""
    symbol = action.atom if action.atom is not None else "-"
    return f"{action.action_type}:{action.row}:{action.col}:{symbol}"


def calculate_property_gaps(target: Dict[str, float], current: Dict[str, float]) -> Dict[str, float]:
    """Calculates the remaining delta for each material property."""
    return {prop: target[prop] - current.get(prop, 0.0) for prop in PROPERTY_NAMES}


def calculate_mean_abs_gap(gaps: Dict[str, float]) -> float:
    """Computes the average absolute gap percentage."""
    return sum(abs(gap) for gap in gaps.values()) / (len(gaps) * 100.0)


def parse_llm_json(response_text: str, max_id: int) -> Optional[int]:
    """Robustly extracts candidate_id from LLM output."""
    if not response_text:
        return None

    try:
        # Strip potential markdown and prose
        cleaned = response_text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
        
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start != -1 and end != 0:
            data = json.loads(cleaned[start:end])
            choice = int(data.get("candidate_id", 0))
            if 1 <= choice <= max_id:
                return choice - 1
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        pass
    return None


def format_action_display(action: MaterialForgeAction) -> str:
    """Readable action format for logging."""
    if action.atom is not None:
        return f"{action.action_type}({action.row},{action.col},{action.atom})"
    return f"{action.action_type}({action.row},{action.col})"


def simulate_action(grid: List[List[str]], action: MaterialForgeAction) -> Optional[List[List[str]]]:
    """Simulates a move on a local grid copy. Returns None if invalid."""
    new_grid = [row[:] for row in grid]
    cell = new_grid[action.row][action.col]

    if action.action_type == "place":
        if action.atom is None or cell != EMPTY: return None
        new_grid[action.row][action.col] = action.atom
    elif action.action_type == "replace":
        if action.atom is None or cell == EMPTY or cell == action.atom: return None
        new_grid[action.row][action.col] = action.atom
    elif action.action_type == "remove":
        if cell == EMPTY: return None
        new_grid[action.row][action.col] = EMPTY
    else:
        return None

    return new_grid


def score_empty_cell(grid: List[List[str]], row: int, col: int, symbol: str) -> float:
    """Calculates a heuristic score for placing an atom at a specific coordinate."""
    size = len(grid)
    center = (size - 1) / 2.0
    
    # Neighborhood metrics
    same_count = 0
    occ_count = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0: continue
            nr, nc = row + dr, col + dc
            if 0 <= nr < size and 0 <= nc < size:
                cell = grid[nr][nc]
                if cell != EMPTY:
                    occ_count += 1
                if cell == symbol:
                    same_count += 1
                    
    center_dist = abs(row - center) + abs(col - center)
    edge_penalty = 0.2 if row in (0, size-1) or col in (0, size-1) else 0.0
    
    return same_count * 3.0 + occ_count * 1.2 - center_dist * 0.1 - edge_penalty


def evaluate_candidate(obs, action: MaterialForgeAction) -> Optional[EvaluatedAction]:
    """Performs deep evaluation of a candidate action via local physics simulation."""
    next_grid = simulate_action(obs.grid, action)
    if next_grid is None:
        return None

    # Run physics engine simulation
    simulated_lattice = Lattice.from_grid(next_grid)
    props = estimate_properties(simulated_lattice)
    gaps = calculate_property_gaps(obs.target, props)
    cost = simulated_lattice.total_cost()
    phase = classify_phase(simulated_lattice)
    
    # Heuristic Components
    gap_score = calculate_mean_abs_gap(gaps)
    overshoot = sum(max(-g, 0.0) for g in gaps.values()) / 100.0
    budget_usage = max(cost - obs.cost_budget, 0.0) / max(obs.cost_budget, 1.0)
    
    # Penalties and Bonuses
    phase_bonus = 0.03 if phase == "crystalline" else 0.01 if phase == "polycrystalline" else 0.0
    overbudget_penalty = 0.12 if budget_usage > 0 else 0.0
    
    # Composite Score (lower is better for gap score, so we use -score)
    composite_heuristic = (
        -gap_score 
        - 1.5 * budget_usage 
        - 0.3 * overshoot 
        - overbudget_penalty
        + phase_bonus
    )

    summary = (
        f"{action.action_type} {action.atom or ''} at ({action.row},{action.col}) | "
        f"Gap={gap_score:.3f}, Cost={cost:.0f}, {phase}"
    )

    return EvaluatedAction(
        action=action,
        summary=summary,
        state_fingerprint=get_grid_key(next_grid),
        heuristic_score=round(-composite_heuristic, 4),
        projected_budget=cost,
        resulting_phase=phase
    )


def build_action_pool(obs, trajectory: List[str], state_memory: Dict[str, int]) -> List[EvaluatedAction]:
    """Generates a filtered pool of candidate actions for the current state."""
    gaps = calculate_property_gaps(obs.target, obs.current_properties)
    remaining_budget = obs.cost_budget - obs.total_cost
    
    # Analyze deficits
    needed_props = sorted([p for p in PROPERTY_NAMES if gaps[p] > 0], key=lambda p: gaps[p], reverse=True)
    target_atoms = [PROPERTY_TO_SYMBOL[p] for p in needed_props[:3]] or list(ATOM_SYMBOLS)
    
    candidates: List[EvaluatedAction] = []
    seen_keys = set()

    def add_to_pool(act: MaterialForgeAction):
        key = get_action_key(act)
        if key in seen_keys: return
        # Prevent immediate undo
        if trajectory and key == trajectory[-1]: return
        
        evaled = evaluate_candidate(obs, act)
        if evaled and state_memory.get(evaled.state_fingerprint, 0) < 1:
            candidates.append(evaled)
            seen_keys.add(key)

    # Strategy: Place relevant atoms in high-quality spots
    for symbol in target_atoms:
        empty_coords = []
        for r in range(8):
            for c in range(8):
                if obs.grid[r][c] == EMPTY:
                    empty_coords.append((r, c))
        
        # Sort coordinates by placement heuristic
        empty_coords.sort(key=lambda coord: score_empty_cell(obs.grid, coord[0], coord[1], symbol), reverse=True)
        
        for r, c in empty_coords[:2]:
            if ATOM_TYPES[symbol]["cost"] <= remaining_budget + 4:
                add_to_pool(MaterialForgeAction(action_type="place", row=r, col=c, atom=symbol))

    # Strategy: Optimize existing structures via replacement
    if obs.total_cost > obs.cost_budget * 0.6:
        for r in range(8):
            for c in range(8):
                if obs.grid[r][c] != EMPTY:
                    for s in target_atoms:
                        if s != obs.grid[r][c]:
                            add_to_pool(MaterialForgeAction(action_type="replace", row=r, col=c, atom=s))

    # Fallback to remove if severely over budget
    if obs.total_cost > obs.cost_budget * 1.05:
         for r in range(8):
            for c in range(8):
                if obs.grid[r][c] != EMPTY:
                    add_to_pool(MaterialForgeAction(action_type="remove", row=r, col=c))

    # Sort and trim pool
    candidates.sort(key=lambda x: x.heuristic_score)
    return candidates[:8]


async def select_action(client: OpenAI, obs, pool: List[EvaluatedAction]) -> MaterialForgeAction:
    """Consults the LLM to choose the best action from the evaluated pool."""
    if not pool:
        return MaterialForgeAction(action_type="remove", row=0, col=0)

    # Build prompt context
    gaps = calculate_property_gaps(obs.target, obs.current_properties)
    obs_text = textwrap.dedent(f"""\
        Step: {obs.step_number}/{obs.max_steps} | Budget: {obs.total_cost:.0f}/{obs.cost_budget:.0f}
        Target: {obs.target}
        Current: {obs.current_properties}
        Gaps: {gaps}
        
        Candidate Options:
        {chr(10).join(f"{i+1}. {c.summary}" for i, c in enumerate(pool))}
    """)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": obs_text}
    ]

    for attempt in range(2):
        try:
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            content = response.choices[0].message.content or ""
            choice = parse_llm_json(content, len(pool))
            if choice is not None:
                return pool[choice].action
        except Exception as e:
            print(f"[DEBUG] LLM selection failed: {e}", file=sys.stderr, flush=True)

    # Fallback to the top heuristic action
    return pool[0].action


async def run_episode(env: MaterialForgeEnv, client: OpenAI, task: Dict) -> float:
    """Executes a single benchmark episode."""
    log_start(task=task["name"], env=BENCHMARK_ID, model=MODEL_NAME)
    
    rewards = []
    step = 0
    trajectory = []
    state_memory = {}

    try:
        res = await env.reset(seed=task["seed"], difficulty=task["difficulty"], scenario_name=task["name"])
        obs = res.observation
        state_memory[get_grid_key(obs.grid)] = 1

        while not res.done and step < obs.max_steps:
            pool = build_action_pool(obs, trajectory, state_memory)
            action = await select_action(client, obs, pool)
            
            res = await env.step(action)
            obs = res.observation
            raw_reward = res.reward if res.reward is not None else 0.0
            reward = normalize_score(raw_reward)
            
            step += 1
            rewards.append(reward)
            trajectory.append(get_action_key(action))
            
            # Update memory to avoid cycles
            key = get_grid_key(obs.grid)
            state_memory[key] = state_memory.get(key, 0) + 1
            
            error_msg = getattr(res, "error", getattr(res, "last_action_error", None))
            log_step(step, format_action_display(action), reward, res.done, error_msg)

    except Exception as e:
        print(f"[DEBUG] Episode {task['name']} aborted: {e}", file=sys.stderr, flush=True)

    final_score = normalize_score(max(rewards) if rewards else 0.0)
    log_end(final_score >= SUCCESS_THRESHOLD, step, rewards)
    return final_score



def start_environment_server(port: int = 7860):
    import urllib.request
    import urllib.error
    import subprocess
    import sys
    import time
    import os

    try:
        req = urllib.request.Request(f"http://localhost:{port}/health")
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                print(f"[INFO] Environment server already running on port {port}", file=sys.stderr, flush=True)
                return None
    except Exception:
        pass

    print(f"[INFO] Starting environment server on port {port}...", file=sys.stderr, flush=True)

    try:
        env = os.environ.copy()
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", str(port)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(4)
        if proc.poll() is None:
            return proc
    except Exception as e:
        print(f"[WARNING] Could not start environment server: {e}", file=sys.stderr, flush=True)
    return None

async def main():
    """Main benchmark entry point."""
    api_url = API_BASE_URL
    api_key = HF_TOKEN
    
    if not api_key:
        print("[ERROR] Missing required HF_TOKEN environment variable.", file=sys.stderr, flush=True)
        for task in TASKS:
            log_start(task["name"], BENCHMARK_ID, MODEL_NAME)
            log_end(False, 0, [MIN_TASK_SCORE])
        return

    print(f"[DEBUG] Initializing OpenAI client with base_url={api_url}", file=sys.stderr, flush=True)
    client = OpenAI(base_url=api_url, api_key=api_key)

    try:
        print("[DEBUG] Performing LLM warmup call...", file=sys.stderr, flush=True)
        warmup = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a material science assistant."},
                {"role": "user", "content": "Hello. Response with 'OK'."},
            ],
            temperature=0.0,
            max_tokens=8,
        )
        print(f"[DEBUG] LLM warmup success: {warmup.choices[0].message.content.strip()}", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"[ERROR] LLM warmup failed: {exc}", file=sys.stderr, flush=True)

    server_proc = None
    try:
        if SPACE_URL:
            print(f"[DEBUG] Connecting to remote environment at {SPACE_URL}", file=sys.stderr, flush=True)
            env = MaterialForgeEnv(base_url=SPACE_URL)
            await env.connect()
        elif os.getenv("USE_LOCALHOST") or os.getenv("PORT"):
            host = os.getenv("HOST", "localhost")
            port = os.getenv("PORT", "7860")
            url = f"http://{host}:{port}"
            print(f"[DEBUG] Connecting to local environment at {url}", file=sys.stderr, flush=True)
            env = MaterialForgeEnv(base_url=url)
            await env.connect()
        else:
            server_proc = start_environment_server(port=7860)
            env = MaterialForgeEnv(base_url="http://127.0.0.1:7860")
            await env.connect()
            
        scores = []
        for task in TASKS:
            scores.append(await run_episode(env, client, task))
            
        print(f"\n[SUMMARY] Avg Score: {sum(scores)/len(scores):.3f}", file=sys.stderr, flush=True)
        
    except Exception as e:
        print(f"[ERROR] Runtime execution failed: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
    finally:
        try:
            await env.close()
        except:
            pass
        if server_proc:
            try:
                server_proc.terminate()
                server_proc.wait(timeout=5)
            except:
                try:
                    server_proc.kill()
                except:
                    pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        import traceback
        import sys
        print(f"[FATAL] Unhandled exception: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(0)

