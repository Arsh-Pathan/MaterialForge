"""
Inference Script — MaterialForge
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local Docker image for the environment
                     (used by from_docker_image()).

- Defaults are set for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The inference script must be named `inference.py` and placed in the root directory of the project.
- Participants must use the OpenAI Client for all LLM calls using the above variables.

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each task should return score in [0, 1].

  Example:
    [START] task=diamond-like env=material_forge_env model=Qwen2.5-72B-Instruct
    [STEP] step=1 action=place(0,0,A) reward=0.12 done=false error=null
    [STEP] step=2 action=place(0,1,A) reward=0.18 done=false error=null
    ...
    [END] success=true steps=25 score=0.72 rewards=0.12,0.18,...
"""

import asyncio
import json
import os
import random
import textwrap
from typing import Dict, List, Optional

from openai import OpenAI

from material_forge_env import MaterialForgeAction, MaterialForgeEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "material-forge-env:latest")
SPACE_URL = os.getenv(
    "SPACE_URL"
)  # e.g. https://ArshPathan-material-forge-env.hf.space
API_KEY = (
    os.getenv("HF_TOKEN")
    or os.getenv("API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("OPENROUTER_API_KEY")
)

API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv(
    "MODEL_NAME", "openrouter/auto"
)
BENCHMARK = "material_forge_env"

# The 3 benchmark tasks — deterministic named scenarios at medium difficulty
TASKS = [
    {"name": "diamond-like", "difficulty": "medium", "seed": 42},
    {"name": "conductor", "difficulty": "medium", "seed": 43},
    {"name": "heat-shield", "difficulty": "medium", "seed": 44},
]

TEMPERATURE = 0.3
MAX_TOKENS = 512
SUCCESS_THRESHOLD = 0.3  # score >= this is considered success

# ---------------------------------------------------------------------------
# Logging helpers (mandatory stdout format)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System prompt for the LLM agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a materials scientist optimizing crystal structures.

GOAL: Match target properties by placing atoms on an 8x8 grid.

ATOM TYPES (use CHEAP atoms first to save budget):
- P (Polymer): cost=2, +elasticity (use first!)
- C (Ceramic): cost=4, +thermal_resistance
- B (Conductor): cost=6, +conductivity  
- A (Metal): cost=8, +hardness (use LAST!)

CRITICAL STRATEGY:
1. Use P first (cost 2) - save budget for important atoms
2. Cluster same atoms together for bonding BONUS
3. Aim for CRYSTALLINE phase for +10% bonus
4. Fill grid evenly - don't bunch in one area

Pick from numbered empty cells.
Respond ONLY: {"cell_num": 1, "atom": "P"}"""


def format_observation(obs) -> str:
    """Format the observation into a concise prompt for the LLM."""
    grid_str = "\n".join("  ".join(row) for row in obs.grid)

    target_str = ", ".join(f"{k}: {v:.1f}" for k, v in obs.target.items())
    current_str = ", ".join(f"{k}: {v:.1f}" for k, v in obs.current_properties.items())

    gaps = {}
    for k in obs.target:
        gaps[k] = obs.target[k] - obs.current_properties.get(k, 0.0)
    gaps_str = ", ".join(f"{k}: {v:+.1f}" for k, v in gaps.items())

    worst = max(gaps, key=lambda k: abs(gaps[k]))
    atom_map = {
        "hardness": "A",
        "conductivity": "B",
        "thermal_resistance": "C",
        "elasticity": "P",
    }
    suggested_atom = atom_map.get(worst, "A")

    empty_cells = [(r, c) for r in range(8) for c in range(8) if obs.grid[r][c] == "."]

    # Format as numbered options - much easier for LLM to pick
    options = []
    for i, (r, c) in enumerate(empty_cells[:20]):  # Limit to 20
        options.append(f"{i + 1}: ({r},{c})")
    empty_options = ", ".join(options)

    return textwrap.dedent(f"""\
Step {obs.step_number}/{obs.max_steps} | Cost: {obs.total_cost:.0f}/{obs.cost_budget:.0f} | Phase: {obs.phase}

TARGET:  {target_str}
CURRENT: {current_str}
GAP:     {gaps_str}

Worst gap: {worst} → use atom {suggested_atom}
Cost budget: {obs.cost_budget:.0f}, used: {obs.total_cost:.0f}

STRATEGY:
- Place atoms NEAR same type for bonding bonus (+reward)
- Crystalline phase = extra bonus!
- Stay under cost budget

AVAILABLE EMPTY CELLS (pick ONE by number): {empty_options}

Grid:
{grid_str}

Respond with JSON: {{"cell_num": 1, "atom": "A"}}""")


def parse_llm_action(text: str, obs) -> Optional[MaterialForgeAction]:
    """Parse the LLM's JSON response into a MaterialForgeAction."""
    if not text:
        return None

    text = text.strip()

    # Remove common prefixes like "Here is the JSON:"
    text = (
        text.replace("Here is the JSON:", "")
        .replace("Here is the JSON", "")
        .replace("Here's the JSON:", "")
    )

    # Strip markdown code fences
    text = text.replace("```json", "").replace("```", "").strip()

    # Find the first JSON object in the response
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return None

    try:
        data = json.loads(text[start:end])

        # Handle new format: cell_num (pick from list)
        if "cell_num" in data:
            cell_num = int(data["cell_num"])
            empty_cells = [
                (r, c) for r in range(8) for c in range(8) if obs.grid[r][c] == "."
            ]
            if 1 <= cell_num <= len(empty_cells):
                r, c = empty_cells[cell_num - 1]
                return MaterialForgeAction(
                    action_type="place",
                    row=r,
                    col=c,
                    atom=data.get("atom", "A"),
                )
            return None

        # Handle old format: row, col
        return MaterialForgeAction(
            action_type=data["action_type"],
            row=int(data["row"]),
            col=int(data["col"]),
            atom=data.get("atom"),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def get_action_from_llm(
    client: OpenAI,
    obs,
    history: List[Dict],
) -> MaterialForgeAction:
    """Query the LLM for the next action given the current observation."""
    user_prompt = format_observation(obs)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-12:])
    messages.append({"role": "user", "content": user_prompt})

    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            text = (completion.choices[0].message.content or "").strip()
            action = parse_llm_action(text, obs)
            if action is not None:
                # Validate: place must be on empty cell, replace on occupied
                if action.action_type == "place":
                    if obs.grid[action.row][action.col] != ".":
                        continue
                elif action.action_type == "replace":
                    if obs.grid[action.row][action.col] == ".":
                        continue
                return action
        except Exception as exc:
            print(
                f"[DEBUG] LLM request failed (attempt {attempt + 1}): {exc}", flush=True
            )

    # Fallback: smart placement on empty cell
    empty_cells = [(r, c) for r in range(8) for c in range(8) if obs.grid[r][c] == "."]
    if empty_cells:
        gaps = {
            k: obs.target[k] - obs.current_properties.get(k, 0.0) for k in obs.target
        }
        worst = max(gaps, key=lambda k: abs(gaps[k]))
        atom_map = {
            "hardness": "A",
            "conductivity": "B",
            "thermal_resistance": "C",
            "elasticity": "P",
        }
        atom = atom_map.get(worst, "P")
        row, col = random.choice(empty_cells)
        return MaterialForgeAction(action_type="place", row=row, col=col, atom=atom)

    return MaterialForgeAction(action_type="remove", row=0, col=0, atom=None)


def action_str(action: MaterialForgeAction) -> str:
    """Format action for [STEP] log line."""
    if action.atom:
        return f"{action.action_type}({action.row},{action.col},{action.atom})"
    return f"{action.action_type}({action.row},{action.col})"


# ---------------------------------------------------------------------------
# Run a single task (episode)
# ---------------------------------------------------------------------------


async def run_task(env: MaterialForgeEnv, client: OpenAI, task: Dict) -> float:
    """Run one episode for a named scenario. Returns the final score in [0, 1]."""
    task_name = task["name"]
    seed = task["seed"]
    difficulty = task["difficulty"]

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: List[Dict] = []

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(
            seed=seed,
            difficulty=difficulty,
            scenario_name=task_name,
        )
        obs = result.observation

        while not result.done and steps_taken < obs.max_steps:
            action = get_action_from_llm(client, obs, history)

            # Record in conversation history
            history.append({"role": "user", "content": format_observation(obs)})
            history.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "action_type": action.action_type,
                            "row": action.row,
                            "col": action.col,
                            "atom": action.atom,
                        }
                    ),
                }
            )

            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            steps_taken += 1

            rewards.append(reward)

            # Check for action errors via metadata
            error = None
            if obs.metadata and not obs.metadata.get("action_success", True):
                error = "invalid_action"

            log_step(
                step=steps_taken,
                action=action_str(action),
                reward=reward,
                done=done,
                error=error,
            )

        # Score = best reward achieved (not last reward)
        score = max(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_name} error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if SPACE_URL:
        env = MaterialForgeEnv(base_url=SPACE_URL)
        await env.connect()
    elif os.getenv("USE_LOCALHOST"):
        env = MaterialForgeEnv(base_url="http://localhost:8000")
        await env.connect()
    else:
        env = await MaterialForgeEnv.from_docker_image(IMAGE_NAME)

    try:
        scores = []
        for task in TASKS:
            score = await run_task(env, client, task)
            scores.append(score)

        avg = sum(scores) / len(scores) if scores else 0.0
        print(
            f"\n[SUMMARY] tasks={len(TASKS)} avg_score={avg:.3f} scores={','.join(f'{s:.3f}' for s in scores)}",
            flush=True,
        )

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
