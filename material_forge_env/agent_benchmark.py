"""MaterialForge Discovery Benchmark Suite v3.0

Runs an LLM agent (local Ollama) across multiple scenarios and difficulties,
gathering episode data for the analytics dashboard. Falls back to a smart
heuristic agent if the LLM is unavailable.

Usage:
    1. Start server:  uv run uvicorn server.app:app --port 7860
    2. Start Ollama:  ollama serve  (with qwen3:8b pulled)
    3. Run:           uv run python agent_benchmark.py
"""

import os
import re
import json
import random
import httpx
from typing import Dict

# ── Config ──────────────────────────────────────────────────────────────
BASE_URL = "http://localhost:7860"
OLLAMA_URL = "http://localhost:11434/v1"
MODEL = "qwen3:8b"
MAX_STEPS_PER_RUN = 15

SCENARIOS = [
    ("diamond-like", "easy"),
    ("diamond-like", "medium"),
    ("diamond-like", "hard"),
    ("conductor", "easy"),
    ("conductor", "medium"),
    ("conductor", "hard"),
    ("heat-shield", "easy"),
    ("heat-shield", "medium"),
    ("heat-shield", "hard"),
    ("", "medium"),  # random scenario
]

SYSTEM_PROMPT = """You are a materials science AI designing crystal structures on an 8x8 lattice.
Available atoms: A (Metal, cost 8, boosts hardness), B (Conductor, cost 6, boosts conductivity),
C (Ceramic, cost 4, boosts thermal_resistance), P (Polymer, cost 2, boosts elasticity).
Actions: place, replace, remove.

Strategy:
- Match target properties by choosing atoms whose primary property matches the highest targets
- Build clusters of same-type atoms for stability bonuses
- Maintain lattice symmetry for quality bonuses
- Stay within budget

Respond with ONLY a JSON object (no markdown, no explanation):
{"action_type": "place", "row": <0-7>, "col": <0-7>, "atom": "<A|B|C|P>"}"""


def extract_json(text: str) -> dict | None:
    """Extract JSON from LLM response, handling thinking tokens and markdown."""
    if not text:
        return None
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try finding JSON in the text
    match = re.search(r'\{[^{}]*"action_type"[^{}]*\}', text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def smart_heuristic_action(obs: Dict, step: int) -> Dict:
    """Smart heuristic that prioritizes target-matching atoms with spatial clustering."""
    grid = obs.get("grid", [])
    target = obs.get("target", {})
    current = obs.get("current_properties", {})
    budget_left = obs.get("cost_budget", 80) - obs.get("total_cost", 0)

    # Determine which property needs the most improvement
    gaps = {}
    prop_to_atom = {
        "hardness": "A", "conductivity": "B",
        "thermal_resistance": "C", "elasticity": "P",
    }
    atom_costs = {"A": 8, "B": 6, "C": 4, "P": 2}

    for prop, atom in prop_to_atom.items():
        tgt = target.get(prop, 0)
        cur = current.get(prop, 0)
        if tgt > 0:
            gaps[prop] = (tgt - cur) / max(tgt, 1)

    # Sort by gap descending
    sorted_props = sorted(gaps.keys(), key=lambda p: gaps[p], reverse=True)

    # Pick atom for highest gap, respecting budget
    chosen_atom = "P"  # cheapest default
    for prop in sorted_props:
        atom = prop_to_atom[prop]
        if atom_costs[atom] <= budget_left:
            chosen_atom = atom
            break

    # Find best empty position (prefer near same-type atoms for clustering)
    best_pos = None
    best_score = -1

    for r in range(8):
        for c in range(8):
            if grid[r][c] != ".":
                continue
            score = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    if grid[nr][nc] == chosen_atom:
                        score += 3  # cluster bonus
                    elif grid[nr][nc] != ".":
                        score += 1  # neighbor bonus
            # Prefer center positions
            score += (3.5 - abs(r - 3.5)) * 0.1 + (3.5 - abs(c - 3.5)) * 0.1
            if score > best_score:
                best_score = score
                best_pos = (r, c)

    if best_pos is None:
        return {"action_type": "remove", "row": 0, "col": 0}

    return {
        "action_type": "place",
        "row": best_pos[0],
        "col": best_pos[1],
        "atom": chosen_atom,
    }


def get_obs(data) -> Dict:
    """Extract observation dict containing 'grid' from nested response."""
    if isinstance(data, dict):
        if "grid" in data:
            return data
        for v in data.values():
            res = get_obs(v)
            if res:
                return res
    if isinstance(data, list):
        for item in data:
            res = get_obs(item)
            if res:
                return res
    return {}


def try_llm_action(obs: Dict) -> dict | None:
    """Try to get an action from the local LLM."""
    try:
        grid_str = "\n".join(" ".join(row) for row in obs.get("grid", []))
        target = obs.get("target", {})
        current = obs.get("current_properties", {})
        cost = obs.get("total_cost", 0)
        budget = obs.get("cost_budget", 80)
        phase = obs.get("phase", "amorphous")

        user_msg = (
            f"Target: {json.dumps(target)}\n"
            f"Current: {json.dumps(current)}\n"
            f"Phase: {phase} | Cost: {cost}/{budget}\n"
            f"Grid:\n{grid_str}\n\n"
            f"Choose your next action as JSON:"
        )

        response = httpx.post(
            f"{OLLAMA_URL}/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                "max_tokens": 512,
                "temperature": 0.3,
            },
            timeout=30.0,
        )
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return extract_json(content)
    except Exception:
        return None


def run_episode(scenario: str, difficulty: str, use_llm: bool = True) -> Dict:
    """Run a single episode, returning metrics."""
    label = scenario or "random"
    print(f"\n--- {label} ({difficulty}) ---")

    try:
        payload = {"difficulty": difficulty}
        if scenario:
            payload["scenario_name"] = scenario
        r = httpx.post(f"{BASE_URL}/reset", json=payload, timeout=10.0)
        res = r.json()
        if isinstance(res, list):
            res = res[0]
        obs = get_obs(res)
        if not obs:
            print(f"  [ERROR] No observation in reset response")
            return {}
    except Exception as e:
        print(f"  [ERROR] Reset failed: {e}")
        return {}

    best_reward = 0.0
    current_reward = 0.0
    steps = 0
    llm_used = 0
    heuristic_used = 0
    reward_trace = []

    for step in range(MAX_STEPS_PER_RUN):
        steps = step + 1
        action = None

        if use_llm:
            action = try_llm_action(obs)
            if action:
                llm_used += 1
            else:
                action = smart_heuristic_action(obs, step)
                heuristic_used += 1
        else:
            action = smart_heuristic_action(obs, step)
            heuristic_used += 1

        # Validate action
        if not action or "action_type" not in action:
            action = smart_heuristic_action(obs, step)
            heuristic_used += 1

        try:
            r = httpx.post(f"{BASE_URL}/step", json=action, timeout=10.0)
            result = r.json()
            if isinstance(result, list):
                result = result[0]
        except Exception as e:
            print(f"  Step {steps}: API error: {e}")
            continue

        new_obs = get_obs(result)
        if new_obs and "grid" in new_obs:
            obs = new_obs

        current_reward = result.get("reward") or 0.0
        best_reward = max(best_reward, current_reward)
        reward_trace.append(round(current_reward, 4))

        src = "LLM" if (use_llm and action == try_llm_action) else "H"
        print(f"  Step {steps}: {action['action_type']} ({action.get('atom','')}) [{action.get('row','')},{action.get('col','')}] -> {current_reward:.4f}")

        if result.get("done", False):
            print(f"  Episode complete!")
            break

    agent_type = "llm" if llm_used > 0 else "heuristic"
    print(f"  Result: best={best_reward:.4f} phase={obs.get('phase','?')} agent={agent_type} (llm:{llm_used}/heur:{heuristic_used})")

    return {
        "scenario": label,
        "difficulty": difficulty,
        "final_reward": round(current_reward, 4),
        "best_reward": round(best_reward, 4),
        "steps": steps,
        "phase": obs.get("phase", "amorphous"),
        "total_cost": round(obs.get("total_cost", 0), 1),
        "cost_budget": obs.get("cost_budget", 80),
        "agent_type": agent_type,
        "llm_steps": llm_used,
        "heuristic_steps": heuristic_used,
        "reward_trace": reward_trace,
    }


def main():
    print("=" * 60)
    print("MaterialForge Discovery Benchmark Suite v3.0")
    print("=" * 60)

    # Check server
    try:
        httpx.get(f"{BASE_URL}/health", timeout=5.0)
        print(f"Server: {BASE_URL} [OK]")
    except Exception:
        print(f"CRITICAL: Server not found at {BASE_URL}")
        print("Run: uv run uvicorn server.app:app --port 7860")
        return

    # Check Ollama
    llm_available = False
    try:
        r = httpx.get(f"{OLLAMA_URL[:-3]}/api/tags", timeout=5.0)
        models = [m["name"] for m in r.json().get("models", [])]
        if any(MODEL.split(":")[0] in m for m in models):
            llm_available = True
            print(f"LLM: {MODEL} via Ollama [OK]")
        else:
            print(f"LLM: {MODEL} not found in Ollama (available: {models})")
    except Exception:
        print("LLM: Ollama not reachable, using heuristic agent only")

    all_runs = []
    for scenario, diff in SCENARIOS:
        result = run_episode(scenario, diff, use_llm=llm_available)
        if result:
            all_runs.append(result)

    # Save results
    output_path = os.path.join("server", "static", "benchmarks.json")
    with open(output_path, "w") as f:
        json.dump(all_runs, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"BENCHMARK COMPLETE: {len(all_runs)} episodes")
    print(f"Output: {output_path}")

    # Summary stats
    if all_runs:
        avg_best = sum(r["best_reward"] for r in all_runs) / len(all_runs)
        phases = {}
        for r in all_runs:
            phases[r["phase"]] = phases.get(r["phase"], 0) + 1
        print(f"Avg best reward: {avg_best:.4f}")
        print(f"Phase distribution: {phases}")
    print("=" * 60)


if __name__ == "__main__":
    main()
