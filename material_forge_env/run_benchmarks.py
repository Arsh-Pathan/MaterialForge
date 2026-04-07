"""Quick benchmark runner using the environment directly (no server needed)."""

import json
import random
from environment.rubrics import HeuristicRewardRubric
from server.material_forge_env_environment import MaterialForgeEnvironment
from models import MaterialForgeAction

PROP_TO_ATOM = {"hardness": "A", "conductivity": "B", "thermal_resistance": "C", "elasticity": "P"}
ATOM_COSTS = {"A": 8, "B": 6, "C": 4, "P": 2}

SCENARIOS = [
    ("diamond-like", "easy"), ("diamond-like", "medium"), ("diamond-like", "hard"),
    ("conductor", "easy"), ("conductor", "medium"), ("conductor", "hard"),
    ("heat-shield", "easy"), ("heat-shield", "medium"), ("heat-shield", "hard"),
    ("", "medium"), ("", "easy"), ("", "hard"),
]


def smart_action(obs_dict, step):
    grid = obs_dict["grid"]
    target = obs_dict["target"]
    current = obs_dict["current_properties"]
    budget_left = obs_dict["cost_budget"] - obs_dict["total_cost"]

    gaps = {}
    for prop in PROP_TO_ATOM:
        tgt = target.get(prop, 0)
        cur = current.get(prop, 0)
        if tgt > 0:
            gaps[prop] = (tgt - cur) / max(tgt, 1)

    sorted_props = sorted(gaps.keys(), key=lambda p: gaps[p], reverse=True)
    chosen = "P"
    for p in sorted_props:
        a = PROP_TO_ATOM[p]
        if ATOM_COSTS[a] <= budget_left:
            chosen = a
            break

    best_pos, best_sc = None, -1
    for r in range(8):
        for c in range(8):
            if grid[r][c] != ".":
                continue
            sc = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 8 and 0 <= nc < 8:
                    if grid[nr][nc] == chosen:
                        sc += 3
                    elif grid[nr][nc] != ".":
                        sc += 1
            sc += (3.5 - abs(r - 3.5)) * 0.1 + (3.5 - abs(c - 3.5)) * 0.1
            if sc > best_sc:
                best_sc, best_pos = sc, (r, c)

    if not best_pos:
        return MaterialForgeAction(action_type="remove", row=0, col=0)
    return MaterialForgeAction(action_type="place", row=best_pos[0], col=best_pos[1], atom=chosen)


def run_episode(scenario, difficulty, max_steps=25):
    env = MaterialForgeEnvironment(rubric=HeuristicRewardRubric())
    kwargs = {"difficulty": difficulty}
    if scenario:
        kwargs["scenario_name"] = scenario
    obs = env.reset(**kwargs)
    obs_dict = obs.model_dump()

    best_rwd, cur_rwd, steps = 0.0, 0.0, 0
    trace = []

    for step in range(max_steps):
        steps = step + 1
        action = smart_action(obs_dict, step)
        obs = env.step(action)
        obs_dict = obs.model_dump()
        cur_rwd = obs_dict.get("reward", 0.0)
        done = obs_dict.get("done", False)
        best_rwd = max(best_rwd, cur_rwd)
        trace.append(round(cur_rwd, 4))
        if done:
            break

    label = scenario if scenario else "random"
    return {
        "scenario": label,
        "difficulty": difficulty,
        "final_reward": round(cur_rwd, 4),
        "best_reward": round(best_rwd, 4),
        "steps": steps,
        "phase": obs_dict.get("phase", "amorphous"),
        "total_cost": round(obs_dict.get("total_cost", 0), 1),
        "cost_budget": obs_dict.get("cost_budget", 80),
        "agent_type": "heuristic",
        "reward_trace": trace,
    }


def main():
    print("MaterialForge Benchmark Suite (Direct Mode)")
    print("=" * 50)

    all_runs = []
    for scenario, diff in SCENARIOS:
        label = scenario or "random"
        result = run_episode(scenario, diff)
        all_runs.append(result)
        print(f"  {label:15s} ({diff:6s}): best={result['best_reward']:.4f}  phase={result['phase']:18s}  cost={result['total_cost']:.0f}/{result['cost_budget']}")

    with open("server/static/benchmarks.json", "w") as f:
        json.dump(all_runs, f, indent=2)

    print(f"\n{'=' * 50}")
    print(f"Saved {len(all_runs)} episodes to server/static/benchmarks.json")

    avg = sum(r["best_reward"] for r in all_runs) / len(all_runs)
    phases = {}
    for r in all_runs:
        phases[r["phase"]] = phases.get(r["phase"], 0) + 1
    print(f"Avg best reward: {avg:.4f}")
    print(f"Phase distribution: {phases}")


if __name__ == "__main__":
    main()
