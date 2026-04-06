#!/usr/bin/env python3
"""Demo script for MaterialForge environment.

Shows a complete agent interaction loop: reset → step → observe → repeat.
Runs locally without needing a server (direct environment usage).

Usage:
    cd material_forge_env
    uv run python demo.py
"""

from material_forge_env.environment.config import GRID_SIZE, PROPERTY_NAMES
from material_forge_env.environment.rubrics import HeuristicRewardRubric
from material_forge_env.models import MaterialForgeAction
from material_forge_env.server.material_forge_env_environment import (
    MaterialForgeEnvironment,
)


def print_grid(grid: list[list[str]]) -> None:
    """Pretty-print the lattice grid."""
    header = "  " + " ".join(str(c) for c in range(len(grid[0])))
    print(header)
    for r, row in enumerate(grid):
        print(f"{r} " + " ".join(row))


def print_properties(target: dict, current: dict) -> None:
    """Print target vs current properties side by side."""
    print(f"  {'Property':<22} {'Target':>8} {'Current':>8} {'Gap':>8}")
    print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8}")
    for prop in PROPERTY_NAMES:
        t = target.get(prop, 0)
        c = current.get(prop, 0)
        gap = c - t
        sign = "+" if gap >= 0 else ""
        print(f"  {prop:<22} {t:>8.1f} {c:>8.1f} {sign}{gap:>7.1f}")


def run_demo():
    """Run a full episode demonstrating the environment."""
    print("=" * 60)
    print("  MaterialForge Demo — AI-Driven Crystal Structure Design")
    print("=" * 60)

    # Create environment with heuristic reward rubric
    env = MaterialForgeEnvironment(rubric=HeuristicRewardRubric())

    # --- Episode 1: Diamond-like material ---
    print("\n--- Episode: Diamond-Like Material (Easy) ---\n")
    obs = env.reset(seed=42, difficulty="easy", scenario_name="diamond-like")

    print(f"Target: {obs.target}")
    print(f"Budget: {obs.cost_budget} | Max Steps: {obs.max_steps}")
    print()

    # Strategy: fill a 4x4 block with metal (A) for high hardness,
    # add some ceramic (C) for thermal resistance
    actions = [
        # Metal core for hardness
        *[MaterialForgeAction(action_type="place", row=r, col=c, atom="A")
          for r in range(4) for c in range(4)],
        # Ceramic border for thermal resistance
        *[MaterialForgeAction(action_type="place", row=r, col=4, atom="C")
          for r in range(4)],
        *[MaterialForgeAction(action_type="place", row=4, col=c, atom="C")
          for c in range(5)],
    ]

    best_reward = 0.0
    for i, action in enumerate(actions):
        obs = env.step(action)
        if obs.reward > best_reward:
            best_reward = obs.reward
        if (i + 1) % 5 == 0 or obs.done:
            print(f"  Step {obs.step_number:>2}: reward={obs.reward:.4f}  "
                  f"phase={obs.phase:<16} cost={obs.total_cost:.0f}/{obs.cost_budget:.0f}")
        if obs.done:
            break

    print(f"\n  Final Grid:")
    print_grid(obs.grid)
    print()
    print_properties(obs.target, obs.current_properties)
    print(f"\n  Best reward: {best_reward:.4f}")
    print(f"  Phase: {obs.phase}")
    print(f"  Done: {obs.done} (step {obs.step_number}/{obs.max_steps})")

    # --- Episode 2: Conductor material ---
    print("\n\n--- Episode: Conductor Material (Medium) ---\n")
    obs = env.reset(seed=43, difficulty="medium", scenario_name="conductor")

    print(f"Target: {obs.target}")
    print(f"Budget: {obs.cost_budget} | Max Steps: {obs.max_steps}")
    print()

    # Strategy: conductor atoms (B) in a connected network
    actions = [
        *[MaterialForgeAction(action_type="place", row=r, col=c, atom="B")
          for r in range(4) for c in range(4)],
        # Add some polymer for elasticity
        *[MaterialForgeAction(action_type="place", row=r, col=c, atom="P")
          for r in range(4, 6) for c in range(3)],
    ]

    best_reward = 0.0
    for i, action in enumerate(actions):
        obs = env.step(action)
        if obs.reward > best_reward:
            best_reward = obs.reward
        if (i + 1) % 5 == 0 or obs.done:
            print(f"  Step {obs.step_number:>2}: reward={obs.reward:.4f}  "
                  f"phase={obs.phase:<16} cost={obs.total_cost:.0f}/{obs.cost_budget:.0f}")
        if obs.done:
            break

    print(f"\n  Final Grid:")
    print_grid(obs.grid)
    print()
    print_properties(obs.target, obs.current_properties)
    print(f"\n  Best reward: {best_reward:.4f}")
    print(f"  Phase: {obs.phase}")
    print(f"  Done: {obs.done} (step {obs.step_number}/{obs.max_steps})")

    print("\n" + "=" * 60)
    print("  Demo complete. The environment is ready for AI agents!")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
