"""
MaterialForge TRL Environment Wrapper
======================================
Wraps the MaterialForge environment for use with TRL's GRPOTrainer
via the environment_factory pattern. Exposes crystal lattice operations
as named tool methods that the LLM discovers and calls during training.
"""

import random
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.config import GRID_SIZE, PROPERTY_NAMES, ATOM_TYPES
from environment.rubrics import HeuristicRewardRubric
from models import MaterialForgeAction, MaterialForgeObservation
from server.material_forge_env_environment import MaterialForgeEnvironment


# --- Curriculum Learning State ---
TOTAL_EPISODES_STARTED = 0

def get_curriculum_difficulty() -> str:
    """Selects difficulty based on global episode progress (scaled for 100 episodes)."""
    global TOTAL_EPISODES_STARTED
    if TOTAL_EPISODES_STARTED < 25:
        difficulty = "easy"
    elif TOTAL_EPISODES_STARTED < 75:
        difficulty = "medium"
    else:
        difficulty = "hard"
    
    TOTAL_EPISODES_STARTED += 1
    return difficulty


class MaterialForgeTRLEnv:
    """TRL-compatible wrapper for MaterialForge crystal design."""

    def __init__(self):
        self.env = MaterialForgeEnvironment(rubric=HeuristicRewardRubric())
        self.reward = 0.0
        self.done = False
        self._obs = None
        
        # Performance Tracking
        self._best_reward = 0.0
        self._invalid_actions = 0
        self._total_actions = 0
        self._best_phase = "amorphous"
        self._best_rows_used = 0
        self._best_cols_used = 0

    def reset(self, **kwargs) -> str:
        """Resets the environment with a curriculum-based difficulty."""
        seed = random.randint(0, 99999)
        difficulty = get_curriculum_difficulty()
        
        obs = self.env.reset(seed=seed, difficulty=difficulty)
        self._obs = obs
        self.reward = 0.0
        self.done = False
        
        # Reset counters
        self._best_reward = 0.0
        self._invalid_actions = 0
        self._total_actions = 0
        self._best_phase = obs.phase
        self._best_rows_used = 0
        self._best_cols_used = 0
        
        return self._format_observation(obs)

    def place_atom(self, row: int, col: int, atom: str) -> str:
        """Place an atom on an empty cell of the crystal lattice."""
        return self._do_step("place", row, col, atom)

    def remove_atom(self, row: int, col: int) -> str:
        """Remove an atom from the crystal lattice."""
        return self._do_step("remove", row, col, None)

    def replace_atom(self, row: int, col: int, atom: str) -> str:
        """Replace an existing atom with a different species."""
        return self._do_step("replace", row, col, atom)

    def _do_step(self, action_type: str, row: int, col: int, atom: Optional[str]) -> str:
        if self.done:
            return "ERROR: Episode is over. Stop calling tools."

        self._total_actions += 1
        
        try:
            row = max(0, min(int(row), GRID_SIZE - 1))
            col = max(0, min(int(col), GRID_SIZE - 1))
        except (ValueError, TypeError):
            self._invalid_actions += 1
            self.reward = self.compute_episode_reward()
            return "ERROR: Coordinates must be integers 0-7."

        if atom is not None:
            atom = str(atom).upper().strip()
            if atom not in ATOM_TYPES:
                atom = "A"

        # --- Invalid Action Pre-check ---
        grid = self._obs.grid if self._obs else None
        if grid:
            cell = grid[row][col]
            if action_type == "place" and cell != ".":
                self._invalid_actions += 1
                self.reward = self.compute_episode_reward()
                return (
                    f"INVALID ACTION: ({row},{col}) is already occupied by '{cell}'. "
                    "Use replace_atom.\n" + self._format_observation(self._obs)
                )
            
            if action_type == "remove" and cell == ".":
                self._invalid_actions += 1
                self.reward = self.compute_episode_reward()
                return (
                    f"INVALID ACTION: Cannot remove from empty cell ({row},{col}).\n"
                    + self._format_observation(self._obs)
                )
            
            if action_type == "replace" and (cell == "." or cell == atom):
                self._invalid_actions += 1
                self.reward = self.compute_episode_reward()
                return (
                    f"INVALID ACTION: replace_atom at ({row},{col}) is redundant or invalid.\n"
                    + self._format_observation(self._obs)
                )

        action = MaterialForgeAction(action_type=action_type, row=row, col=col, atom=atom)
        obs = self.env.step(action)
        self._obs = obs
        
        # Cache the structure associated with the best reward so shaping stays coherent.
        step_reward = obs.reward if obs.reward is not None else 0.0
        if step_reward >= self._best_reward:
            self._best_reward = step_reward
            self._best_rows_used, self._best_cols_used = self._count_rows_and_cols(obs.grid)
            self._best_phase = obs.phase

        self.reward = self.compute_episode_reward()
        
        self.done = obs.done
        return self._format_observation(obs)

    def _count_rows_and_cols(self, grid: list[list[str]]) -> tuple[int, int]:
        rows_used = set()
        cols_used = set()
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if cell != ".":
                    rows_used.add(r)
                    cols_used.add(c)
        return len(rows_used), len(cols_used)

    def compute_episode_reward(self) -> float:
        total = max(self._total_actions, 1)
        invalid_ratio = self._invalid_actions / total
        invalid_penalty = 0.3 * invalid_ratio

        row_spread = min(self._best_rows_used / 4.0, 1.0)
        col_spread = min(self._best_cols_used / 4.0, 1.0)
        spatial_bonus = 0.10 * (row_spread * col_spread)

        phase_bonus = 0.0
        if self._best_phase == "crystalline":
            phase_bonus = 0.15
        elif self._best_phase == "polycrystalline":
            phase_bonus = 0.05

        final = self._best_reward + spatial_bonus + phase_bonus - invalid_penalty
        return max(min(final, 1.0), 0.0)

    def _format_observation(self, obs: MaterialForgeObservation) -> str:
        grid_lines = []
        for r, row in enumerate(obs.grid):
            grid_lines.append(f"  {r}: {' '.join(cell if cell != '.' else '.' for cell in row)}")
        
        gaps = ", ".join([f"{p}={obs.current_properties.get(p,0):.1f}/{obs.target.get(p,0):.1f}" for p in PROPERTY_NAMES])
        sb = obs.score_breakdown or {}

        return (
            f"Step {obs.step_number}/{obs.max_steps} | Cost: {obs.total_cost:.0f}/{obs.cost_budget:.0f} | Phase: {obs.phase}\n"
            f"Reward: {obs.reward:.4f} | Properties: {gaps}\n"
            f"Stability: {sb.get('structural_stability',0):.3f} | Order: {sb.get('lattice_order_index',0):.3f}\n"
            f"Grid:\n{chr(10).join(grid_lines)}"
        )


def reward_func(environments, **kwargs) -> list[float]:
    """Episode reward using the best-scoring state plus aligned shaping penalties."""
    return [env.compute_episode_reward() for env in environments]
