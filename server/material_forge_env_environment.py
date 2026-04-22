# MaterialForge Environment Implementation.
# This class acts as the OpenEnv-compliant wrapper for our scientific simulation.

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# Direct imports from root-level modules/packages
from environment.config import (
    MIN_ATOMS_FOR_COMPLETION,
    PROPERTY_NAMES,
    VALID_COMPLETION_PHASES,
)
from environment.lattice import Lattice
from models import MaterialForgeAction, MaterialForgeObservation
from environment.physics import (
    classify_phase,
    compute_lattice_order,
    compute_structural_stability,
    estimate_properties,
)
from scenarios.scenarios import generate_scenario


# Main environment class: bridges the abstract OpenEnv protocol with our lattice physics.
class MaterialForgeEnvironment(Environment):
    """RL environment for designing crystal structures on a lattice grid.

    The agent places, replaces, or removes atoms on an 8x8 grid to achieve
    target material properties (hardness, conductivity, thermal_resistance,
    elasticity). Episodes end when max_steps is reached or all properties
    are within tolerance of the target.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, rubric=None):
        super().__init__(rubric=rubric)
        self._lattice = Lattice()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._scenario = {}
        self._target = {}
        self._cost_budget = 80.0
        self._tolerance = 10.0
        self._max_steps = 50
        self._difficulty = "medium"

    # Reset function: triggers a new scenario and initializes the physics lattice.
    def reset(self, seed: int | None = None, **kwargs) -> MaterialForgeObservation:
        """Reset environment with a new scenario.

        Args:
            seed: Optional random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)

        difficulty = kwargs.get("difficulty", "medium")
        scenario_name = kwargs.get("scenario_name", None)
        self._scenario = generate_scenario(difficulty=difficulty, name=scenario_name)
        self._target = self._scenario["target"]
        self._cost_budget = float(self._scenario["cost_budget"])
        self._tolerance = float(self._scenario["tolerance"])
        self._max_steps = int(self._scenario["max_steps"])
        self._difficulty = self._scenario["difficulty"]

        self._lattice = Lattice()
        self._state = State(episode_id=str(uuid4()), step_count=0)

        self._reset_rubric()

        return self._build_observation(done=False, reward=0.0)

    # Core execution step: applies the agent's action and computes the resulting physics state.
    def step(self, action: MaterialForgeAction) -> MaterialForgeObservation:  # type: ignore[override]
        """Execute one step: apply action, compute properties, score, check done."""
        self._state.step_count += 1

        # Apply action to lattice
        success = self._apply_action(action)

        # Compute current state via physics heuristics
        properties = estimate_properties(self._lattice)
        phase = classify_phase(self._lattice)
        stability = compute_structural_stability(self._lattice)
        order = compute_lattice_order(self._lattice)

        # Check termination
        done = self._check_done(properties, phase)

        # Build observation with advanced scientific score breakdown
        score_breakdown = {
            "structural_stability": stability,
            "lattice_order_index": order,
            "action_valid": 1.0 if success else 0.0,
        }

        obs = MaterialForgeObservation(
            grid=self._lattice.get_grid(),
            target=self._target,
            current_properties=properties,
            phase=phase,
            total_cost=self._lattice.total_cost(),
            cost_budget=self._cost_budget,
            step_number=self._state.step_count,
            max_steps=self._max_steps,
            score_breakdown=score_breakdown,
            done=done,
            reward=0.0,  # placeholder, rubric will compute
            metadata={
                "difficulty": self._difficulty,
                "scenario_name": self._scenario.get("name"),
                "tolerance": self._tolerance,
                "action_success": success,
                "phase_id": phase,
            },
        )

        # Apply rubric to get reward
        obs.reward = self._apply_rubric(action, obs)

        return obs

    # Internal helper to mutate the lattice state.
    def _apply_action(self, action: MaterialForgeAction) -> bool:
        """Apply the action to the lattice. Returns True if action was valid."""
        if action.action_type == "place":
            if action.atom is None:
                return False
            return self._lattice.place(action.row, action.col, action.atom)
        elif action.action_type == "replace":
            if action.atom is None:
                return False
            return self._lattice.replace(action.row, action.col, action.atom)
        elif action.action_type == "remove":
            return self._lattice.remove(action.row, action.col)
        return False

    # Evaluation function: checks if the material has reached a stable crystalline phase.
    def _check_done(self, properties: dict, phase: str) -> bool:
        """Check if episode should end based on target matching and structural validity."""
        if self._state.step_count >= self._max_steps:
            return True

        atom_count = self._lattice.atom_count()

        # Success requires minimal structure size and ordered phase
        if atom_count >= MIN_ATOMS_FOR_COMPLETION and phase in VALID_COMPLETION_PHASES:
            all_within = all(
                abs(properties.get(p, 0.0) - self._target.get(p, 0.0))
                <= self._tolerance
                for p in PROPERTY_NAMES
            )
            if all_within:
                return True

        return False

    # State serialization: packages the current grid and metrics into a standardized object.
    def _build_observation(self, done: bool, reward: float) -> MaterialForgeObservation:
        """Build an observation from the current lattice state."""
        properties = estimate_properties(self._lattice)
        phase = classify_phase(self._lattice)
        stability = compute_structural_stability(self._lattice)
        order = compute_lattice_order(self._lattice)

        return MaterialForgeObservation(
            grid=self._lattice.get_grid(),
            target=self._target,
            current_properties=properties,
            phase=phase,
            total_cost=self._lattice.total_cost(),
            cost_budget=self._cost_budget,
            step_number=self._state.step_count,
            max_steps=self._max_steps,
            score_breakdown={
                "structural_stability": stability,
                "lattice_order_index": order,
            },
            done=done,
            reward=reward,
            metadata={
                "difficulty": self._difficulty,
                "scenario_name": self._scenario.get("name"),
                "tolerance": self._tolerance,
            },
        )

    @property
    def state(self) -> State:
        return self._state
