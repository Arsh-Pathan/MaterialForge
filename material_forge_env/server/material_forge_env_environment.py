"""MaterialForge Environment Implementation.

An RL environment where an agent designs atomic crystal structures on an 8x8
lattice to match target material properties.
"""

import random
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..environment.config import PROPERTY_NAMES
    from ..environment.lattice import Lattice
    from ..models import MaterialForgeAction, MaterialForgeObservation
    from ..environment.physics import (
        classify_phase,
        compute_lattice_quality,
        compute_stability,
        estimate_properties,
    )
    from ..scenarios.scenarios import generate_scenario
except ImportError:
    from environment.config import PROPERTY_NAMES
    from environment.lattice import Lattice
    from models import MaterialForgeAction, MaterialForgeObservation
    from environment.physics import (
        classify_phase,
        compute_lattice_quality,
        compute_stability,
        estimate_properties,
    )
    from scenarios.scenarios import generate_scenario


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

    def step(self, action: MaterialForgeAction) -> MaterialForgeObservation:  # type: ignore[override]
        """Execute one step: apply action, compute properties, score, check done."""
        self._state.step_count += 1

        # Apply action to lattice
        success = self._apply_action(action)

        # Compute current state
        properties = estimate_properties(self._lattice)
        phase = classify_phase(self._lattice)
        stability = compute_stability(self._lattice)
        quality = compute_lattice_quality(self._lattice)

        # Check termination
        done = self._check_done(properties)

        # Build observation with score breakdown
        score_breakdown = {
            "stability": stability,
            "lattice_quality": quality,
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
            },
        )

        # Apply rubric to get reward
        obs.reward = self._apply_rubric(action, obs)

        return obs

    def _apply_rubric(self, action: MaterialForgeAction, obs: MaterialForgeObservation) -> float:
        """Compute the scalar reward using the lab's scientific rubric.
        
        Formula: R = 0.5*match + 0.25*stability + 0.15*quality + 0.1*phase - cost_pen
        """
        # 1. Property Match (0.0 to 1.0)
        diffs = []
        for p in obs.target:
            tgt = max(obs.target[p], 1.0)
            curr = obs.current_properties.get(p, 0.0)
            diffs.append(abs(curr - tgt) / tgt)
        
        property_match = max(0.0, 1.0 - (sum(diffs) / max(len(diffs), 1)))
        
        # 2. Structural Features
        sb = obs.score_breakdown
        stability = sb.get("stability", 0.0)
        quality = sb.get("lattice_quality", 0.0)
        
        # 3. Phase Bonus
        phase_bonus = 0.0
        if obs.phase == "crystalline": phase_bonus = 1.0
        elif obs.phase == "polycrystalline": phase_bonus = 0.5
        
        # 4. Cost Offset
        cost_penalty = 0.0
        if obs.total_cost > obs.cost_budget:
            cost_penalty = (obs.total_cost - obs.cost_budget) / obs.cost_budget
            
        # Composite Reward calculation
        reward = (
            0.50 * property_match + 
            0.25 * stability + 
            0.15 * quality + 
            0.10 * phase_bonus - 
            0.10 * cost_penalty
        )
        
        final_r = max(0.01, float(reward)) # MINIMUM REWARD OF 0.01 FOR VISIBILITY
        
        print(f"DEBUG: Step {obs.step_number} | Action: {action.action_type} | Match: {property_match:.3f} | Rwd: {final_r:.4f}")
        return final_r

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

    def _check_done(self, properties: dict) -> bool:
        """Check if episode should end."""
        # Step limit
        if self._state.step_count >= self._max_steps:
            return True

        # All properties within tolerance
        if self._lattice.atom_count() > 0:
            all_within = all(
                abs(properties.get(p, 0.0) - self._target.get(p, 0.0))
                <= self._tolerance
                for p in PROPERTY_NAMES
            )
            if all_within:
                return True

        return False

    def _build_observation(self, done: bool, reward: float) -> MaterialForgeObservation:
        """Build an observation from current state."""
        properties = estimate_properties(self._lattice)
        phase = classify_phase(self._lattice)
        stability = compute_stability(self._lattice)
        quality = compute_lattice_quality(self._lattice)

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
                "stability": stability,
                "lattice_quality": quality,
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
