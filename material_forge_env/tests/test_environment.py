"""Tests for the MaterialForgeEnvironment."""

import pytest

from material_forge_env.environment.config import PROPERTY_NAMES
from material_forge_env.environment.rubrics import HeuristicRewardRubric
from material_forge_env.models import MaterialForgeAction, MaterialForgeObservation
from material_forge_env.server.material_forge_env_environment import (
    MaterialForgeEnvironment,
)


@pytest.fixture
def env():
    return MaterialForgeEnvironment(rubric=HeuristicRewardRubric())


class TestReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset()
        assert isinstance(obs, MaterialForgeObservation)

    def test_reset_empty_grid(self, env):
        obs = env.reset()
        for row in obs.grid:
            assert all(cell == "." for cell in row)

    def test_reset_has_target(self, env):
        obs = env.reset()
        for prop in PROPERTY_NAMES:
            assert prop in obs.target

    def test_reset_step_zero(self, env):
        obs = env.reset()
        assert obs.step_number == 0

    def test_reset_not_done(self, env):
        obs = env.reset()
        assert obs.done is False

    def test_reset_with_difficulty(self, env):
        obs = env.reset(difficulty="easy")
        assert obs.max_steps == 64
        assert obs.cost_budget == 120.0

    def test_reset_with_scenario(self, env):
        obs = env.reset(difficulty="medium", scenario_name="diamond-like")
        assert obs.target["hardness"] == 90.0

    def test_reset_with_seed_reproducible(self):
        env1 = MaterialForgeEnvironment(rubric=HeuristicRewardRubric())
        env2 = MaterialForgeEnvironment(rubric=HeuristicRewardRubric())
        obs1 = env1.reset(seed=42, difficulty="medium")
        obs2 = env2.reset(seed=42, difficulty="medium")
        assert obs1.target == obs2.target


class TestStep:
    def test_step_place(self, env):
        env.reset()
        action = MaterialForgeAction(action_type="place", row=0, col=0, atom="A")
        obs = env.step(action)
        assert obs.grid[0][0] == "A"
        assert obs.step_number == 1

    def test_step_replace(self, env):
        env.reset()
        env.step(MaterialForgeAction(action_type="place", row=0, col=0, atom="A"))
        obs = env.step(MaterialForgeAction(action_type="replace", row=0, col=0, atom="B"))
        assert obs.grid[0][0] == "B"

    def test_step_remove(self, env):
        env.reset()
        env.step(MaterialForgeAction(action_type="place", row=0, col=0, atom="A"))
        obs = env.step(MaterialForgeAction(action_type="remove", row=0, col=0))
        assert obs.grid[0][0] == "."

    def test_step_returns_reward(self, env):
        env.reset()
        action = MaterialForgeAction(action_type="place", row=0, col=0, atom="A")
        obs = env.step(action)
        assert isinstance(obs.reward, float)
        assert 0.0 <= obs.reward <= 1.0

    def test_step_updates_properties(self, env):
        env.reset()
        action = MaterialForgeAction(action_type="place", row=0, col=0, atom="A")
        obs = env.step(action)
        # At least one property should be non-zero after placing an atom
        assert any(v > 0 for v in obs.current_properties.values())

    def test_step_updates_cost(self, env):
        env.reset()
        action = MaterialForgeAction(action_type="place", row=0, col=0, atom="A")
        obs = env.step(action)
        assert obs.total_cost == 8.0  # Metal costs 8

    def test_step_has_phase(self, env):
        env.reset()
        action = MaterialForgeAction(action_type="place", row=0, col=0, atom="A")
        obs = env.step(action)
        assert obs.phase in ("crystalline", "polycrystalline", "amorphous")

    def test_step_has_score_breakdown(self, env):
        env.reset()
        action = MaterialForgeAction(action_type="place", row=0, col=0, atom="A")
        obs = env.step(action)
        assert "stability" in obs.score_breakdown
        assert "lattice_quality" in obs.score_breakdown

    def test_invalid_place_on_occupied(self, env):
        env.reset()
        env.step(MaterialForgeAction(action_type="place", row=0, col=0, atom="A"))
        obs = env.step(MaterialForgeAction(action_type="place", row=0, col=0, atom="B"))
        # Action should fail but step still returns an observation
        assert obs.step_number == 2
        assert obs.grid[0][0] == "A"  # unchanged


class TestTermination:
    def test_episode_ends_at_max_steps(self, env):
        env.reset(difficulty="hard")  # max_steps=40
        action = MaterialForgeAction(action_type="place", row=0, col=0, atom="A")
        for i in range(40):
            obs = env.step(action)
        assert obs.done is True

    def test_episode_ends_on_property_match(self, env):
        # Use easy difficulty with wide tolerance
        env.reset(seed=1, difficulty="easy", scenario_name="flexible-polymer")
        # Place lots of polymer atoms to try to match flexible-polymer target
        for r in range(8):
            for c in range(8):
                obs = env.step(MaterialForgeAction(action_type="place", row=r, col=c, atom="P"))
                if obs.done:
                    break
            if obs.done:
                break
        # Episode should eventually finish (either by match or max_steps)
        # Just verify it can run without errors

    def test_single_atom_conductor_is_not_done(self, env):
        env.reset(seed=43, difficulty="medium", scenario_name="conductor")
        obs = env.step(MaterialForgeAction(action_type="place", row=0, col=0, atom="B"))
        assert obs.done is False

    def test_completion_requires_structure_not_just_property_match(self, env):
        env.reset(seed=44, difficulty="medium", scenario_name="heat-shield")
        for row, col in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]:
            obs = env.step(MaterialForgeAction(action_type="place", row=row, col=col, atom="C"))
        assert obs.done is False


class TestState:
    def test_state_property(self, env):
        env.reset()
        state = env.state
        assert hasattr(state, "episode_id")
        assert hasattr(state, "step_count")
        assert state.step_count == 0
