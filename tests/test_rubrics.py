"""Tests for the reward rubric."""

import pytest

from material_forge_env.environment.rubrics import HeuristicRewardRubric
from material_forge_env.models import MaterialForgeAction, MaterialForgeObservation


def _make_obs(
    target=None,
    current=None,
    phase="amorphous",
    total_cost=0.0,
    cost_budget=80.0,
    stability=0.5,
    quality=0.5,
):
    """Helper to build a test observation."""
    if target is None:
        target = {"hardness": 50, "conductivity": 50, "thermal_resistance": 50, "elasticity": 50}
    if current is None:
        current = {"hardness": 50, "conductivity": 50, "thermal_resistance": 50, "elasticity": 50}
    return MaterialForgeObservation(
        grid=[["." for _ in range(8)] for _ in range(8)],
        target=target,
        current_properties=current,
        phase=phase,
        total_cost=total_cost,
        cost_budget=cost_budget,
        step_number=1,
        max_steps=50,
        score_breakdown={"stability": stability, "lattice_quality": quality, "action_valid": 1.0},
        done=False,
        reward=0.0,
    )


def _make_action():
    return MaterialForgeAction(action_type="place", row=0, col=0, atom="A")


class TestHeuristicRewardRubric:
    def test_perfect_match(self):
        rubric = HeuristicRewardRubric()
        obs = _make_obs(stability=1.0, quality=1.0, phase="crystalline")
        reward = rubric.forward(_make_action(), obs)
        # Perfect property match (0.5) + stability (0.25) + quality (0.15) + phase (0.10)
        assert reward == 1.0

    def test_zero_match(self):
        rubric = HeuristicRewardRubric()
        target = {"hardness": 100, "conductivity": 100, "thermal_resistance": 100, "elasticity": 100}
        current = {"hardness": 0, "conductivity": 0, "thermal_resistance": 0, "elasticity": 0}
        obs = _make_obs(target=target, current=current, stability=0.0, quality=0.0)
        reward = rubric.forward(_make_action(), obs)
        assert reward == 0.0

    def test_reward_in_range(self):
        rubric = HeuristicRewardRubric()
        obs = _make_obs()
        reward = rubric.forward(_make_action(), obs)
        assert 0.0 <= reward <= 1.0

    def test_cost_penalty(self):
        rubric = HeuristicRewardRubric()
        obs_under = _make_obs(total_cost=50.0, cost_budget=80.0)
        obs_over = _make_obs(total_cost=160.0, cost_budget=80.0)
        r_under = rubric.forward(_make_action(), obs_under)
        r_over = rubric.forward(_make_action(), obs_over)
        assert r_under > r_over

    def test_phase_bonus(self):
        rubric = HeuristicRewardRubric()
        obs_cryst = _make_obs(phase="crystalline")
        obs_amor = _make_obs(phase="amorphous")
        r_cryst = rubric.forward(_make_action(), obs_cryst)
        r_amor = rubric.forward(_make_action(), obs_amor)
        assert r_cryst > r_amor
