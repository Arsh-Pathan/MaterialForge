"""Tests for scenario generation."""

import pytest

from material_forge_env.environment.config import PROPERTY_NAMES
from material_forge_env.scenarios.scenarios import (
    NAMED_SCENARIOS,
    generate_scenario,
    get_random_scenario,
)


class TestGenerateScenario:
    def test_returns_required_keys(self):
        s = generate_scenario("easy")
        assert "target" in s
        assert "cost_budget" in s
        assert "tolerance" in s
        assert "max_steps" in s
        assert "difficulty" in s

    def test_target_has_all_properties(self):
        s = generate_scenario("medium")
        for prop in PROPERTY_NAMES:
            assert prop in s["target"]
            assert 0.0 <= s["target"][prop] <= 100.0

    def test_easy_preset(self):
        s = generate_scenario("easy")
        assert s["tolerance"] == 20
        assert s["cost_budget"] == 120
        assert s["max_steps"] == 64

    def test_medium_preset(self):
        s = generate_scenario("medium")
        assert s["tolerance"] == 10
        assert s["cost_budget"] == 80
        assert s["max_steps"] == 50

    def test_hard_preset(self):
        s = generate_scenario("hard")
        assert s["tolerance"] == 5
        assert s["cost_budget"] == 60

    def test_named_scenario(self):
        s = generate_scenario("easy", name="diamond-like")
        assert s["target"]["hardness"] == 90.0
        assert s["name"] == "diamond-like"

    def test_all_named_scenarios(self):
        for name in NAMED_SCENARIOS:
            s = generate_scenario("medium", name=name)
            assert s["name"] == name
            assert s["target"] == NAMED_SCENARIOS[name]


class TestGetRandomScenario:
    def test_returns_valid_scenario(self):
        s = get_random_scenario()
        assert "target" in s
        assert s["difficulty"] in ("easy", "medium", "hard")

    def test_multiple_calls_vary(self):
        results = set()
        for _ in range(20):
            s = get_random_scenario()
            results.add(s["difficulty"])
        # With 20 calls, should hit more than one difficulty
        assert len(results) >= 2
