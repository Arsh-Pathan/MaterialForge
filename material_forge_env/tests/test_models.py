"""Tests for the Pydantic data models."""

import pytest
from pydantic import ValidationError

from material_forge_env.models import MaterialForgeAction, MaterialForgeObservation


class TestMaterialForgeAction:
    def test_place_action(self):
        a = MaterialForgeAction(action_type="place", row=0, col=0, atom="A")
        assert a.action_type == "place"
        assert a.row == 0
        assert a.col == 0
        assert a.atom == "A"

    def test_replace_action(self):
        a = MaterialForgeAction(action_type="replace", row=3, col=5, atom="B")
        assert a.action_type == "replace"

    def test_remove_action(self):
        a = MaterialForgeAction(action_type="remove", row=7, col=7)
        assert a.action_type == "remove"
        assert a.atom is None

    def test_invalid_action_type(self):
        with pytest.raises(ValidationError):
            MaterialForgeAction(action_type="invalid", row=0, col=0)

    def test_row_out_of_range(self):
        with pytest.raises(ValidationError):
            MaterialForgeAction(action_type="place", row=8, col=0, atom="A")

    def test_negative_row(self):
        with pytest.raises(ValidationError):
            MaterialForgeAction(action_type="place", row=-1, col=0, atom="A")

    def test_invalid_atom(self):
        with pytest.raises(ValidationError):
            MaterialForgeAction(action_type="place", row=0, col=0, atom="X")

    def test_serialization(self):
        a = MaterialForgeAction(action_type="place", row=0, col=0, atom="A")
        d = a.model_dump()
        assert d["action_type"] == "place"
        assert d["row"] == 0


class TestMaterialForgeObservation:
    def test_observation_creation(self):
        obs = MaterialForgeObservation(
            grid=[["." for _ in range(8)] for _ in range(8)],
            target={"hardness": 50, "conductivity": 50, "thermal_resistance": 50, "elasticity": 50},
            current_properties={"hardness": 0, "conductivity": 0, "thermal_resistance": 0, "elasticity": 0},
            phase="amorphous",
            total_cost=0.0,
            cost_budget=80.0,
            step_number=0,
            max_steps=50,
            score_breakdown={"stability": 0.0, "lattice_quality": 0.0},
            done=False,
            reward=0.0,
        )
        assert obs.phase == "amorphous"
        assert obs.step_number == 0
        assert obs.done is False
