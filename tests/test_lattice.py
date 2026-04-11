"""Tests for the Lattice grid engine."""

import pytest

from material_forge_env.environment.config import GRID_SIZE, ATOM_TYPES, EMPTY
from material_forge_env.environment.lattice import Lattice


class TestLatticeInit:
    def test_default_size(self):
        lat = Lattice()
        assert lat.size == GRID_SIZE

    def test_custom_size(self):
        lat = Lattice(4)
        assert lat.size == 4

    def test_empty_grid(self):
        lat = Lattice()
        grid = lat.get_grid()
        for row in grid:
            assert all(cell == EMPTY for cell in row)

    def test_atom_count_empty(self):
        lat = Lattice()
        assert lat.atom_count() == 0

    def test_total_cost_empty(self):
        lat = Lattice()
        assert lat.total_cost() == 0.0


class TestPlace:
    def test_place_valid(self):
        lat = Lattice()
        assert lat.place(0, 0, "A") is True
        assert lat.get(0, 0) == "A"

    def test_place_all_atom_types(self):
        lat = Lattice()
        for i, sym in enumerate(ATOM_TYPES):
            assert lat.place(0, i, sym) is True
            assert lat.get(0, i) == sym

    def test_place_occupied_fails(self):
        lat = Lattice()
        lat.place(0, 0, "A")
        assert lat.place(0, 0, "B") is False

    def test_place_out_of_bounds(self):
        lat = Lattice()
        assert lat.place(-1, 0, "A") is False
        assert lat.place(0, GRID_SIZE, "A") is False
        assert lat.place(GRID_SIZE, 0, "A") is False

    def test_place_invalid_atom(self):
        lat = Lattice()
        assert lat.place(0, 0, "X") is False
        assert lat.place(0, 0, ".") is False


class TestReplace:
    def test_replace_valid(self):
        lat = Lattice()
        lat.place(0, 0, "A")
        assert lat.replace(0, 0, "B") is True
        assert lat.get(0, 0) == "B"

    def test_replace_empty_fails(self):
        lat = Lattice()
        assert lat.replace(0, 0, "A") is False

    def test_replace_same_atom_fails(self):
        lat = Lattice()
        lat.place(0, 0, "A")
        assert lat.replace(0, 0, "A") is False

    def test_replace_out_of_bounds(self):
        lat = Lattice()
        assert lat.replace(-1, 0, "A") is False


class TestRemove:
    def test_remove_valid(self):
        lat = Lattice()
        lat.place(0, 0, "A")
        assert lat.remove(0, 0) is True
        assert lat.get(0, 0) == EMPTY

    def test_remove_empty_fails(self):
        lat = Lattice()
        assert lat.remove(0, 0) is False

    def test_remove_out_of_bounds(self):
        lat = Lattice()
        assert lat.remove(-1, 0) is False


class TestCountAndCost:
    def test_count_atoms(self):
        lat = Lattice()
        lat.place(0, 0, "A")
        lat.place(0, 1, "A")
        lat.place(1, 0, "B")
        counts = lat.count_atoms()
        assert counts["A"] == 2
        assert counts["B"] == 1
        assert counts["C"] == 0
        assert counts["P"] == 0

    def test_atom_count(self):
        lat = Lattice()
        lat.place(0, 0, "A")
        lat.place(1, 0, "B")
        assert lat.atom_count() == 2

    def test_total_cost(self):
        lat = Lattice()
        lat.place(0, 0, "A")  # cost 8
        lat.place(0, 1, "P")  # cost 2
        assert lat.total_cost() == 10.0


class TestNeighbors:
    def test_corner_neighbors(self):
        lat = Lattice()
        neighbors = lat.get_neighbors(0, 0)
        assert len(neighbors) == 3  # only 3 neighbors for a corner

    def test_edge_neighbors(self):
        lat = Lattice()
        neighbors = lat.get_neighbors(0, 3)
        assert len(neighbors) == 5

    def test_center_neighbors(self):
        lat = Lattice()
        neighbors = lat.get_neighbors(3, 3)
        assert len(neighbors) == 8


class TestCloneAndFromGrid:
    def test_clone_independent(self):
        lat = Lattice()
        lat.place(0, 0, "A")
        clone = lat.clone()
        clone.remove(0, 0)
        assert lat.get(0, 0) == "A"
        assert clone.get(0, 0) == EMPTY

    def test_from_grid(self):
        lat = Lattice()
        lat.place(0, 0, "A")
        lat.place(1, 1, "B")
        grid = lat.get_grid()
        restored = Lattice.from_grid(grid)
        assert restored.get(0, 0) == "A"
        assert restored.get(1, 1) == "B"
        assert restored.atom_count() == 2
