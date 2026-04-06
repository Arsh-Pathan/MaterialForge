"""Tests for the physics estimation engine."""

import pytest

from material_forge_env.environment.config import PROPERTY_NAMES
from material_forge_env.environment.lattice import Lattice
from material_forge_env.environment.physics import (
    classify_phase,
    compute_lattice_quality,
    compute_stability,
    estimate_properties,
)


class TestEstimateProperties:
    def test_empty_lattice_returns_zeros(self):
        lat = Lattice()
        props = estimate_properties(lat)
        for p in PROPERTY_NAMES:
            assert props[p] == 0.0

    def test_all_properties_in_range(self):
        lat = Lattice()
        for r in range(4):
            for c in range(4):
                lat.place(r, c, "A")
        props = estimate_properties(lat)
        for p in PROPERTY_NAMES:
            assert 0.0 <= props[p] <= 100.0

    def test_metal_heavy_boosts_hardness(self):
        lat = Lattice()
        for r in range(4):
            for c in range(4):
                lat.place(r, c, "A")
        props = estimate_properties(lat)
        # Metal atoms should make hardness the dominant property
        assert props["hardness"] > props["elasticity"]

    def test_conductor_heavy_boosts_conductivity(self):
        lat = Lattice()
        for r in range(4):
            for c in range(4):
                lat.place(r, c, "B")
        props = estimate_properties(lat)
        assert props["conductivity"] > props["hardness"]

    def test_ceramic_boosts_thermal_resistance(self):
        lat = Lattice()
        for r in range(4):
            for c in range(4):
                lat.place(r, c, "C")
        props = estimate_properties(lat)
        assert props["thermal_resistance"] > props["conductivity"]

    def test_polymer_boosts_elasticity(self):
        lat = Lattice()
        for r in range(4):
            for c in range(4):
                lat.place(r, c, "P")
        props = estimate_properties(lat)
        assert props["elasticity"] > props["hardness"]

    def test_properties_change_with_atoms(self):
        lat = Lattice()
        lat.place(0, 0, "A")
        props1 = estimate_properties(lat)
        lat.place(0, 1, "A")
        props2 = estimate_properties(lat)
        # Adding more metal should change properties
        assert props1 != props2


class TestClassifyPhase:
    def test_empty_is_amorphous(self):
        lat = Lattice()
        assert classify_phase(lat) == "amorphous"

    def test_few_atoms_amorphous(self):
        lat = Lattice()
        lat.place(0, 0, "A")
        lat.place(4, 4, "B")
        assert classify_phase(lat) == "amorphous"

    def test_regular_pattern_crystalline(self):
        lat = Lattice()
        # Fill every other cell with A in a regular pattern
        for r in range(8):
            for c in range(8):
                lat.place(r, c, "A")
        phase = classify_phase(lat)
        assert phase == "crystalline"

    def test_returns_valid_phase(self):
        lat = Lattice()
        for r in range(3):
            for c in range(3):
                lat.place(r, c, ["A", "B", "C", "P"][(r + c) % 4])
        phase = classify_phase(lat)
        assert phase in ("crystalline", "polycrystalline", "amorphous")


class TestComputeStability:
    def test_empty_lattice_zero(self):
        lat = Lattice()
        assert compute_stability(lat) == 0.0

    def test_stability_in_range(self):
        lat = Lattice()
        lat.place(0, 0, "A")
        lat.place(0, 1, "A")
        s = compute_stability(lat)
        assert 0.0 <= s <= 1.0

    def test_cluster_more_stable_than_isolated(self):
        # Clustered atoms
        lat_cluster = Lattice()
        for r in range(3):
            for c in range(3):
                lat_cluster.place(r, c, "A")
        s_cluster = compute_stability(lat_cluster)

        # Isolated atoms
        lat_iso = Lattice()
        lat_iso.place(0, 0, "A")
        lat_iso.place(0, 7, "A")
        lat_iso.place(7, 0, "A")
        lat_iso.place(7, 7, "A")
        s_iso = compute_stability(lat_iso)

        assert s_cluster > s_iso


class TestComputeLatticeQuality:
    def test_empty_lattice_zero(self):
        lat = Lattice()
        assert compute_lattice_quality(lat) == 0.0

    def test_quality_in_range(self):
        lat = Lattice()
        for r in range(4):
            for c in range(4):
                lat.place(r, c, "A")
        q = compute_lattice_quality(lat)
        assert 0.0 <= q <= 1.0
