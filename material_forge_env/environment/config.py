"""Constants and configuration for the MaterialForge environment."""

GRID_SIZE = 8
MAX_STEPS = 50
COST_BUDGET_DEFAULT = 80
EMPTY = "."

PROPERTY_NAMES = ["hardness", "conductivity", "thermal_resistance", "elasticity"]

# Atom definitions: symbol -> (category, cost, property contributions)
# Property contributions are base weights for [hardness, conductivity, thermal_resistance, elasticity]
ATOM_TYPES = {
    "A": {
        "name": "metal",
        "cost": 8,
        "contributions": {
            "hardness": 0.85,
            "conductivity": 0.40,
            "thermal_resistance": 0.30,
            "elasticity": 0.10,
        },
    },
    "B": {
        "name": "conductor",
        "cost": 6,
        "contributions": {
            "hardness": 0.20,
            "conductivity": 0.90,
            "thermal_resistance": 0.15,
            "elasticity": 0.25,
        },
    },
    "C": {
        "name": "ceramic",
        "cost": 4,
        "contributions": {
            "hardness": 0.60,
            "conductivity": 0.05,
            "thermal_resistance": 0.85,
            "elasticity": 0.05,
        },
    },
    "P": {
        "name": "polymer",
        "cost": 2,
        "contributions": {
            "hardness": 0.10,
            "conductivity": 0.10,
            "thermal_resistance": 0.20,
            "elasticity": 0.85,
        },
    },
}

ATOM_SYMBOLS = list(ATOM_TYPES.keys())

# Difficulty presets: tolerance is max acceptable deviation per property
DIFFICULTY_PRESETS = {
    "easy": {"tolerance": 20, "cost_budget": 120, "max_steps": 64},
    "medium": {"tolerance": 10, "cost_budget": 80, "max_steps": 50},
    "hard": {"tolerance": 5, "cost_budget": 60, "max_steps": 40},
}
