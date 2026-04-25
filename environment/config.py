"""Constants and configuration for the MaterialForge environment."""

# Simulation grid dimensions and default episode constraints.
GRID_SIZE = 8
MAX_STEPS = 50
COST_BUDGET_DEFAULT = 80
EMPTY = "."

# Completion requirements: minimal structure size and required crystal quality.
MIN_ATOMS_FOR_COMPLETION = 6
VALID_COMPLETION_PHASES = {"crystalline", "polycrystalline"}

# The four macro-physical properties the agent is tasked to engineer.
PROPERTY_NAMES = ["hardness", "conductivity", "thermal_resistance", "elasticity"]

# Atom definitions: defines species symbols, their resource costs, and physical contributions.
# Each species is a SPECIALIST in one physical property to force chemical diversity.
ATOM_TYPES = {
    "A": {
        "name": "metal",
        "cost": 8,
        "contributions": {
            "hardness": 1.00,           # Master of Hardness
            "conductivity": 0.15,
            "thermal_resistance": 0.10,
            "elasticity": 0.05,
        },
    },
    "B": {
        "name": "conductor",
        "cost": 6,
        "contributions": {
            "hardness": 0.10,
            "conductivity": 1.00,       # Master of Conductivity
            "thermal_resistance": 0.10,
            "elasticity": 0.20,
        },
    },
    "C": {
        "name": "ceramic",
        "cost": 4,
        "contributions": {
            "hardness": 0.30,
            "conductivity": 0.05,
            "thermal_resistance": 1.00, # Master of Thermal Resistance
            "elasticity": 0.05,
        },
    },
    "P": {
        "name": "polymer",
        "cost": 2,
        "contributions": {
            "hardness": 0.05,
            "conductivity": 0.15,
            "thermal_resistance": 0.15,
            "elasticity": 1.00,         # Master of Elasticity
        },
    },
}

ATOM_SYMBOLS = list(ATOM_TYPES.keys())

# Difficulty presets: adjusts matching tolerance and budget for different hackathon scenarios.
DIFFICULTY_PRESETS = {
    "easy": {"tolerance": 20, "cost_budget": 120, "max_steps": 64},
    "medium": {"tolerance": 10, "cost_budget": 80, "max_steps": 50},
    "hard": {"tolerance": 5, "cost_budget": 60, "max_steps": 40},
}
