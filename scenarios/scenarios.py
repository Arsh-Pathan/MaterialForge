"""Target scenario generation for the MaterialForge environment."""

import random
from typing import Dict

try:
    from ..environment.config import DIFFICULTY_PRESETS, PROPERTY_NAMES
except ImportError:
    from environment.config import DIFFICULTY_PRESETS, PROPERTY_NAMES


# Defined named tasks representing specific material science goals.
NAMED_SCENARIOS: Dict[str, Dict[str, float]] = {
    "diamond-like": {
        "hardness": 90.0,
        "conductivity": 30.0,
        "thermal_resistance": 60.0,
        "elasticity": 10.0,
    },
    "conductor": {
        "hardness": 25.0,
        "conductivity": 90.0,
        "thermal_resistance": 20.0,
        "elasticity": 30.0,
    },
    "heat-shield": {
        "hardness": 50.0,
        "conductivity": 10.0,
        "thermal_resistance": 90.0,
        "elasticity": 15.0,
    },
    "flexible-polymer": {
        "hardness": 15.0,
        "conductivity": 20.0,
        "thermal_resistance": 25.0,
        "elasticity": 85.0,
    },
    "balanced-alloy": {
        "hardness": 55.0,
        "conductivity": 50.0,
        "thermal_resistance": 50.0,
        "elasticity": 45.0,
    },
}

# Plausible property ranges per difficulty: scales target complexity.
_RANGES = {
    "easy": (25.0, 75.0),
    "medium": (15.0, 85.0),
    "hard": (10.0, 95.0),
}


# Primary logic for creating a new task episode: defines the "What" for the agent.
def generate_scenario(difficulty: str = "medium", name: str | None = None) -> Dict:
    """Generate a scenario with target properties and constraints.

    Args:
        difficulty: One of "easy", "medium", "hard".
        name: Optional named scenario (e.g. "diamond-like"). Overrides random generation.

    Returns:
        Dict with keys: target, cost_budget, tolerance, max_steps, difficulty, name.
    """
    preset = DIFFICULTY_PRESETS.get(difficulty, DIFFICULTY_PRESETS["medium"])

    # Retrieves a fixed scientific profile or generates a random target vector.
    if name and name in NAMED_SCENARIOS:
        target = dict(NAMED_SCENARIOS[name])
    else:
        lo, hi = _RANGES.get(difficulty, _RANGES["medium"])
        target = {prop: round(random.uniform(lo, hi), 1) for prop in PROPERTY_NAMES}
        name = None

    return {
        "target": target,
        "cost_budget": preset["cost_budget"],
        "tolerance": preset["tolerance"],
        "max_steps": preset["max_steps"],
        "difficulty": difficulty,
        "name": name,
    }


# Randomized selector for benchmarking the agent across various task complexities.
def get_random_scenario() -> Dict:
    """Pick a random difficulty and optionally a named scenario."""
    difficulty = random.choice(["easy", "medium", "hard"])

    # 30% chance of using a named scenario for specific scientific testing.
    if random.random() < 0.3:
        name = random.choice(list(NAMED_SCENARIOS.keys()))
        return generate_scenario(difficulty=difficulty, name=name)

    return generate_scenario(difficulty=difficulty)
