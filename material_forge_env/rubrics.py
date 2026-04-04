"""Reward rubric for the MaterialForge environment."""

from openenv.core.rubrics import Rubric

from .config import PROPERTY_NAMES


class HeuristicRewardRubric(Rubric):
    """Heuristic reward based on property matching, stability, quality, and phase.

    Reward formula:
        reward = 0.50 * property_match
               + 0.25 * stability
               + 0.15 * lattice_quality
               + 0.10 * phase_bonus
               - cost_penalty
    Clamped to [0.0, 1.0].
    """

    def forward(self, action, observation) -> float:
        """Compute reward from the observation's score breakdown and properties."""
        target = observation.target
        current = observation.current_properties
        phase = observation.phase
        total_cost = observation.total_cost
        cost_budget = observation.cost_budget

        # Property match: 1 - mean absolute deviation (normalized to 0-1)
        deviations = []
        for prop in PROPERTY_NAMES:
            t = target.get(prop, 0.0)
            c = current.get(prop, 0.0)
            deviations.append(abs(c - t) / 100.0)
        property_match = max(1.0 - (sum(deviations) / len(deviations)), 0.0)

        # Stability and quality from score_breakdown (pre-computed in environment step)
        breakdown = observation.score_breakdown
        stability = breakdown.get("stability", 0.0)
        lattice_quality = breakdown.get("lattice_quality", 0.0)

        # Phase bonus
        if phase == "crystalline":
            phase_bonus = 1.0
        elif phase == "polycrystalline":
            phase_bonus = 0.5
        else:
            phase_bonus = 0.0

        # Cost penalty: over-budget penalty
        if cost_budget > 0 and total_cost > cost_budget:
            cost_penalty = (total_cost - cost_budget) / cost_budget
        else:
            cost_penalty = 0.0

        reward = (
            0.50 * property_match
            + 0.25 * stability
            + 0.15 * lattice_quality
            + 0.10 * phase_bonus
            - cost_penalty
        )

        return round(min(max(reward, 0.0), 1.0), 4)
