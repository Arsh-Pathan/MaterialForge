# Reward rubric for the MaterialForge environment.
# Defines the multi-objective reward signal for training the AI agent.

from openenv.core.rubrics import Rubric

try:
    from .config import PROPERTY_NAMES
except ImportError:
    from config import PROPERTY_NAMES


# Holistic reward calculator: balances property matching with structural stability.
class HeuristicRewardRubric(Rubric):
    """Heuristic reward based on property matching, structural stability, and order.

    Reward components:
        - 50% Property Delta Minimization
        - 25% Structural Stability (Coordination Energy)
        - 15% Lattice Order (Positional Entropy)
        - 10% Phase Bonus (Crystalline regularity)
    """

    # Computes a single scalar reward [0, 1] for every action taken by the agent.
    def forward(self, action, observation) -> float:
        """Compute holistic reward from observation metrics."""
        target = observation.target
        current = observation.current_properties
        phase = observation.phase
        total_cost = observation.total_cost
        cost_budget = observation.cost_budget

        # 1. Property Match (50%): Measures how close the current lattice is to the goal.
        mismatches = []
        for prop in PROPERTY_NAMES:
            t = target.get(prop, 0.0)
            c = current.get(prop, 0.0)
            # Normalize delta to a 0-1 match score (using 100 as broad scale)
            mismatches.append(abs(c - t) / 100.0)
        
        avg_mismatch = sum(mismatches) / len(mismatches)
        property_match_score = max(1.0 - avg_mismatch, 0.0)

        # 2. Physics Metrics: Rewards structural integrity (Gibbs stability and order).
        breakdown = observation.score_breakdown
        stability = breakdown.get("structural_stability", 0.0)
        lattice_order = breakdown.get("lattice_order_index", 0.0)

        # 3. Phase Regularity Bonus: Large incentive to reach the "crystalline" state.
        if phase == "crystalline":
            phase_bonus = 1.0
        elif phase == "polycrystalline":
            phase_bonus = 0.5
        else:
            phase_bonus = 0.0

        # 4. Economic Penalty: Strongly discourages exceeding the atom cost budget.
        if cost_budget > 0 and total_cost > cost_budget:
            # Quadratic penalty for budget overruns to encourage efficiency
            cost_penalty = ((total_cost - cost_budget) / cost_budget) ** 2
        else:
            cost_penalty = 0.0

        # Composite Reward Calculation: Final weighted sum returned to the agent.
        reward = (
            0.50 * property_match_score
            + 0.25 * stability
            + 0.15 * lattice_order
            + 0.10 * phase_bonus
            - cost_penalty
        )

        return round(min(max(reward, 0.0), 1.0), 4)
