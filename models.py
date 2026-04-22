"""Data models for the MaterialForge environment."""

from typing import Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field

try:
    from .environment.config import GRID_SIZE
except ImportError:
    from environment.config import GRID_SIZE


# Standardized Action schema: Defines how the agent interacts with the 8x8 lattice.
class MaterialForgeAction(Action):
    """Action to manipulate atoms on the crystal lattice."""

    action_type: Literal["place", "replace", "remove"] = Field(
        ..., description="Type of lattice operation"
    )
    row: int = Field(..., ge=0, lt=GRID_SIZE, description="Row index on the grid")
    col: int = Field(..., ge=0, lt=GRID_SIZE, description="Column index on the grid")
    atom: Optional[Literal["A", "B", "C", "P"]] = Field(
        default=None,
        description="Atom type (required for place/replace, ignored for remove)",
    )


# Standardized Observation schema: Packages the physical state and goals for the agent.
class MaterialForgeObservation(Observation):
    """Observation from the MaterialForge environment after each step."""

    grid: List[List[str]] = Field(description="8x8 lattice grid state")
    target: Dict[str, float] = Field(description="Target property values to achieve")
    current_properties: Dict[str, float] = Field(
        description="Current estimated material properties"
    )
    phase: str = Field(description="Crystal phase classification")
    total_cost: float = Field(description="Total cost of atoms on the grid")
    cost_budget: float = Field(description="Maximum cost budget for this episode")
    step_number: int = Field(description="Current step number")
    max_steps: int = Field(description="Maximum steps allowed")
    score_breakdown: Dict[str, float] = Field(
        description="Breakdown of reward components"
    )
    hint: Optional[str] = Field(
        default=None, description="Optional hint from LLM engine"
    )
