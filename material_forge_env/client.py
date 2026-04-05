"""MaterialForge Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from material_forge_env.models import MaterialForgeAction, MaterialForgeObservation


class MaterialForgeEnv(EnvClient[MaterialForgeAction, MaterialForgeObservation, State]):
    """Client for the MaterialForge Environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated environment session.

    Example:
        >>> with MaterialForgeEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.grid)
        ...     action = MaterialForgeAction(action_type="place", row=0, col=0, atom="A")
        ...     result = client.step(action)
        ...     print(result.observation.current_properties)
    """

    def _step_payload(self, action: MaterialForgeAction) -> Dict:
        """Convert MaterialForgeAction to JSON payload for step message."""
        payload = {
            "action_type": action.action_type,
            "row": action.row,
            "col": action.col,
        }
        if action.atom is not None:
            payload["atom"] = action.atom
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[MaterialForgeObservation]:
        """Parse server response into StepResult[MaterialForgeObservation]."""
        obs_data = payload.get("observation", {})
        observation = MaterialForgeObservation(
            grid=obs_data.get("grid", []),
            target=obs_data.get("target", {}),
            current_properties=obs_data.get("current_properties", {}),
            phase=obs_data.get("phase", "amorphous"),
            total_cost=obs_data.get("total_cost", 0.0),
            cost_budget=obs_data.get("cost_budget", 80.0),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 50),
            score_breakdown=obs_data.get("score_breakdown", {}),
            hint=obs_data.get("hint"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
