from dataclasses import dataclass
from typing import Any

from main import AGENT_IDS, DT, SoccerEnvironment


@dataclass(frozen=True)
class MultiDiscreteSpace:
    nvec: tuple[int, ...]

    def sample(self):
        import random

        return tuple(random.randrange(n) for n in self.nvec)


@dataclass(frozen=True)
class BoxSpace:
    shape: tuple[int, ...]
    low: float
    high: float


class StrikersVsGoalieGymEnv:
    """
    Gym-style wrapper around the local soccer simulator.

    API:
    - reset(seed=None, options=None) -> (observations, info)
    - step(actions) -> (observations, rewards, terminated, truncated, info)
    - render() -> state dict

    `actions` is a dict keyed by `striker_0`, `striker_1`, `goalie_0`.
    Any missing agent action falls back to the scripted policy, which makes
    it easy to train one agent or one team at a time.
    """

    metadata = {"render_modes": ["human", "state_dict"], "render_fps": int(1 / DT)}

    def __init__(self, dt: float = DT):
        self.dt = dt
        self.sim = SoccerEnvironment()
        self.agent_ids = AGENT_IDS
        self.action_space = {agent_id: MultiDiscreteSpace((3, 3, 3)) for agent_id in self.agent_ids}
        self.observation_space = {
            "striker_0": BoxSpace((294,), 0.0, 1.0),
            "striker_1": BoxSpace((294,), 0.0, 1.0),
            "goalie_0": BoxSpace((738,), 0.0, 1.0),
        }

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            import random

            random.seed(seed)

        if options:
            if "ball_scale" in options:
                self.sim.ball_scale = float(options["ball_scale"])
            if "gravity" in options:
                self.sim.gravity = float(options["gravity"])

        self.sim.reset()
        return self.sim.get_observation_dict(), self.sim.get_info_dict()

    def step(self, actions: dict[str, Any] | None = None):
        previous_score = self.sim.score_strikers
        self.sim.step(self.dt, action_overrides=actions or {})
        observations = self.sim.get_observation_dict()
        rewards = self.sim.get_reward_dict()
        terminated = previous_score != self.sim.score_strikers
        truncated = self.sim.last_reset_reason in {"Episode timeout", "Deadlock reset"}
        info = self.sim.get_info_dict()
        info["controlled_agents"] = sorted((actions or {}).keys())
        return observations, rewards, terminated, truncated, info

    def render(self, mode: str = "state_dict"):
        if mode == "state_dict":
            return {
                "agents": {
                    "striker_0": {
                        "position": (self.sim.strikers[0].position.x, self.sim.strikers[0].position.y),
                        "velocity": (self.sim.strikers[0].velocity.x, self.sim.strikers[0].velocity.y),
                    },
                    "striker_1": {
                        "position": (self.sim.strikers[1].position.x, self.sim.strikers[1].position.y),
                        "velocity": (self.sim.strikers[1].velocity.x, self.sim.strikers[1].velocity.y),
                    },
                    "goalie_0": {
                        "position": (self.sim.goalie.position.x, self.sim.goalie.position.y),
                        "velocity": (self.sim.goalie.velocity.x, self.sim.goalie.velocity.y),
                    },
                },
                "ball": {
                    "position": (self.sim.ball.position.x, self.sim.ball.position.y),
                    "velocity": (self.sim.ball.velocity.x, self.sim.ball.velocity.y),
                },
                "info": self.sim.get_info_dict(),
            }
        if mode == "human":
            return self.render("state_dict")
        raise ValueError(f"Unsupported render mode: {mode}")

    def close(self):
        return None
