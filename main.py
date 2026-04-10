import math
import random
import time
import tkinter as tk
from dataclasses import dataclass, field


WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 680
FIELD_MARGIN = 60
GOAL_WIDTH = 220
MAX_EPISODE_TIME = 90.0
STALL_RESET_TIME = 6.0
DT = 1 / 60

BALL_SCALE_DEFAULT = 7.5
GRAVITY_DEFAULT = 9.81

STRIKER_FORWARD_RAYS = 11
STRIKER_BACK_RAYS = 3
STRIKER_FORWARD_FOV = math.radians(120)
STRIKER_BACK_FOV = math.radians(90)
STRIKER_STACKS = 3
STRIKER_OBJECT_TYPES = ("ball", "ally", "goalie", "goal", "wall")

GOALIE_RAYS = 41
GOALIE_FOV = math.radians(360)
GOALIE_STACKS = 3
GOALIE_OBJECT_TYPES = ("ball", "striker", "goal", "wall")
AGENT_IDS = ("striker_0", "striker_1", "goalie_0")


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def lerp(a, b, t):
    return a + (b - a) * t


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


@dataclass
class Vector2:
    x: float
    y: float

    def copy(self):
        return Vector2(self.x, self.y)

    def length(self):
        return math.hypot(self.x, self.y)

    def normalized(self):
        mag = self.length()
        if mag == 0:
            return Vector2(0, 0)
        return Vector2(self.x / mag, self.y / mag)

    def scale(self, amount):
        return Vector2(self.x * amount, self.y * amount)

    def add(self, other):
        return Vector2(self.x + other.x, self.y + other.y)

    def subtract(self, other):
        return Vector2(self.x - other.x, self.y - other.y)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def rotate(self, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        return Vector2(self.x * c - self.y * s, self.x * s + self.y * c)


@dataclass
class Ball:
    position: Vector2
    velocity: Vector2
    radius: float
    max_speed: float = 480.0

    def update(self, dt, bounds, gravity_strength):
        speed = self.velocity.length()
        if speed > self.max_speed:
            self.velocity = self.velocity.normalized().scale(self.max_speed)

        self.position.x += self.velocity.x * dt
        self.position.y += self.velocity.y * dt

        left, top, right, bottom = bounds
        bounce = clamp(0.72 + ((gravity_strength - 6.0) / 14.0) * 0.24, 0.72, 0.96)

        if self.position.y - self.radius < top:
            self.position.y = top + self.radius
            self.velocity.y = abs(self.velocity.y) * bounce
        if self.position.y + self.radius > bottom:
            self.position.y = bottom - self.radius
            self.velocity.y = -abs(self.velocity.y) * bounce

        if self.position.x - self.radius < left:
            self.position.x = left + self.radius
            self.velocity.x = abs(self.velocity.x) * bounce
        if self.position.x + self.radius > right:
            self.position.x = right - self.radius
            self.velocity.x = -abs(self.velocity.x) * bounce

        drag = clamp(0.997 - ((gravity_strength - 6.0) / 14.0) * 0.002, 0.994, 0.997)
        self.velocity.x *= drag
        self.velocity.y *= drag


@dataclass
class AgentRewardState:
    total_reward: float = 0.0
    existential_rate: float = 0.0

    def step(self, dt):
        self.total_reward += self.existential_rate * dt * 60.0


@dataclass
class SoccerAgent:
    name: str
    role: str
    color: str
    position: Vector2
    facing: float
    radius: float
    speed: float
    turn_speed: float
    kick_strength: float
    reward: AgentRewardState
    observation_stack: list = field(default_factory=list)
    velocity: Vector2 = field(default_factory=lambda: Vector2(0, 0))
    highlight: str = ""

    def forward(self):
        return Vector2(math.cos(self.facing), math.sin(self.facing))

    def right(self):
        return Vector2(-math.sin(self.facing), math.cos(self.facing))

    def update(self, move_x, move_y, turn, dt, bounds):
        target_facing = None
        if move_x != 0 or move_y != 0:
            target_facing = math.atan2(move_y, move_x)
        elif turn != 0:
            target_facing = self.facing + turn * 0.35

        if target_facing is not None:
            angle = math.atan2(math.sin(target_facing - self.facing), math.cos(target_facing - self.facing))
            max_turn = self.turn_speed * dt
            self.facing += clamp(angle, -max_turn, max_turn)

        move = Vector2(move_x, move_y)
        desired = move.normalized().scale(self.speed if move.length() else 0.0)
        self.velocity = Vector2(lerp(self.velocity.x, desired.x, 0.22), lerp(self.velocity.y, desired.y, 0.22))
        self.position.x += self.velocity.x * dt
        self.position.y += self.velocity.y * dt

        left, top, right, bottom = bounds
        self.position.x = clamp(self.position.x, left + self.radius, right - self.radius)
        self.position.y = clamp(self.position.y, top + self.radius, bottom - self.radius)

    def maybe_kick(self, ball):
        offset = ball.position.subtract(self.position)
        if offset.length() <= self.radius + ball.radius + 6:
            # Drive the ball using the agent heading so approach angle matters.
            direction = self.forward()
            if direction.length() == 0:
                direction = offset.normalized()
            ball.velocity = ball.velocity.add(direction.scale(self.kick_strength))


class StrikerBrain:
    def decide(self, agent, env):
        obs = env.get_striker_observation(agent)
        agent.observation_stack.append(obs)
        if len(agent.observation_stack) > STRIKER_STACKS:
            agent.observation_stack.pop(0)

        ball = env.ball.position
        goal = env.goal_center
        teammate = env.strikers[0] if env.strikers[1] is agent else env.strikers[1]

        to_ball = ball.subtract(agent.position)
        to_goal = goal.subtract(agent.position)
        to_teammate = teammate.position.subtract(agent.position)
        agent_ball_dist = to_ball.length()
        teammate_ball_dist = distance(teammate.position, env.ball.position)
        predicted_ball = env.ball.position.add(env.ball.velocity.scale(0.24))
        teammate_has_ball = teammate_ball_dist < teammate.radius + env.ball.radius + 16
        agent_has_ball = agent_ball_dist < agent.radius + env.ball.radius + 16
        goal_from_ball = goal.subtract(env.ball.position).normalized()
        behind_ball_target = env.ball.position.add(goal_from_ball.scale(-(agent.radius + env.ball.radius + 14)))

        primary_chaser = agent_ball_dist <= teammate_ball_dist + 6
        if teammate_has_ball and not agent_has_ball:
            primary_chaser = False
        elif agent_has_ball:
            primary_chaser = True

        if primary_chaser:
            if agent_ball_dist > 60:
                desired = behind_ball_target.subtract(agent.position).normalized()
            else:
                dribble_target = env.ball.position.add(goal_from_ball.scale(36))
                desired = dribble_target.subtract(agent.position).normalized()

            if agent_ball_dist < 90:
                shot_target = goal.add(Vector2(0, (goal.y - agent.position.y) * 0.14))
                desired = desired.add(shot_target.subtract(agent.position).normalized().scale(0.6)).normalized()

            if to_teammate.length() < 95:
                desired = desired.add(to_teammate.normalized().scale(-0.9)).normalized()
        else:
            lane_sign = -1 if agent.position.y <= goal.y else 1
            if abs(agent.position.y - teammate.position.y) < 50:
                lane_sign = -1 if teammate.position.y >= goal.y else 1

            support_target = Vector2(
                clamp(env.ball.position.x - 150, env.field_left + 80, env.field_right - 220),
                clamp(env.ball.position.y + lane_sign * 95, env.field_top + 55, env.field_bottom - 55),
            )

            if teammate_has_ball:
                support_target = Vector2(
                    clamp(teammate.position.x - 125, env.field_left + 80, env.field_right - 240),
                    clamp(teammate.position.y + lane_sign * 135, env.field_top + 55, env.field_bottom - 55),
                )

            desired = support_target.subtract(agent.position).normalized()

            if to_teammate.length() < 110:
                desired = desired.add(to_teammate.normalized().scale(-1.0)).normalized()

            pass_lane = teammate.position.subtract(agent.position).normalized()
            desired = desired.add(pass_lane.scale(0.08)).normalized()

            if teammate_ball_dist > 130 and agent_ball_dist < teammate_ball_dist - 12:
                desired = predicted_ball.subtract(agent.position).normalized()

        return self._discrete_actions(agent, desired)

    def _discrete_actions(self, agent, desired):
        angle = math.atan2(desired.y, desired.x) - agent.facing
        angle = math.atan2(math.sin(angle), math.cos(angle))
        move_x = 1 if desired.x > 0.12 else -1 if desired.x < -0.12 else 0
        move_y = 1 if desired.y > 0.12 else -1 if desired.y < -0.12 else 0
        turn = 1 if angle > 0.15 else -1 if angle < -0.15 else 0
        return move_x, move_y, turn


class GoalieBrain:
    def decide(self, agent, env):
        obs = env.get_goalie_observation(agent)
        agent.observation_stack.append(obs)
        if len(agent.observation_stack) > GOALIE_STACKS:
            agent.observation_stack.pop(0)

        ball = env.ball.position
        target_x = env.field_right - 85
        predicted_y = ball.y + env.ball.velocity.y * 0.22
        desired_target = Vector2(target_x, clamp(predicted_y, env.field_top + 45, env.field_bottom - 45))

        if ball.x > env.field_right - 220:
            desired_target.x = env.field_right - 58

        desired = desired_target.subtract(agent.position).normalized()
        if ball.x > env.field_right - 220 or distance(ball, agent.position) < 140:
            desired = ball.subtract(agent.position).normalized()

        return self._discrete_actions(agent, desired)

    def _discrete_actions(self, agent, desired):
        angle = math.atan2(desired.y, desired.x) - agent.facing
        angle = math.atan2(math.sin(angle), math.cos(angle))
        move_x = 1 if desired.x > 0.1 else -1 if desired.x < -0.1 else 0
        move_y = 1 if desired.y > 0.1 else -1 if desired.y < -0.1 else 0
        turn = 1 if angle > 0.12 else -1 if angle < -0.12 else 0
        return move_x, move_y, turn


class SoccerEnvironment:
    def __init__(self):
        self.field_left = FIELD_MARGIN
        self.field_top = FIELD_MARGIN
        self.field_right = WINDOW_WIDTH - FIELD_MARGIN
        self.field_bottom = WINDOW_HEIGHT - FIELD_MARGIN
        self.goal_center = Vector2(self.field_right, WINDOW_HEIGHT / 2)
        self.ball_scale = BALL_SCALE_DEFAULT
        self.gravity = GRAVITY_DEFAULT
        self.score_strikers = 0
        self.score_goalie = 0
        self.episode_time = 0.0
        self.stall_time = 0.0
        self.last_goal_flash = 0.0
        self.last_ball_position = Vector2(WINDOW_WIDTH * 0.42, WINDOW_HEIGHT / 2)
        self.last_progress_ball_position = Vector2(WINDOW_WIDTH * 0.42, WINDOW_HEIGHT / 2)
        self.last_reset_reason = "Kickoff"
        self.last_step_rewards = {agent_id: 0.0 for agent_id in AGENT_IDS}
        self.striker_brain = StrikerBrain()
        self.goalie_brain = GoalieBrain()
        self.reset()

    def reset(self):
        self.episode_time = 0.0
        self.stall_time = 0.0
        self.last_goal_flash = 0.0
        self.last_reset_reason = "Kickoff"
        self.last_step_rewards = {agent_id: 0.0 for agent_id in AGENT_IDS}
        self.ball = Ball(
            position=Vector2(WINDOW_WIDTH * 0.42, WINDOW_HEIGHT / 2),
            velocity=Vector2(random.uniform(-50, 50), random.uniform(-30, 30)),
            radius=self.ball_scale * 1.6,
        )
        self.last_ball_position = self.ball.position.copy()
        self.last_progress_ball_position = self.ball.position.copy()
        self.strikers = [
            SoccerAgent(
                name="Striker A",
                role="striker",
                color="#ff9f1c",
                position=Vector2(WINDOW_WIDTH * 0.23, WINDOW_HEIGHT / 2 - 120),
                facing=0.0,
                radius=18,
                speed=180,
                turn_speed=4.7,
                kick_strength=110,
                reward=AgentRewardState(existential_rate=-0.001),
            ),
            SoccerAgent(
                name="Striker B",
                role="striker",
                color="#ffbf69",
                position=Vector2(WINDOW_WIDTH * 0.23, WINDOW_HEIGHT / 2 + 120),
                facing=0.0,
                radius=18,
                speed=180,
                turn_speed=4.7,
                kick_strength=110,
                reward=AgentRewardState(existential_rate=-0.001),
            ),
        ]
        self.goalie = SoccerAgent(
            name="Goalie",
            role="goalie",
            color="#2ec4b6",
            position=Vector2(WINDOW_WIDTH * 0.84, WINDOW_HEIGHT / 2),
            facing=math.pi,
            radius=22,
            speed=205,
            turn_speed=5.4,
            kick_strength=125,
            reward=AgentRewardState(existential_rate=0.001),
        )

    def step(self, dt, action_overrides=None):
        self.episode_time += dt
        bounds = (self.field_left, self.field_top, self.field_right, self.field_bottom)
        self.last_step_rewards = {agent_id: 0.0 for agent_id in AGENT_IDS}

        # External agent actions are integrated here on each environment step.
        for index, striker in enumerate(self.strikers):
            agent_id = AGENT_IDS[index]
            move_x, move_y, turn = self.resolve_action(agent_id, striker, action_overrides)
            striker.update(move_x, move_y, turn, dt, bounds)
            striker.maybe_kick(self.ball)
            striker.reward.step(dt)
            self.last_step_rewards[agent_id] += striker.reward.existential_rate * dt * 60.0
            striker.highlight = f"a=({move_x},{move_y},{turn})"

        move_x, move_y, turn = self.resolve_action("goalie_0", self.goalie, action_overrides)
        self.goalie.update(move_x, move_y, turn, dt, bounds)
        self.goalie.maybe_kick(self.ball)
        self.goalie.reward.step(dt)
        self.last_step_rewards["goalie_0"] += self.goalie.reward.existential_rate * dt * 60.0
        self.goalie.highlight = f"a=({move_x},{move_y},{turn})"

        self.resolve_collisions()
        self.ball.radius = self.ball_scale * 1.6
        self.ball.update(dt, bounds, self.gravity)

        self.update_stall_state(dt)
        scored = self.check_goal()
        reset_reason = None
        if scored:
            reset_reason = "Goal scored"
        elif self.episode_time >= MAX_EPISODE_TIME:
            reset_reason = "Episode timeout"
        elif self.stall_time >= STALL_RESET_TIME:
            reset_reason = "Deadlock reset"

        if reset_reason:
            self.last_reset_reason = reset_reason
            self.reset_positions()

    def resolve_action(self, agent_id, agent, action_overrides):
        # Gym/RL agents can override scripted behavior by supplying an action for this agent_id.
        if action_overrides and agent_id in action_overrides:
            return self.decode_action(action_overrides[agent_id])

        # If no external action is provided, fall back to the built-in scripted policy.
        if agent.role == "goalie":
            return self.goalie_brain.decide(agent, self)
        return self.striker_brain.decide(agent, self)

    def decode_action(self, action):
        if isinstance(action, int):
            move_x = (action % 3) - 1
            move_y = ((action // 3) % 3) - 1
            turn = ((action // 9) % 3) - 1
            return move_x, move_y, turn

        if isinstance(action, (tuple, list)) and len(action) == 3:
            return tuple(int(clamp(v, -1, 1)) for v in action)

        if isinstance(action, dict):
            return (
                int(clamp(action.get("move_x", 0), -1, 1)),
                int(clamp(action.get("move_y", 0), -1, 1)),
                int(clamp(action.get("turn", 0), -1, 1)),
            )

        raise ValueError("Action must be an int in [0, 26], a length-3 tuple/list, or a dict with move_x/move_y/turn.")

    def update_stall_state(self, dt):
        ball_delta = distance(self.ball.position, self.last_ball_position)
        progress_delta = distance(self.ball.position, self.last_progress_ball_position)
        x_progress = abs(self.ball.position.x - self.last_progress_ball_position.x)

        if progress_delta > 36 or x_progress > 24:
            self.stall_time = 0.0
            self.last_progress_ball_position = self.ball.position.copy()
        elif ball_delta < 1.0:
            self.stall_time += dt
        else:
            self.stall_time += dt * 0.25

        self.last_ball_position = self.ball.position.copy()

    def resolve_collisions(self):
        agents = self.strikers + [self.goalie]
        for agent in agents:
            delta = self.ball.position.subtract(agent.position)
            dist = delta.length()
            min_dist = self.ball.radius + agent.radius
            if 0 < dist < min_dist:
                push = delta.normalized().scale(min_dist - dist + 0.5)
                self.ball.position = self.ball.position.add(push)
                impact = delta.normalized()
                relative_speed = self.ball.velocity.subtract(agent.velocity).dot(impact)
                rebound = max(70, abs(relative_speed) + agent.kick_strength * 0.55)
                self.ball.velocity = self.ball.velocity.add(impact.scale(rebound))
                self.ball.velocity = self.ball.velocity.add(agent.velocity.scale(0.35))

        for i, first in enumerate(agents):
            for second in agents[i + 1 :]:
                delta = second.position.subtract(first.position)
                dist = delta.length()
                min_dist = first.radius + second.radius
                if 0 < dist < min_dist:
                    normal = delta.normalized()
                    push = normal.scale((min_dist - dist) / 2 + 0.5)
                    first.position = first.position.add(push.scale(-1))
                    second.position = second.position.add(push)

                    relative_velocity = second.velocity.subtract(first.velocity)
                    separation_speed = relative_velocity.dot(normal)
                    bounce_strength = 45 if first.role == "striker" and second.role == "striker" else 28

                    if separation_speed < bounce_strength:
                        impulse = normal.scale((bounce_strength - separation_speed) / 2)
                        first.velocity = first.velocity.add(impulse.scale(-1))
                        second.velocity = second.velocity.add(impulse)

    def check_goal(self):
        goal_top = WINDOW_HEIGHT / 2 - GOAL_WIDTH / 2
        goal_bottom = WINDOW_HEIGHT / 2 + GOAL_WIDTH / 2
        crossed = self.ball.position.x + self.ball.radius >= self.field_right
        in_window = goal_top <= self.ball.position.y <= goal_bottom

        if crossed and in_window:
            self.score_strikers += 1
            for index, striker in enumerate(self.strikers):
                striker.reward.total_reward += 1.0
                self.last_step_rewards[AGENT_IDS[index]] += 1.0
            self.goalie.reward.total_reward -= 1.0
            self.last_step_rewards["goalie_0"] -= 1.0
            self.last_goal_flash = time.time()
            return True
        return False

    def reset_positions(self):
        self.ball.position = Vector2(WINDOW_WIDTH * 0.42, WINDOW_HEIGHT / 2)
        self.ball.velocity = Vector2(random.uniform(-80, 40), random.uniform(-45, 45))
        self.strikers[0].position = Vector2(WINDOW_WIDTH * 0.23, WINDOW_HEIGHT / 2 - 120)
        self.strikers[1].position = Vector2(WINDOW_WIDTH * 0.23, WINDOW_HEIGHT / 2 + 120)
        for striker in self.strikers:
            striker.velocity = Vector2(0, 0)
            striker.facing = 0.0
        self.goalie.position = Vector2(WINDOW_WIDTH * 0.84, WINDOW_HEIGHT / 2)
        self.goalie.velocity = Vector2(0, 0)
        self.goalie.facing = math.pi
        self.episode_time = 0.0
        self.stall_time = 0.0
        self.last_ball_position = self.ball.position.copy()
        self.last_progress_ball_position = self.ball.position.copy()

    def get_striker_observation(self, agent):
        rays = []
        rays.extend(self._ray_bundle(agent, STRIKER_FORWARD_RAYS, STRIKER_FORWARD_FOV, 0.0, STRIKER_OBJECT_TYPES, include_alignment=True))
        rays.extend(self._ray_bundle(agent, STRIKER_BACK_RAYS, STRIKER_BACK_FOV, math.pi, STRIKER_OBJECT_TYPES, include_alignment=True))
        return rays

    def get_goalie_observation(self, agent):
        return self._ray_bundle(agent, GOALIE_RAYS, GOALIE_FOV, 0.0, GOALIE_OBJECT_TYPES, include_alignment=True)

    def get_stacked_observation(self, agent):
        if agent.role == "goalie":
            base_obs = self.get_goalie_observation(agent)
            stack_size = GOALIE_STACKS
        else:
            base_obs = self.get_striker_observation(agent)
            stack_size = STRIKER_STACKS

        history = list(agent.observation_stack)
        history.append(base_obs)
        history = history[-stack_size:]

        while len(history) < stack_size:
            history.insert(0, [0.0] * len(base_obs))

        stacked = []
        for frame in history:
            stacked.extend(frame)
        return stacked

    def get_observation_dict(self):
        return {
            "striker_0": self.get_stacked_observation(self.strikers[0]),
            "striker_1": self.get_stacked_observation(self.strikers[1]),
            "goalie_0": self.get_stacked_observation(self.goalie),
        }

    def get_reward_dict(self):
        return dict(self.last_step_rewards)

    def get_info_dict(self):
        return {
            "score_strikers": self.score_strikers,
            "score_goalie": self.score_goalie,
            "episode_time": self.episode_time,
            "stall_time": self.stall_time,
            "last_reset_reason": self.last_reset_reason,
            "ball_position": (self.ball.position.x, self.ball.position.y),
            "ball_velocity": (self.ball.velocity.x, self.ball.velocity.y),
        }

    def _ray_bundle(self, agent, count, fov, offset, object_types, include_alignment=False):
        if count == 1:
            angles = [agent.facing + offset]
        else:
            start = agent.facing + offset - fov / 2
            step = fov / (count - 1)
            angles = [start + i * step for i in range(count)]

        max_distance = 260.0
        result = []
        for angle in angles:
            direction = Vector2(math.cos(angle), math.sin(angle))
            encoded = [0.0] * (len(object_types) + 1 + (1 if include_alignment else 0))
            hit_type, hit_distance = self._cast_ray(agent, direction, max_distance, object_types)
            if hit_type in object_types:
                encoded[object_types.index(hit_type)] = 1.0
            encoded[-1] = hit_distance / max_distance
            if include_alignment:
                encoded[-2] = max(0.0, direction.dot(self.ball.velocity.normalized())) if self.ball.velocity.length() else 0.0
            result.extend(encoded)
        return result

    def _cast_ray(self, agent, direction, max_distance, object_types):
        samples = [
            ("ball", self.ball.position),
            ("goal", Vector2(self.field_right, WINDOW_HEIGHT / 2)),
            ("wall", Vector2(clamp(agent.position.x + direction.x * max_distance, self.field_left, self.field_right),
                             clamp(agent.position.y + direction.y * max_distance, self.field_top, self.field_bottom))),
        ]

        for striker in self.strikers:
            if striker is not agent:
                samples.append(("ally" if agent.role == "striker" else "striker", striker.position))
        if agent.role == "striker":
            samples.append(("goalie", self.goalie.position))

        best_type = "wall"
        best_dist = max_distance
        for obj_type, pos in samples:
            if obj_type not in object_types:
                continue
            to_obj = pos.subtract(agent.position)
            proj = to_obj.dot(direction)
            if 0 < proj < best_dist:
                lateral = abs(direction.x * to_obj.y - direction.y * to_obj.x)
                if lateral < 26:
                    best_type = obj_type
                    best_dist = proj
        return best_type, best_dist


class GameApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Strikers vs. Goalie")
        self.root.configure(bg="#08131c")
        self.env = SoccerEnvironment()

        self.canvas = tk.Canvas(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, bg="#0c4a3a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        controls = tk.Frame(root, bg="#08131c", padx=12, pady=10)
        controls.pack(fill="x")

        self.ball_scale_var = tk.DoubleVar(value=BALL_SCALE_DEFAULT)
        self.gravity_var = tk.DoubleVar(value=GRAVITY_DEFAULT)
        self.status_var = tk.StringVar(value="Autonomous simulation running")

        tk.Label(controls, text="ball_scale", fg="#d8f3dc", bg="#08131c").pack(side="left")
        tk.Scale(
            controls,
            from_=4.0,
            to=10.0,
            resolution=0.1,
            orient="horizontal",
            variable=self.ball_scale_var,
            command=self.on_ball_scale,
            length=180,
            bg="#08131c",
            fg="#d8f3dc",
            troughcolor="#184e77",
            highlightthickness=0,
        ).pack(side="left", padx=(8, 20))

        tk.Label(controls, text="gravity", fg="#d8f3dc", bg="#08131c").pack(side="left")
        tk.Scale(
            controls,
            from_=6.0,
            to=20.0,
            resolution=0.1,
            orient="horizontal",
            variable=self.gravity_var,
            command=self.on_gravity,
            length=180,
            bg="#08131c",
            fg="#d8f3dc",
            troughcolor="#184e77",
            highlightthickness=0,
        ).pack(side="left", padx=(8, 20))

        tk.Button(controls, text="Reset Match", command=self.reset_match, bg="#ffd166", fg="#08131c").pack(side="left")
        tk.Label(controls, textvariable=self.status_var, fg="#d8f3dc", bg="#08131c").pack(side="right")

        self.last_time = time.time()
        self.running = True
        self.loop()

    def on_ball_scale(self, _value):
        self.env.ball_scale = self.ball_scale_var.get()

    def on_gravity(self, _value):
        self.env.gravity = self.gravity_var.get()

    def reset_match(self):
        self.env.score_strikers = 0
        self.env.score_goalie = 0
        self.env.reset()

    def loop(self):
        now = time.time()
        dt = min(now - self.last_time, 0.03)
        self.last_time = now
        if self.running:
            self.env.step(dt if dt > 0 else DT)
            self.draw()
        self.root.after(int(DT * 1000), self.loop)

    def draw(self):
        c = self.canvas
        c.delete("all")

        c.create_rectangle(FIELD_MARGIN, FIELD_MARGIN, WINDOW_WIDTH - FIELD_MARGIN, WINDOW_HEIGHT - FIELD_MARGIN, fill="#157a5f", outline="#d8f3dc", width=3)
        c.create_line(WINDOW_WIDTH / 2, FIELD_MARGIN, WINDOW_WIDTH / 2, WINDOW_HEIGHT - FIELD_MARGIN, fill="#d8f3dc", dash=(12, 8), width=2)
        c.create_oval(WINDOW_WIDTH / 2 - 85, WINDOW_HEIGHT / 2 - 85, WINDOW_WIDTH / 2 + 85, WINDOW_HEIGHT / 2 + 85, outline="#d8f3dc", width=2)

        goal_top = WINDOW_HEIGHT / 2 - GOAL_WIDTH / 2
        goal_bottom = WINDOW_HEIGHT / 2 + GOAL_WIDTH / 2
        c.create_rectangle(self.env.field_right, goal_top, self.env.field_right + 24, goal_bottom, fill="#dde5b6", outline="#d8f3dc", width=3)

        flash = time.time() - self.env.last_goal_flash < 0.35
        if flash:
            c.create_rectangle(FIELD_MARGIN, FIELD_MARGIN, WINDOW_WIDTH - FIELD_MARGIN, WINDOW_HEIGHT - FIELD_MARGIN, outline="#ffe66d", width=8)

        for striker in self.env.strikers:
            self.draw_agent(striker)
        self.draw_agent(self.env.goalie)

        ball = self.env.ball
        c.create_oval(
            ball.position.x - ball.radius,
            ball.position.y - ball.radius,
            ball.position.x + ball.radius,
            ball.position.y + ball.radius,
            fill="#faf3dd",
            outline="#1d3557",
            width=2,
        )

        c.create_text(130, 28, text=f"Strikers {self.env.score_strikers}", fill="#ffbf69", font=("Segoe UI", 19, "bold"))
        c.create_text(WINDOW_WIDTH - 120, 28, text=f"Goalie {-self.env.goalie.reward.total_reward:.2f}", fill="#2ec4b6", font=("Segoe UI", 16, "bold"))
        c.create_text(WINDOW_WIDTH / 2, 28, text=f"Episode {self.env.episode_time:04.1f}s", fill="#f1faee", font=("Segoe UI", 15, "bold"))
        c.create_text(WINDOW_WIDTH / 2, 52, text=f"{self.env.last_reset_reason} | stall {self.env.stall_time:0.1f}/{STALL_RESET_TIME:.0f}s", fill="#d8f3dc", font=("Segoe UI", 10, "bold"))

        info_y = WINDOW_HEIGHT - 24
        striker_total = sum(agent.reward.total_reward for agent in self.env.strikers)
        c.create_text(180, info_y, text=f"Team reward {striker_total:.3f}", fill="#f1faee", font=("Consolas", 12))
        c.create_text(WINDOW_WIDTH / 2, info_y, text=f"Obs sizes: striker {len(self.env.get_striker_observation(self.env.strikers[0])) * STRIKER_STACKS}, goalie {len(self.env.get_goalie_observation(self.env.goalie)) * GOALIE_STACKS}", fill="#f1faee", font=("Consolas", 12))
        c.create_text(WINDOW_WIDTH - 200, info_y, text=f"ball_scale={self.env.ball_scale:.1f} gravity={self.env.gravity:.1f}", fill="#f1faee", font=("Consolas", 12))

        self.status_var.set(f"Autonomous simulation running | {self.env.last_reset_reason}")

    def draw_agent(self, agent):
        c = self.canvas
        c.create_oval(
            agent.position.x - agent.radius,
            agent.position.y - agent.radius,
            agent.position.x + agent.radius,
            agent.position.y + agent.radius,
            fill=agent.color,
            outline="#f1faee",
            width=2,
        )
        heading = agent.forward().scale(agent.radius + 12)
        c.create_line(agent.position.x, agent.position.y, agent.position.x + heading.x, agent.position.y + heading.y, fill="#08131c", width=3)
        c.create_text(agent.position.x, agent.position.y - agent.radius - 14, text=agent.name, fill="#f1faee", font=("Segoe UI", 11, "bold"))
        c.create_text(agent.position.x, agent.position.y + agent.radius + 12, text=agent.highlight, fill="#e0fbfc", font=("Consolas", 9))


def main():
    root = tk.Tk()
    GameApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
