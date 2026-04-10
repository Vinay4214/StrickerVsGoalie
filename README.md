# Strikers vs. Goalie

A simple 2-vs-1 soccer simulation built in Python.

Two striker agents work together to move the ball into the goal, while one goalie agent tries to block them. The project includes both a playable visual simulation and a Gym-style interface for AI training.

## Requirements

- Python 3.11 or newer recommended
- No external packages required

## Setup

- Two strikers vs one goalie
- Top-down soccer environment
- Multi-agent setup
- Built-in autonomous behavior is included

## Goal

- Strikers: move the ball into the opponent goal
- Goalie: keep the ball out of the goal

## Rewards

- Striker reward:
  - `+1` when the ball enters the goal
  - `-0.001` existential penalty
- Goalie reward:
  - `-1` when the ball enters the goal
  - `+0.001` existential bonus

Note:
- The striker value shown in the game window is the match score.
- The goalie value shown in the game window is the goalie reward, so it can become negative.

## Observation And Actions

- Striker observation space: `294`
- Goalie observation space: `738`
- Striker actions: discrete movement and rotation
- Goalie actions: discrete movement and rotation
- Visual observations: none

## Float Properties

- `ball_scale`
  - Default: `7.5`
  - Recommended range: `4` to `10`
- `gravity`
  - Default: `9.81`
  - Recommended range: `6` to `20`

## Run The Game

```bash
python main.py
```

This opens the visual simulation where the strikers and goalie play automatically.

## Gym-Style Interface

```python
from gym_env import StrikersVsGoalieGymEnv

env = StrikersVsGoalieGymEnv()
obs, info = env.reset()
obs, rewards, terminated, truncated, info = env.step({})
```

Use this interface if you want to connect your own training loop or control one or more agents programmatically.

## Agents

- 2 striker agents
- 1 goalie agent
- Multi-agent setup
- Built-in autonomous behavior is included
- You can also provide your own actions through the Gym-style interface

## End Result
<img width="1356" height="824" alt="image" src="https://github.com/user-attachments/assets/1b77160b-f355-44c3-9c87-e0990191fd87" />
<img width="1362" height="825" alt="image" src="https://github.com/user-attachments/assets/74e12c5f-fdcd-4441-a094-79afdff8865c" />
<img width="1355" height="836" alt="image" src="https://github.com/user-attachments/assets/91315db2-28df-4e37-b05c-d192ce34a050" />
<img width="1365" height="840" alt="image" src="https://github.com/user-attachments/assets/bfd7ce8f-57be-430b-ad8d-76a226d0b75d" />




