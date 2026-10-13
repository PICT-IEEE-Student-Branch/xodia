# Xodia @ Credenz '25 — Angry Birds RL Competition

**Xodia** is the AI/RL game-bot event at **Credenz**, the PICT IEEE Student Branch's annual technical fest. This repository holds the complete **Credenz 2025** edition: a physics-based *Angry Birds* environment in which participants trained reinforcement-learning agents to launch birds at pigs. It also has the organisers' baseline PPO agent and the contest evaluation harness.

> Part of [`xodia`](../README.md), one folder per year. Credenz '26 Xodia (Stranger Things, multi-universe Q-learning) is in [`2026/`](../2026/). All Credenz events are listed in [the org page](https://github.com/PICT-IEEE-Student-Branch).

---

## Game overview

- A bird is launched from a **slingshot** toward a **target pig**.
- Its trajectory is shaped by **gravity**.
- The agent controls **launch power** along x and y.

### Observation space (8-dim)

| Feature | Description | Range |
|---|---|---|
| `bird_x` | Bird x-coordinate | `[0, 800]` |
| `bird_y` | Bird y-coordinate | `[0, 450]` |
| `velocity_x` | Bird x-velocity | `[-30, 30]` |
| `velocity_y` | Bird y-velocity | `[-30, 30]` |
| `pig_x` | Pig x-coordinate | `[0, 800]` |
| `pig_y` | Pig y-coordinate | `[0, 450]` |
| `max_height` | Maximum height reached | `[0, 450]` |
| `launch_angle` | Launch angle (radians) | `[-π, π]` |

### Action space

| Action | Description | Range |
|---|---|---|
| `power_x` | Launch power, x-direction (farther) | `[1, 15]` |
| `power_y` | Launch power, y-direction (higher arc) | `[1, 15]` |

### Rewards

- **+200** for hitting the pig
- **+0.5 per unit** decrease in distance to the pig
- **Bonus** for interesting trajectories
- **−5** for going out of bounds
- An episode ends on a pig hit, on going out of bounds, or after **200 steps**

---

## Repository structure

```
xodia/2025/
├── angry_birds_core.py           # Gymnasium environment (Bird, Pig, AngryBirdsEnv), the one participants train on
├── train.py                      # PPO baseline: train_model() + run_game() pygame playback
├── rl-agent-2.py                 # Self-contained earlier variant (env + training in one file)
├── angry_birds_model_v2.2.zip    # Trained PPO baseline (Stable-Baselines3)
├── Contest/
│   ├── angry_birds_environment.py   # Participant skeleton: physics given; observation_space / _get_obs / reward left for teams to design
│   └── game_levels_environment.py   # Multi-level evaluation harness: scores a participant's saved model
├── UI/
│   ├── main.py                   # Standalone pygame game UI
│   └── images/                   # Sprites + backgrounds (bird, pig, sling, blocks, Credenz logo)
└── requirements.txt
```

## Quick start

Run everything from inside this `2025/` folder (asset paths are relative to it): `cd 2025`.

```bash
pip install -r requirements.txt

# Watch the bundled baseline agent play
python train.py

# Train your own agent: in train.py, uncomment `model = train_model()` in __main__
# (PPO, 8 parallel envs, 600k timesteps)
```

**Evaluating a submission** (organisers): `Contest/game_levels_environment.py` loads a participant's Stable-Baselines3 model with `test_participant_model(model_path)` and reports average/max/min score and success rate per level.

## Reusing this for a future Xodia

- **Environment design is the event.** The difficulty lies in the observation and reward design, not in the training script. Change the reward shaping and the observation vector before you change anything else.
- Give participants a template plus a fixed evaluation harness, and evaluate every submission on identical seeds and levels.
- Test the baseline agent end to end before releasing the problem, so you know the task can actually be learned.

## Notes

- Original development history: [`malharinamdar/credenz-xodia`](https://github.com/malharinamdar/credenz-xodia).
- Fixes made in this org copy (verified with Stable-Baselines3's `check_env` and by loading the bundled baseline):
  - The env imports were moved from legacy `gym` to `gymnasium`, because current Stable-Baselines3 requires it. The env already used the Gymnasium 5-tuple `step` API.
  - `_get_obs()` returned 10 values while `observation_space` and the trained baseline both expect 8. The two extra `dx, dy` entries were removed, so the env now matches the table above.
  - A hard-coded local asset path in `UI/main.py` was made relative.
  - After the fix, the bundled baseline hits the pig in 18 of 20 seeded test episodes.
- Known quirk (left as is, because the baseline was trained with it): bird velocity can exceed the declared `±30` bounds. PPO doesn't clip observations to the Box, so training is unaffected, but `check_env` will warn.

## Credits

Built for Credenz '25 by the PISB Xodia team: Malhar Inamdar and Advait Joshi.
