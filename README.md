# Xodia — Credenz's RL Bot Competition

**Xodia** is the bot-development event at **Credenz**, PISB's annual technical festival. Participants build intelligent bots, using reinforcement learning or classic algorithms, that compete in a game environment designed by the organisers. Each year has a new game.

## Editions

| Year | Game | Folder |
|---|---|---|
| **2026** | *Stranger Things*: 3 universes, a separate Q-learning agent for each (Find Will, Escape Room, Fight Vecna) | [`2026/`](2026/) |
| **2025** | *Angry Birds*: a physics-based slingshot environment, PPO baseline | [`2025/`](2025/) |
| 2027 | — | 🔜 *Coming soon. Next year's organisers, this is yours* |

Each year's folder is self-contained: game engine, participant template, reference/baseline solution, evaluation runner, and a README explaining how to run it.

## 📤 Organising next year's Xodia? Put it here

```
xodia/
└── 2027/          ← create this folder
    ├── README.md  ← the game, observations/actions/rewards, how to run, how scoring works
    ├── <engine, template, runner, assets ...>
```

1. Create `2027/` and add your code.
2. Add a row to the *Editions* table above.
3. **Do not commit** participants' submissions, per-team results, or anything with participants' names.

New to Git? The step-by-step push guide is here: [**How to push to PISB GitHub**](https://github.com/PICT-IEEE-Student-Branch/.github/blob/main/CONTRIBUTING.md).

## Tips from past organisers

- **The environment design is the event.** Spend most of your time on the observations and the reward shaping, not on the training script.
- Ship a **template plus a fixed runner**, and evaluate every submission on the same seeds. Run each one several times and average, because RL has high variance between seeds.
- Solve your own game with a baseline **before** you release it, so you know it can actually be learned.

---

Part of the [PICT IEEE Student Branch](https://github.com/PICT-IEEE-Student-Branch) · See all [Credenz events](https://github.com/PICT-IEEE-Student-Branch/credenz)
