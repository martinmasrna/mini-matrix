# Mini Matrix

A working miniature of [MatrAIx](https://arxiv.org/abs/2608.04205)'s persona pipeline: build a small population of simulated people, offer each one the red pill or the blue pill, and measure whether the answer depends on which LLM is playing them.

| | |
|---|---|
| `world.json` | the world — 13 traits, 3 hidden dials, dependency graph, hard masks. Data, no logic. |
| `mini_matrix.py` | the machine — sampler, renderer, scene, verifier, harness. Exactly 200 lines. |
| `trials/` | 900 recorded trials: 100 citizens × 3 models × 3 runs each |
| `analyze.py` | recomputes majorities and flips from the trial files |

## Run it

```
python3 mini_matrix.py --dry-run --n 3 --show-hidden
python3 mini_matrix.py --n 100 --models haiku --reps 3 --out trials/trials_haiku.jsonl
python3 analyze.py
```

No API keys: Claude models run through `claude -p`, GPT models through `codex exec` — the subscriptions you already have. Personas are seeded, so same seed, same people; the only variable left is the model.
