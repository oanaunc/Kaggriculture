# ARC-AGI-3 results log

Public 25 games, one run each, same model (Qwen3.8 Flash Next NVFP4), 132 min/game, 28 concurrent.

| Run | Date | Mean score | Levels | Zero-score games | Notes |
|---|---|---|---|---|---|
| Public Duck A (wuliao0) | 2026-09-18 | 5.78 | 35 | 5 | reference |
| Public Duck B (keithtyser) | 2026-09-01 | 6.76 | 36 | 6 | reference |
| Ours v1 | 2026-09-27 | 6.25 | 33 | 5 | game-over fix, animation evidence, note parsing, context budget, level memory |
| Ours v2 | 2026-09-27 | 8.04 | 40 | 4 | v1 + >=420 s turn yield + idle/stuck nudges |

Leaderboard submissions:

| Ref | Version | Submitted | Public LB |
|---|---|---|---|
| 56605759 | v1 | 2026-09-27 10:19 UTC | pending |
