# ARC-AGI-3 results log

Public 25 games, one run each, same model (Qwen3.8 Flash Next NVFP4), 132 min/game, 28 concurrent.

| Run | Date | Mean score | Levels | Zero-score games | Notes |
|---|---|---|---|---|---|
| Public Duck A (wuliao0) | 2026-09-18 | 5.78 | 35 | 5 | reference |
| Public Duck B (keithtyser) | 2026-09-01 | 6.76 | 36 | 6 | reference |
| Ours v1 | 2026-09-27 | 6.25 | 33 | 5 | game-over fix, animation evidence, note parsing, context budget, level memory |
| Ours v2 | 2026-09-27 | 8.04 | 40 | 4 | v1 + >=420 s turn yield + idle/stuck nudges |
| Ours v2 (rerun) | 2026-09-27 | 6.77 | 36 | - | identical code; run-to-run noise ~±1 point. v2 mean of 2 runs = 7.41 vs public mean 6.27 |
| Ours v3 | 2026-09-27 | 5.71 | 31 | 6 | v2 + keep reasoning only for newest turn -> worse; reverted (DUCK_KEEP_REASONING_TURNS=-1 default) |

Leaderboard submissions:

| Ref | Version | Submitted | Public LB |
|---|---|---|---|
| 56605759 | v1 | 2026-09-27 10:19 UTC | 3.42 (rank 470/3401) |
| 56623477 | v2 | 2026-09-28 00:15 UTC | pending |

Leaderboard context (2026-09-27 20:17 UTC, 3406 teams): #1 27.29, #5 18.80, #10 11.64, #20 7.36, #50 5.11, #100 4.43, #200 4.02, #300 3.75.
Most teams sit at 3-5 (public Duck forks). Public-25 of the public Qwen3.8-27B-FP8 Duck variant (foysal, 2026-08-18): 3.79 / 25 levels, i.e. worse than Flash-Next.

Robust effect across v1-v3: dc22, sp80, tn36 complete >=1 level in every run (0 in both public reference runs).
