# ARC-AGI-3 results log

Public 25 games, one run each, same model (Qwen3.8 Flash Next NVFP4), 132 min/game, 28 concurrent.

| Run | Date | Mean score | Levels | Zero-score games | Notes |
|---|---|---|---|---|---|
| Public Duck A (wuliao0) | 2026-09-18 | 5.78 | 35 | 5 | reference |
| Public Duck B (keithtyser) | 2026-09-01 | 6.76 | 36 | 6 | reference |
| Ours v1 | 2026-09-27 | 6.25 | 33 | 5 | game-over fix, animation evidence, note parsing, context budget, level memory |
| Ours v2 | 2026-09-27 | 8.04 | 40 | 4 | v1 + >=420 s turn yield + idle/stuck nudges |
| Ours v2 (rerun) | 2026-09-27 | 6.77 | 36 | - | identical code; run-to-run noise ~±1 point. v2 mean of 2 runs = 7.41 vs public mean 6.27 |
| Ours v5 | 2026-09-28 | 10.34 | 44 | 3 | v2 + reallocation, stall-stop only after first level (10 games stalled). g50t first level ever, tr87 4 levels. Best so far; submit 2026-09-29; rerun pushed. |
| Ours v5 (rerun) | 2026-09-28 | 7.35 | 39 | 2 | identical code; v5 mean of 2 = 8.85 (v2 mean 7.41, public mean 6.27). Noise ~±1.5. |
| Ours v4 | 2026-09-28 | 7.02 | 30 | 8 | v2 + stall stop 4500 s / progress extension. Stall fired in 20/25 games: freed GPU helped some (sk48 first level ever, lp85 5 levels) but killed level-1 progress in sc25/cn04/cd82. Extension untestable in single-wave public eval. Not submitted. Next: stall only after >=1 level or longer level-1 window. |
| Ours v3 | 2026-09-27 | 5.71 | 31 | 6 | v2 + keep reasoning only for newest turn -> worse; reverted (DUCK_KEEP_REASONING_TURNS=-1 default) |

Leaderboard submissions:

| Ref | Version | Submitted | Public LB |
|---|---|---|---|
| 56605759 | v1 | 2026-09-27 10:19 UTC | 3.42 (rank 470/3401) |
| 56623477 | v2 | 2026-09-28 00:15 UTC | 3.66 (rank 374/3430) |
| 56656628 | v5 | 2026-09-29 00:14 UTC | 2.53 (worse than v2 3.66) |

Leaderboard context (2026-09-27 20:17 UTC, 3406 teams): #1 27.29, #5 18.80, #10 11.64, #20 7.36, #50 5.11, #100 4.43, #200 4.02, #300 3.75.
Most teams sit at 3-5 (public Duck forks). Public-25 of the public Qwen3.8-27B-FP8 Duck variant (foysal, 2026-08-18): 3.79 / 25 levels, i.e. worse than Flash-Next.

Robust effect across v1-v3: dc22, sp80, tn36 complete >=1 level in every run (0 in both public reference runs).

## Next submissions (plan)
- 2026-09-29 ARC-AGI-3: `kaggle competitions submit arc-prize-2026-arc-agi-3 -k oanaunciuleanu/arc3-duck-ours-v5 -v 1 -f submission.parquet -m "v5: v2 + compute reallocation (stall-stop after first level, progress extensions)"` (version 1 = the 10.34 run; version 2 is the confirmation rerun, identical code).
- 2026-09-29 Gemma: exact 0.12 base `gemma/dist/submission_r12base.zip` uploaded as `submission.zip` (calibration).

### 2026-09-30 analysis of the v5 LB drop
Public-25 evaluation runs all 25 games in one wave (concurrency 28), so the progress *extension* (up to 1.6x base budget)
never activated there; only the stall-stop was measured (v5: 10.34 / 7.35). On the 110-game LB run (4 waves), extensions
lengthen waves, and the last wave most likely got cut by the 9 h notebook limit. v6 = v5 with DUCK_EXTEND_MAX_FACTOR=1.0,
i.e. exactly what the public evaluation measured. Commit runs now use ARC3_QUICK_COMMIT=1 (2 games x 2 min) to save GPU quota.
