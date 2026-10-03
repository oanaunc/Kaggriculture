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
| Swift (v6 harness + Swift 1.5 model) | 2026-09-30 | 7.32 | 35 | 5 | model loads via staged view (SWIFT_MODEL_SANITY ok, VLLM_SETUP_COMPLETE). MTP acceptance 0.667 (v5: 0.596), requests 1333 (v5: 1214), gen tokens ~= same. Within noise of v5 on public-25. |
| Ours v4 | 2026-09-28 | 7.02 | 30 | 8 | v2 + stall stop 4500 s / progress extension. Stall fired in 20/25 games: freed GPU helped some (sk48 first level ever, lp85 5 levels) but killed level-1 progress in sc25/cn04/cd82. Extension untestable in single-wave public eval. Not submitted. Next: stall only after >=1 level or longer level-1 window. |
| Ours v3 | 2026-09-27 | 5.71 | 31 | 6 | v2 + keep reasoning only for newest turn -> worse; reverted (DUCK_KEEP_REASONING_TURNS=-1 default) |

Leaderboard submissions:

| Ref | Version | Submitted | Public LB |
|---|---|---|---|
| 56605759 | v1 | 2026-09-27 10:19 UTC | 3.42 (rank 470/3401) |
| 56623477 | v2 | 2026-09-28 00:15 UTC | 3.66 (rank 374/3430) |
| 56656628 | v5 | 2026-09-29 00:14 UTC | 2.53 (worse than v2 3.66) |
| 56693066 | v6 | 2026-09-30 01:12 UTC | 3.33 |
| (see API) | swift | 2026-10-01 00:13 UTC | pending |

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

### 2026-09-30: LB vs public-25
LB scores of all versions (3.42, 3.66, 2.53, 3.33) lie within ~±0.5 of each other, while public-25 means moved 6.3 -> 8.9.
Public-25 gains do not transfer measurably to the hidden LB games; LB run-to-run noise (~±0.5) dominates.

## Next submissions (plan, updated 2026-09-30 18:10 UTC)
- 2026-10-01 ARC-AGI-3: Swift kernel, already fully run: `kaggle competitions submit arc-prize-2026-arc-agi-3 -k oanaunciuleanu/arc3-duck-ours-swift -v 1 -f submission.parquet -m "swift: v6 harness + Swift 1.5 model"` (LB is the real test; public-25 does not predict LB).
- 2026-10-01 Gemma: gemma/dist/submission_v4.zip (0.12 base + coder tail-output line), uploaded as submission.zip.
- 2026-10-01 ARC-AGI-2: only if v2 (probmul_3) scores; otherwise nothing.

## 2026-10-01 check-in
- LB: Swift (56732275) = **2.23** (v2 3.66 remains best). Swift model does not help on hidden games; drop it.
- Milestone-2 releases: **Daniel Franzen (LB 27.24) published full solution** `dfranzen/arc-agi-3-milestone-2-solution`; Lord Han Solo (23.84) published `lordhansolo/arc-agi-3-milestone-2`.
- Franzen = Duck @7652836 + 500 KB harness patch (16 files, ~5.3k lines in tool_agent), Intel AutoRound W4A16 Qwen3.8-Flash-Next + albucino MTP drafter on Pennyroyal SGLang fork, 128k context, memory sections OFF (long-context history instead), priority scheduler over all 110 games (10 active streams), UNDO exposed, animation/diff images, guards from level 2, persistent functions.
- Already covers our fixes (animations, tolerant headers, wipe controls, stale game-over); our harness is dominated. Models are Qwen Community License (same as current).
- Plan: private fork `arc3/franzen_fork/` (kernel `oanaunciuleanu/arc3-franzen-ours`) -> test run -> LB baseline -> one evidence-backed change per day. Kaggle push of the fork is pending user approval.
- Fork test run (kernel `oanaunciuleanu/arc3-franzen-ours` v1, non-submission mode: 10 public games x 25 min, the 15 hardest excluded by Franzen's demo list): **mean 46.32**, 49/72 levels, ft09 6/6, 0 tracebacks, 560 gen tok/s. Pipeline works on our account. Not comparable to our public-25 numbers (easier subset, different budget).
- Plan: submit v1 unchanged at the 2026-10-02 00:12 daily cycle as our LB reference; then one change per day.
- 2026-10-02 00:13: submitted franzen-fork v1 (unchanged) to LB.

## 2026-10-02
- **franzen-fork v1 LB = 29.28** (Franzen's own 27.24; writeup claims 27.89). Rank 59/3585 (was 624). Cluster of Franzen forks spans ~26-32 -> LB noise at this level ≈ ±3.
- Tufa Labs now 52.51, Yi-Chia Chen 48.07 (private). Lord Han Solo's new public notebook is his own vLLM line (23.84 best), not better.
- Franzen writeup (dfranzen/arc-agi-3-milestone-2): world model off, 10x images, animations, UNDO, guards = his wins; summarisation/death ledger/extra guards/pace = no clear gain. Only stated open lever: **token throughput** ("scores still improving near the end of runs"). Run already uses ~532 min of the 540 min limit, so no free wall-clock.
- Plan: (a) 2026-10-03 resubmit v1 unchanged = second draw to measure noise; (b) throughput A/B on the 10-game demo (metric: gen tok/s and levels, both far less noisy than LB) once weekly GPU quota resets — candidates SPEC_STEPS 3->4, CUDAGRAPH/MAXREQ, prefill chunk; ship only a measured tok/s gain.

## 2026-10-03
- Submitted franzen-fork v1 rerun (56785063) to measure LB noise.
- Launched throughput test `oanaunciuleanu/arc3-franzen-spec4` (only change: SPEC_STEPS 3->4). Compare gen tok/s vs v1 demo (560.22) and levels (49/72).
