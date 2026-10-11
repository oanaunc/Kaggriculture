# Most Agent Tweaks Are Noise
### A paired, repeated-measures study of configuring a Gemma 4 SWE agent under a fixed harness

> DRAFT (2026-10-11). Gemma 4 Developer Agent — Paper Track.

## Abstract

Competitions and benchmarks for coding agents are usually iterated one leaderboard submission at a time. We ask how much of the measured progress from configuring an agent (prompts, budgets, thinking mode, sampling, sub-agent pipelines, verification loops) is real. Using the official evaluator on 40 public tasks, we ran seven full evaluations of six configurations of a two-stage Gemma 4 31B agent, plus five public-leaderboard reruns. Identical configurations differ by up to 3 tasks out of 40 between runs, and by ±2 tasks (±0.04) on the leaderboard. All six configurations land between 6 and 10 of 40 and none differs significantly from the baseline in paired tests. Per-task analysis explains why: 28 of the 40 tasks were never solved by any configuration, 5 were always solved, and every observed difference comes from the same 7 "coin-flip" tasks. Three findings follow for anyone tuning agents under a fixed model. First, single-run comparisons cannot separate configurations within ±3 tasks per 40. Second, the reachable set is small (12 of 40, 30%) and is set by the model rather than the scaffold. Third, adding checks or instructions usually spends the per-task budget faster than it converts failures.

## 1. Introduction

The Gemma 4 Developer Agent competition fixes the model (Gemma 4 31B, int4 QAT, 32k context, served by vLLM on 4×L4) and the tool interface. It scores a declarative agent configuration on SWE-bench-style Python issues. Participants can change only the agent: prompts, sub-agent pipelines, skills, sampling and per-task budgets. The public leaderboard has about 58 tasks, so one task is about 0.017, and each team gets one submission a day. Most public progress, ours included, has been made by editing the configuration and reading a single leaderboard number.

We report what happened when we stopped trusting single numbers. We built an offline harness around the official evaluator, ran configurations repeatedly on the same 40 public tasks, and compared them task by task. Our contribution is a measurement, not a new agent:

1. the run-to-run noise of a fixed configuration, on the leaderboard and offline;
2. a paired comparison of six configurations that span the common tuning levers;
3. a per-task decomposition showing which tasks any configuration can move;
4. a failure taxonomy that explains why intuitive fixes do not convert.

## 2. Setting

**Task and scoring.** Each task is a repository snapshot plus an issue. The agent's final patch (the last `submit_patch`, or the working-tree diff) is graded by hidden tests. The hidden set has about 120 tasks, half from private repositories, and runs in 12 h on 4×L4.

**Agent.** Our baseline (cand1) is the strongest public configuration we found (credited to hsiaosuan). It is a `SequentialAgent` with a read-only *explore* stage that localises the bug and hands a plan of at most 100 words to a *coder* stage. The coder edits, checks and submits. Caps are 5 minutes, 60 tool calls and 100 turns per task, with temperature 0.2 and thinking off. It scored 0.13 on the public leaderboard, our best.

**Offline harness.** A Kaggle notebook installs the competition wheels, serves the model with vLLM (TP=4) and runs the official `Evaluator` on a fixed, deterministic set of 40 public tasks, two at a time. It records pass/fail, tool calls, duration, patch size, errors and the tail of the test output for every task. A run costs about 1.3 GPU-minutes per task after an 11-minute start-up. Comparisons are paired by task and use an exact McNemar test.

## 3. How noisy is one measurement?

| Measurement | Run 1 | Run 2 | Difference |
|---|---|---|---|
| Leaderboard, identical zip (public base) | 0.12 | 0.08 | 2 tasks of ~58 |
| Offline, cand1 | 10/40 | 7/40 | 3 tasks flipped one way |
| Offline, cand1 at temperature 0 | 10/40 | 8/40 | 4 tasks flipped (3 / 1) |

Temperature 0 does not make runs repeatable. Batch composition in vLLM and wall-clock caps both change trajectories, so a task near the time or call limit can flip between runs. The practical consequence is that a single 40-task run cannot separate configurations within about ±3 tasks, and a single leaderboard submission cannot separate them within ±0.04. Several configurations we and others had judged "worse" earlier (0.08 against 0.12) are inside this band.

## 4. Six configurations, one band

| Config | Change vs cand1 | Solved / 40 | Paired vs cand1 run 1 |
|---|---|---|---|
| cand1 (×2) | — | 10, 7 | — |
| cand3 | Caps 100 calls / 7 min / 140 turns | 9 | n.s. |
| cand4 | cand3 + rule "create the requested public names exactly; import-check after edits" | 6 | p = 0.125 (4 losses, 0 wins) |
| cand5 (×2) | Temperature 0 for both stages | 10, 8 | n.s. |
| cand6 | + `LoopAgent` verifier: run existing tests, retry once, finalise | 8 | p = 0.69 |
| cand2 (LB only) | Thinking on (budgets 2048/1024) | LB 0.08 vs 0.13 | — |

Every configuration falls between 6 and 10. None is significantly different from the baseline.

## 5. Why: a small, fixed reachable set

Across the seven offline runs, every task falls into one of three groups:

- **Never solved (28 of 40):** no configuration ever solved these.
- **Always solved (5 of 40):** every configuration solved these.
- **Sometimes solved (7 of 40):** solved by some runs and not others, including identical reruns.

The union over all runs, an oracle that picks the best run per task, is **12 of 40 (30%)**. Every difference in Section 4 comes from the 7 coin-flip tasks. That means the configuration levers we tried redistribute luck within a fixed set; they do not reach new tasks.

## 6. Where the failures are, and why fixes do not convert

Over all 280 task-runs:

| Outcome | Share |
|---|---|
| Wrong patch (tests run, fail) | 26% |
| Patch breaks test collection | 25% |
| Resolved | 21% |
| Timeout | 11% |
| No patch, tool-call cap | 10% |
| No patch, other | 5% |
| Harness error | 4% |

The two largest classes are reasoning failures: the model edits the right place in the wrong way, or the tests import a new name the issue asked for and the agent never created it. Interface fixes aimed at these did not help.

- **Explicit "create the requested names" rule (cand4).** Collection errors did not fall, and the extra instruction coincided with 4 lost tasks.
- **Test-and-retry loop (cand6).** The loop consumed the per-task budget. No-patch-at-cap failures rose from 3–4 to 6 and timeouts from 2–4 to 5. Any failure the retry rescued was offset by tasks that never produced a patch.
- **Larger caps (cand3).** The agent used the extra budget, but timeouts and harness errors rose and the solved count did not move.
- **Thinking on (cand2).** Longer turns inside a 5-minute cap. The leaderboard score fell from 0.13 to 0.08.

The pattern matches what we saw in a separate domain, a small-model agent for ARC-AGI-3 games. There, harness fixes that corrected false or missing information shown to the model (a stale "game over", lost memory notes) changed which games were solved. Additional instructions and budget rules did not. In both domains, cheap interface work pays only when it repairs information the model actually lacks.

## 7. Recommendations

1. **Measure noise before tuning.** Run the baseline twice; any claimed gain smaller than the observed flip count is not evidence.
2. **Compare paired, per task**, on the same task set, and report the coin-flip set alongside the mean.
3. **Report the oracle (union) ceiling.** If the union over all configurations is close to the mean, scaffold tuning has little left to give, and the next gains must come from the model (fine-tuning, a larger model) or from interface changes that expose new information.
4. **Budget is shared.** Every check, retry or instruction spends calls and minutes that the edit loop needs. Evaluate additions against the cap-hit and timeout rates, not just the resolve rate.

## 8. Related work

Agent-computer interfaces and scaffolds for repository repair (SWE-agent, OpenHands, Agentless) show large gains from interface design for frontier models. Our results suggest that, for a small quantised model under tight per-task budgets, the scaffold's effect is bounded by a small reachable set. Variance in agent evaluations is increasingly reported in SWE-bench-style work. Our contribution is a cheap protocol (paired reruns, coin-flip decomposition, oracle ceiling) that a competition participant can run with weekly GPU quota.

## 9. Limitations

Forty public tasks and seven runs; the public repositories (mostly FastAPI and Rich) differ from the private half of the hidden set. Configurations differ in more than one parameter in places (cand4 includes cand3's caps). We did not evaluate fine-tuning (LoRA), which the competition allows and which is the natural next lever by our analysis.

## Reproducibility

Code, notebooks, per-task results for every run and the analysis script are at https://github.com/oanaunc/Kaggriculture/tree/claude/inspiring-thompson-aua2h1/gemma (`offline_eval/`, `offline_eval/runs/`, `offline_eval/analyze.py`).
