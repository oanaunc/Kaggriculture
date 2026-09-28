# Fixing the Harness, Not the Model
### Failure analysis and targeted harness fixes for a small-LLM ARC-AGI-3 agent

> DRAFT — numbers marked **[TBD]** are filled in when the pending leaderboard and evaluation runs finish.
> Track: ARC-AGI-3. Leaderboard submission ID: 56605759 (v1, public LB 3.42) **[TBD: final ID]**. Code: public notebook **[TBD link]**.

## 1. Summary

Small open models driving a Python tool (the "Duck" harness by Tufa Labs, here with Qwen 3.8 on one RTX Pro 6000) play ARC-AGI-3 games but score only ~6% on the public games. We asked a narrower question than "which model or prompt": **how much of the failure is caused by the harness rather than the model's reasoning?** Parsing 50 full transcripts (25 public games × 2 public runs) and checking blocked games against their source code, we found harness defects that silently corrupt what the model sees or remembers. Fixing them made three games that scored zero in both reference runs (sp80, tn36, dc22) complete levels in every one of our runs, with the same model, sampling and budget. Reallocating compute from games stuck after their first level to games still progressing gave the largest further gain: 8.85 mean over two runs versus 6.27 for the reference. We also report two ideas that did not work, since with large run-to-run variance negative results are informative.

The contribution is a transcript-driven method for auditing agent harnesses and a set of general, model-agnostic fixes that apply to any tool-using agent operating an interactive environment.

## 2. Setting

Each ARC-AGI-3 game is a 64×64, 16-colour interactive environment with hidden rules and several levels. Score per level is min(1, human_actions / agent_actions)², weighted by level index. Kaggle provides 9 hours on one RTX Pro 6000 for 110 hidden games; the harness plays 28 games concurrently with a 132-minute wall budget each.

The Duck exposes the game to the model as Python variables (current frame as ASCII and a connected-component segmentation, history, transitions) plus an `action(...)` function, attaches a 4× upscaled image of the board to each turn, keeps a rolling 32k-token context, and asks the model to maintain a labelled "world model" note that the harness carries between turns.

## 3. Method: transcript audit

We wrote parsers over the transcripts and the benchmark records to measure, per game: levels completed, actions vs human baseline, LLM calls and their latency, time between consecutive actions, how often the carried world-model note was present, and every harness message shown to the model. For the five games that scored zero in both runs we compared the model's stated beliefs with the game's source code.

Findings (both reference runs, 50 game-runs):

1. **The binding constraint is LLM calls, not actions.** Every game-run hit the wall budget with only ~55 LLM calls (~142 s each). On levels it did complete, the agent was as action-efficient as humans (median 0.82–0.88× the baseline). 61–64% of all time went to the last, never-solved level.
2. **Stale terminal state.** After a game over, the harness auto-resets the level, but the next prompt still said "The game is over." (207 prompts) while the system prompt says to stop acting on game over. In one run of dc22 the model idled for 28 turns (65 minutes, half its budget).
3. **Invisible mechanics.** The harness kept only the last frame of each action. In sp80, SPACE pours liquid through deflectors and the board reverts when the pour fails; the model only ever saw a timer tick and concluded "SPACE is a no-op" (0 levels in both runs). A random-play probe showed animation-only effects in 12 of 25 games.
4. **Lost memory.** The note parser matched only the literal prefix `World model:`; the model mostly wrote `World model (revised):` or `World model v12:`, so ~70% of its notes were dropped and the carried model was empty on 47–49% of turns. Notes were also wiped on every level transition and game over.
5. **Context mis-budgeting.** The token estimator counted each attached board image by its base64 length (~2.5k "tokens" instead of ~100 real vision tokens), so the 32k window held only ~8 recent calls. A 60 s turn budget, shorter than one LLM call, ended almost every turn after one tool call and re-sent the full user prompt.

## 4. Fixes

All fixes are harness-only and game-agnostic.

| Fix | Mechanism | Targets |
|---|---|---|
| Truthful terminal state | After an auto-reset, clear `game_over` in the carried summary and action result; tell the model the level restarted and ask it to name the cause of the loss | Idle loops after game over |
| Animation evidence | For every action, diff all intermediate frames against the pre-action board; report transient cells, their bounding box, and whether the board reverted | Mechanics that are only visible mid-animation |
| Robust note parsing | Accept markdown and qualified headers (`**World model (revised):**`, `v12`, `update`) | Memory loss |
| Knowledge carry-over | On a level transition keep the action model and the goal that worked, plus the last actions that completed the level, as cross-level notes; on game over keep the models and flag the failed plan | Re-learning each level |
| Context budget | Count images at a fixed vision-token cost; keep only the newest image in history | Short effective memory |
| Turn budget and nudges (v2) | Yield only after ≥420 s; prompt after ≥3 non-acting turns and after ≥50 actions on a level to enumerate untested win hypotheses | Analysis loops, stuck levels |
| Compute reallocation (v5) | A game that completed a level but then made no progress for 75 min stops, freeing GPU share for games still progressing; a recently progressing game may run past its budget (≤1.6×) while the run is ahead of the original worst-case schedule | 61–64% of time spent on never-solved levels |

Every change is covered by unit checks and an offline smoke test against a mock LLM server.

## 5. Results

Public games, same model, sampling and budget as the reference runs. Identical code varies by about ±1 point between runs (v2 twice: 8.04 and 6.77), so we report every run.

| Run | Mean score | Levels | Games with 0 levels |
|---|---|---|---|
| Public Duck, run A | 5.78 | 35 | 5 |
| Public Duck, run B | 6.76 | 36 | 6 |
| Ours v1 (fidelity fixes) | 6.25 | 33 | 5 |
| Ours v2 (v1 + turn budget + stuck nudges), run 1 | 8.04 | 40 | 4 |
| Ours v2, run 2 | 6.77 | 36 | 5 |
| Ours v3 (v2 + keep reasoning only for newest turn) | 5.71 | 31 | 6 |
| Ours v4 (v2 + stall preemption at 75 min) | 7.02 | 30 | 8 |
| Ours v5 (v2 + preemption only after the first level), run 1 | 10.34 | 44 | 3 |
| Ours v5, run 2 | 7.35 | 39 | 2 |

![Levels completed per game](figures/levels_per_game.png)

Leaderboard (hidden games): v1 3.42, v2 3.66 (reference forks of the public Duck cluster at 3–5).

What is robust across runs: sp80, tn36 and dc22 complete at least one level in every run of v1–v4 and never in the references. In sp80 and tn36 the new animation evidence was shown in 23 and 26 prompts respectively; in dc22 the model no longer idles on a false "game over". What is not robust: the aggregate, which single runs cannot resolve.

Negative results. (i) Dropping past hidden reasoning from the carried history (v3) freed context but lowered the score: the model loses continuity of its own hypotheses. (ii) Stopping games after 75 minutes without progress (v4) freed GPU share, which let sk48 complete its first level for the first time, but also killed slow first levels (sc25, cn04, cd82). Restricting preemption to games that had already completed a level (v5) keeps the benefit without the cost: it is our best configuration, averaging 8.85 over two runs (vs 7.41 for v2 and 6.27 for the reference), with g50t completing a level for the first time in any run.

## 6. Why it works

The Duck's model is competent once it has the right evidence: on the levels it completes it matches human action counts. Its failures concentrate where the harness breaks the observation–memory loop: it cannot learn a mechanic it never observes (animation frames), cannot keep a hypothesis it is not allowed to remember (dropped notes, wiped models, a context full of stale images), and cannot act when the environment description is false (stale game over). These are failures of *state fidelity*, and they compound over a long episode because every later hypothesis is built on corrupted evidence. Fixing fidelity is cheap, it makes no assumptions about particular games, and it transfers to any agent that interacts with an environment through a harness — including coding agents, where the equivalent defects are truncated tool output, stale build state and lost scratch notes.

## 7. Limitations and next steps

Two runs per configuration cannot give tight confidence intervals; we report per-game mechanism evidence alongside aggregates. Remaining failures are genuine reasoning gaps (sk48's undo, bp35's physics). Next: expose intermediate frames as Python objects and improve serving throughput.

## Acknowledgements

Built on the Duck harness by Tufa Labs (MIT) and the public Qwen 3.8 serving setup shared by Kaggle community members.
