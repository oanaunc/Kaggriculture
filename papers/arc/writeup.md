# Fixing the Harness, Not the Model
### Failure analysis and targeted harness fixes for a small-LLM ARC-AGI-3 agent

> DRAFT — numbers marked **[TBD]** are filled in when the pending leaderboard and evaluation runs finish.
> Track: ARC-AGI-3. Leaderboard submission ID: 56605759 (v1, public LB 3.42) **[TBD: final ID]**. Code: public notebook **[TBD link]**.

## 1. Summary

Small open models driving a Python tool (the "Duck" harness released by Tufa Labs, here with Qwen 3.8 on one RTX Pro 6000) can play ARC-AGI-3 games, but score only ~6% on the public games. Before changing the model or the prompts, we asked a narrower question: **how much of the failure is caused by the harness rather than by the model's reasoning?** We parsed 50 full game transcripts (25 public games × 2 independent public runs), checked every blocked game against its source code, and found that several of the most damaging failures were harness defects that silently corrupt what the model sees or remembers. Fixing them unlocked the first level of three games that scored zero in both reference runs (sp80, tn36, dc22), without touching the model, the sampling or the per-game budget. **[TBD: leaderboard score and multi-run statistics.]**

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

Every change is covered by unit checks and an offline smoke test that plays all 25 public games against a mock LLM server, so harness regressions are caught without GPU time.

## 5. Results

Public games, one run each (the same model, sampling and budget as the reference runs):

| Run | Mean score | Levels completed | Zero-score games |
|---|---|---|---|
| Public Duck, run A | 5.78 | 35 | 5 |
| Public Duck, run B | 6.76 | 36 | 6 |
| Ours v1 | 6.25 | 33 | 5 |
| Ours v2 | 8.04 | 40 | 4 |

![Levels completed per game](figures/levels_per_game.png)

The targeted games respond as predicted: sp80, tn36 and dc22 each completed a level for the first time; the new animation evidence appeared in 23 (sp80) and 26 (tn36) prompts. Other games moved in both directions (bp35 and tr87 fell to zero), consistent with run-to-run variance. The aggregate score is flat because per-game variance between identical runs is large (ft09 cleared 1 vs 4 levels across the two reference runs), which also dominates single-run comparisons. **[TBD: repeated runs with confidence intervals; leaderboard score.]**

## 6. Why it works

The Duck's model is competent once it has the right evidence: on the levels it completes it matches human action counts. Its failures concentrate where the harness breaks the observation–memory loop: it cannot learn a mechanic it never observes (animation frames), cannot keep a hypothesis it is not allowed to remember (dropped notes, wiped models, a context full of stale images), and cannot act when the environment description is false (stale game over). These are failures of *state fidelity*, and they compound over a long episode because every later hypothesis is built on corrupted evidence. Fixing fidelity is cheap, it makes no assumptions about particular games, and it transfers to any agent that interacts with an environment through a harness — including coding agents, where the equivalent defects are truncated tool output, stale build state and lost scratch notes.

## 7. Limitations and next steps

Single runs are noisy, so we report per-game mechanism evidence alongside the aggregate. Remaining failure modes are genuine reasoning gaps (g50t's record-and-replay clone, sk48's undo) and throughput. Next: run repeated evaluations to measure variance, expose intermediate frames directly as Python objects, persist a small Python workspace across tool calls, and improve serving throughput so each game gets more reasoning within the same budget.

## Acknowledgements

Built on the Duck harness by Tufa Labs (MIT) and the public Qwen 3.8 serving setup shared by Kaggle community members.
