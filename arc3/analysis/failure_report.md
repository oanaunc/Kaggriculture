# Duck (Qwen3.8-Flash-Next-NVFP4 + python tool) on ARC-AGI-3: failure analysis

Runs analysed (both use the same model, system prompt and harness settings: 32k context, 60 s yield, 1024-token tool output, 25 games in parallel, 7 920 s wall cap per game):

| run | label used here | mean score | levels | LLM requests | generated tokens |
|---|---|---|---|---|---|
| `wuliao0_duck-qwen3-8-anim-base` (2026-09-18) | **A** | 5.78 | 35 / 183 | 1 402 | 1.92 M |
| `keithtyser_duck-qwen3-8-flash-next-nvfp4-mtp` (2026-09-01) | **B** | 6.76 | 36 / 183 | 1 371 | 1.97 M |

Method: python parsers in `/tmp/arc3/analysis/` (`parse.py` → `metrics.json`, `agg.py`, `agg2.py`, `idle.py`, `wmcheck.py`, `codecheck.py`, `digest.py`, `lastwm.py`, `anim_probe.py`). Game mechanics were checked against `environment_files/<game>/*/<game>.py`, and harness behaviour against `ARC3-Inference/inference/{agent,framework}`. Step references below are `analysis_step=N` headers in the transcript files.

## 0. Top findings

1. **The binding constraint is LLM calls, not game actions.** Every one of the 50 game-runs ran until the 7 920 s wall cap (the benchmark's `state=gave_up` is the time cap, not the agent quitting). Each game got about 55 LLM requests at about 142 s each (about 9.6 tok/s per stream with 25 streams sharing one server). About 95 % of generated characters are hidden reasoning. On the levels it did clear, the agent was as action-efficient as humans: median actions/baseline was 0.82 (A) and 0.88 (B). Level 1 took a median of 12–14 LLM calls (about 17–19 min).
2. **61–64 % of all wall time was spent on the final level that never got solved.** In B, 15 of 25 games had a gap of more than 15 minutes between two consecutive actions, and 9 had a gap of more than 30 minutes. These are pure analysis loops.
3. **The harness loses most of the model's memory.** The world-model extractor only matches the literal prefix `World model:`, so the model's usual headers `World model (revised):`, `World model v12:` and `World model update:` are dropped. Only 30 % of world-model updates reach the next prompt. The carried model is empty on 47–49 % of turns. It is wiped on every level transition. The 32k context holds only about 8 recent calls, because the token estimator counts the base64 PNG of every retained image (about 2.5k "tokens" each).
4. **The harness keeps saying "The game is over." after it has already auto-reset the level.** This happened in 207 prompts. The stale `last_action_result.game_over=True` persists, and the system prompt says "stop acting immediately". dc22-B sat idle for 28 turns (65 min, half of its run).
5. **Animation frames are thrown away.** The framework keeps only `state.frame[-1]`. In sp80, SPACE pours liquid through the deflectors and then reverts on failure. The agent only ever saw the HUD tick, so it concluded "SPACE is a no-op" (sp80-A step 3) and scored 0 in both runs. A random-play probe found that 12 of 25 games show gameplay-relevant changes only in intermediate frames.
6. **Classic tool and harness errors are negligible.** Python exceptions affect 4 % of calls (about 4 % of wall time). There were 5 sandbox timeouts in total, 0 malformed tool calls, and 0 markup-recovered calls. The 49 request errors are all the one in-flight request cut at the deadline.

## 1. Per-game results and diagnosis

`act/base` is the sum of actions over completed levels against the human baseline for the same levels. `stuck` is the number of actions spent on the unfinished level. The diagnosis names what blocked the *next* level.

| game | lv | A done | A act/base | A stuck | B done | B act/base | B stuck | what blocked progress |
|---|---|---|---|---|---|---|---|---|
| ar25 | 8 | 3 | 158/157 | 50 | 2 | 80/82 | 56 | Mirror/reflection puzzle. Controls and the goal are re-learned on each level after the memory wipe. Later levels are misread: B at step 32 has "both pieces bbox-matched ... yet no completion" and the goal model stays wrong. |
| bp35 | 9 | 1 | 52/21 | 48 | 1 | 23/21 | 48 | Gravity/water "fish" level. There were 4–5 deaths entering the carved shaft (B step 18: "Three deaths, all when the fish entered the carved shaft"). A mistook the auto-reset for "a new round". |
| cd82 | 6 | 1 | 86/55 | 3 | 3 | 108/104 | 10 | Paint-stamp compass. A wrote only 1 assistant note, so the carried world model was empty on 54/54 turns and L1 took until 7 415 s. B decoded the compass and palette masks and solved 3 levels. |
| cn04 | 6 | 1 | 24/29 | 25 | 1 | 14/29 | 80 | Plug-to-socket with rotation. L1 was solved under baseline. On L2 the model picked the wrong socket and wrong plug selection (B step 14: selection only via the red prong). A went 1 655 s with no action and made only 49 actions all game. |
| **dc22** | 6 | 0 | – | 299 | 0 | – | 92 | See §1b. Navigation hidden behind toggle-able bridges. A brute-forced a 36-state toggle×slot space. B got close, then idled 65 min on a stale "game over". |
| ft09 | 6 | 1 | 4/43 | 218 | 4 | 53/106 | 32 | Click-toggle pattern whose target is encoded in icon glyphs. A guessed patterns blindly on L2: "24 hypotheses failed" (step 29), 218 actions against a baseline of 12, 6 game-overs. B read the icon semantics and cleared 4 levels. |
| **g50t** | 7 | 0 | – | 116 | 0 | – | 95 | See §1b. A time-loop "record and replay clone" puzzle that the model read as an echo with a fixed lag. |
| ka59 | 7 | 1 | 23/28 | 168 | 2 | 129/137 | 3 | Push-slide blocks into matching holes. A never modelled "pushed piece slides 15 cells through walls" early enough, and spent 168 actions on L2 (B did L2 in 106). |
| lf52 | 10 | 1 | 10/32 | 201 | 1 | 19/32 | 45 | Peg/wire ferry puzzle. The L2 goal was never identified. Both runs repeatedly re-analysed L1's history to find the win trigger (A step 35: "in L1 the level completed when the ... key was clicked"). |
| lp85 | 8 | 4 | 87/102 | 4 | 3 | 62/86 | 15 | Ring-permutation puzzle. It was solved correctly with BFS, but each level lost 5–20 calls to simulator bugs ("BFS bug: iterating a string gave single chars", B step 17; "my nxt() had directions swapped"). B went 47 min without acting while debugging. |
| ls20 | 7 | 1 | 18/22 | 108 | 1 | 24/22 | 67 | Maze with key-glyph and exit box. The L1 win trigger was misattributed ("Level 1 ended exactly at a timer expiry", B step 16), so the L2 box stayed "locked". B went 43 min without acting. |
| m0r0 | 6 | 2 | 106/141 | 0 | 0 | – | 48 | Twin blocks with mirrored controls. B derived the exact rule by step 16 ("B = LEFT/RIGHT mirrored") but never found the completion condition, and went 47 min without acting. A cleared 2 levels. |
| r11l | 6 | 2 | 23/55 | 45 | 1 | 7/22 | 16 | Click-to-swap diamonds onto rings. B misread its own history as "my 4- and 5-click batches were dropped" (step 9). It had sent those clicks itself, and it made no action for 58 min (between actions 18 and 19). |
| re86 | 8 | 3 | 118/154 | 201 | 3 | 106/154 | 52 | Colour cursors must cover same-colour dots. The repaint-at-palette sub-mechanic was found very late: B went through world model versions v25→v45 across 24 non-acting turns (61 min). |
| s5i5 | 8 | 2 | 224/109 | 0 | 1 | 19/20 | 303 | Extendable coloured bars routed around walls. Routing was done by hand with no search: 184 actions on A-L2 against a baseline of 89, and B spent 303 actions on L2 without solving it. |
| sb26 | 8 | 2 | 113/46 | 98 | 2 | 322/46 | 0 | Sequence/box-filling puzzle. It brute-forced splits ("28 untested splits remain", A step 17), and a buggy solver "looped ... and burned ~24 actions" (B step 21). |
| sc25 | 6 | 2 | 77/42 | 7 | 2 | 38/42 | 19 | A panel pattern drives a key tooth. It was learned only through a game-over. B wrote no assistant content at all (0 notes), so the carried world model was empty on 54/54 turns. |
| **sk48** | 8 | 0 | – | 118 | 0 | – | 174 | See §1b. It got the goal roughly right but spent the whole budget learning rod-grip mechanics. |
| **sp80** | 6 | 0 | – | 430 | 0 | – | 322 | See §1b. The pour animation is invisible to the agent, so it decided "SPACE inert". There were 15/10 game-overs from sweeping. |
| su15 | 9 | 1 | 14/22 | 100 | 2 | 103/64 | 22 | Click-to-teleport pieces that merge. Click-anchor mistakes ("clicked cell becomes its new bottom-right" and "Click (51,18) failed unexpectedly", B step 36). |
| **tn36** | 7 | 0 | – | 311 | 0 | – | 335 | See §1b. It never triggered program execution and instead Gray-code brute-forced 10 bits. |
| tr87 | 6 | 1 | 34/54 | 26 | 1 | 40/54 | 100 | Glyph-cipher wheels. L2 became offline symbol analysis: A used 43 of 54 turns without acting (made only 60 actions all game) and had a 1 906 s gap. |
| tu93 | 9 | 2 | 40/35 | 100 | 2 | 72/35 | 131 | Maze with charging mines. There were 14/17 deaths, and the death rule was never pinned down ("lethality is NOT purely positional ... turn-parity", B step 27). |
| vc33 | 7 | 3 | 70/69 | 71 | 3 | 115/69 | 51 | Water-level pumps. The L4 target was misread ("water is conserved ... my target model is wrong", A step 31), and both runs spent calls re-diffing past winning frames. |
| wa30 | 9 | 1 | 54/71 | 142 | 1 | 47/71 | 198 | Sokoban with a chasing snake plus a step bar. There were timer game-overs on L2. B wrote no notes, so its world model was empty on 51/51 turns. |

### 1b. Zero-score games: real mechanic vs. what the model believed

| game | real mechanic (from source) | model's belief / where it went wrong |
|---|---|---|
| dc22 | The win is `player.xy == goal.xy`. The arrows move the player (green 2×2), but only onto walkable tiles. Clicking the coloured "buezna" buttons (the red/blue shapes on the right) swaps paired pieces between alternate positions, which opens or closes bridges. Later levels add a crane on click-buttons. A step counter causes a loss when it runs out. | **A** learned that the arrows move the token and that the cups toggle the red and blue pieces. It then treated the token as "confined to a 3×3 box" and enumerated "all 36 (slot × combo) states ... zero wins" (step 24). It never walked the token across a toggled bridge to the yellow goal. **B** briefly held the right goal ("deliver green to the yellow core", step 16), abandoned it after one blocked move, and then idled for 28 turns because the prompt kept saying "The game is over." |
| g50t | This is a time-loop puzzle. SPACE rewinds you to the start and turns your recorded path into a clone that replays it step-for-step alongside your next attempt. Standing on switches opens walls. A HUD bar moves 1 px every 2 steps and causes a loss when empty. | The model saw an "ECHO [that] replays my action queue with 14-step lag" and "SPACE teleports me to (0,0)" (A steps 20–23). It never used the clone to hold the switch while it walked through. The carried world model was empty on 41/52 turns (A), and the run produced only 7 assistant notes. |
| sk48 | Each extendable rod's attached blocks must match, colour by colour, a reference rod above the line. Arrows extend, retract or move the rod. A click selects the rod. ACTION7 is undo. There is a move budget. | The goal was close: "HUD legend ... Target = pack blocks against head in order R,N,b" (A step 32). But it treated the reference as HUD, spent more than 100 actions learning grip and carry rules, and never used ACTION7 (undo) to recover from mistakes. |
| sp80 | Click to select a deflector, move it with the arrows, then press SPACE to **pour**. Liquid falls from the source, is split or turned by the deflectors, and must fill every cup without spilling. A failed pour flashes, then **reverts the board** and uses up one of 4 pours. There is a step budget. | Every pour happened in animation frames the harness discards. The agent saw only "SPACE only consumed 2 timer cells, no gameplay change. So SPACE is a no-op" (A step 3) and "SPACE inert" (B step 15). It then swept "all 140 [paddle] positions" and burned the timer (15/10 game-overs). |
| tn36 | This is a programming puzzle. Tick cells encode a move program for a piece, and clicking the panel runs the program as a multi-frame animation. The win comes when the right-hand piece ends at its target. A HUD bar that shifts left per click causes a loss. | It modelled a "10-bit combination lock" and ran "the exhaustive 10-bit Gray sweep" (A step 22). It made 311 clicks on only 33 distinct cells, 79 of them on one cell. Later it conceded "my earlier sweeps were invalid" (step 31). It never found the execute trigger. |

## 2. Failure-mode ranking

The unit is a game-run, 50 in total. Primary is the main blocker of the unfinished level. Contributing counts every game-run where the mode visibly cost more than 3 calls.

| # | failure mode | primary | contributing | evidence |
|---|---|---|---|---|
| 1 | **Call starvation.** About 55 LLM calls per game at about 142 s each, 95 % of them hidden reasoning. | (all) | 50 | All 50 game-runs hit 7 920 s. Median of 13.5 (A) / 12 (B) calls to clear L1. Actions are *not* scarce: 546 actions against a 604 baseline on A's first levels. |
| 2 | **Goal never discovered, or wrong goal on the blocking level** | 17 | 27 | dc22, g50t, sp80, tn36 (both runs); ls20, lf52, vc33, ar25, m0r0-B, cn04, re86 late. vc33-A step 31: "my target model for level 4 is still wrong". |
| 3 | **Analysis paralysis / long non-acting loops** | 6 | 26 | 11 (A) and 15 (B) games had gaps of more than 15 min between actions, totalling 22.5k s (11 %) in A and 42.5k s (21 %) in B. re86-B: 24 turns; r11l-B: 58 min; tr87-A: 43/54 turns yielded. |
| 4 | **Memory loss** (dropped world-model labels, level wipe, 8-call context) | 3 | 50 | 70 % of world-model updates are not captured. cd82-A, sc25-B and wa30-B carried nothing. s5i5-A step 13: "re-grounding from scratch". 56/71 python calls reconstruct past level wins from `history`, in 18/20 games. |
| 5 | **Brute-force enumeration of hypotheses** (looping the same failed plan) | 5 | 9 | tn36 Gray sweep; ft09-A "24 hypotheses failed"; sb26 "28 untested splits"; dc22-A 36-state sweep; sp80 140-position sweep. |
| 6 | **Invisible animation frames** | 2 | 5+ | sp80 (decisive in both runs), tn36 (execution animation). Random play shows animation-only changes in 12/25 games. |
| 7 | **Game-over churn** (timer or hazard) | 3 | 16 | 58 (A) and 48 (B) auto-resets: tu93 14/17, sp80 15/10, ft09-A 6, tn36 5/5, bp35 4/5. Most of these burned the per-attempt step bar while probing. |
| 8 | **Stale "game over" idles the agent** | 1 | 14 | 99 (A) + 108 (B) prompts say "The game is over." after the auto-reset. There were 16 (A) / 38 (B) turns where the model refused to act because of it. dc22-B: "I must not take further actions" (step 19) while about 3 900 s remained. |
| 9 | **Self-written simulator/solver bugs** | 2 | 10 | lp85 (direction swap, string iteration), sb26 (wrong slot columns), m0r0-A ("my floor map is wrong"), r11l-B. 32 (A) / 43 (B) turns are about fixing bugs. |
| 10 | **Click-coordinate and anchor mistakes** | 1 | 6 | su15 (the anchor is the bottom-right corner), cn04 (select via the red prong), sb26 (wrong slot columns), r11l (off-network clicks). |
| 11 | **HUD/timer misattribution** | 0 | 8 | 2 010 (A) / 1 809 (B) mentions of HUD or timer in reasoning; 8–12 % of python calls measure bars. dc22-A step 17: "BREAKTHROUGH — clicking the red block ... changed the board", then "those changes were only the timer bar shrinking". |
| 12 | **Tool errors, timeouts, truncation, malformed calls** | 0 | 3 | 58/53 exceptions (4 % of calls), 3/2 sandbox timeouts, 8/11 truncated outputs, 0 malformed or markup-recovered calls. See §3. |

The run-to-run variance is large even though the configuration is identical: ft09 cleared 1 vs 4 levels, cd82 1 vs 3, m0r0 2 vs 0. Whether the model hits the right hypothesis early dominates the outcome.

## 3. Harness-level waste (A / B)

**Time and throughput**

| metric | A | B |
|---|---|---|
| LLM requests per game | 56.1 | 54.8 |
| mean seconds per request | 141 | 144 |
| generated tokens per request | 1 367 | 1 438 |
| reasoning chars : assistant-content chars | 4.69 M : 0.26 M | 4.67 M : 0.31 M |
| turns ending with *no* action (60 s yield, only 1 call fits) | 588 / 1 362 (43 %) | 648 / 1 331 (49 %) |
| inspection-only python calls | 626 / 1 401 (45 %) | 660 / 1 340 (49 %) |
| actions per acting turn: mean / median / share =1 / share ≥10 | 5.6 / 3 / 28 % / 15 % | 5.6 / 3 / 28 % / 17 % |
| wall time spent on the final, never-solved level | 120 k s (61 %) | 128 k s (64 %) |
| actions spent on the final, never-solved level | 2 889 / 4 224 (68 %) | 2 314 / 3 695 (63 %) |
| time in action gaps longer than 15 min | 22.6 k s (11 %) | 42.5 k s (21 %) |

**Errors**

| metric | A | B |
|---|---|---|
| python exceptions | 58 (4.1 %) | 53 (4.0 %) |
| wall time of turns with an exception | 8.2 k s (4.2 %) | 7.9 k s (4.0 %) |
| exception mix (NameError / TypeError / IndexError / ImportError) | 14 / 12 / 11 / 10 | 7 / 9 / 7 / 11 |
| sandbox-disallowed imports (`difflib`, `sys`, `os`, `time`) | 9 | 11 |
| python 30 s sandbox timeouts | 3 | 2 |
| truncated tool outputs (1 024-token cap) | 8 | 11 |
| malformed JSON, markup-in-text, `finish_reason=length` | 0 / 0 / 2 | 0 / 0 / 0 |
| request errors | 25 (all the final in-flight request at the deadline) | 24 (same) |
| context-overflow retries | 0 | 0 |

Most NameErrors reference variables from a previous call (`piece`, `diff`, `clusters`), because the sandbox is stateless.

**Context and memory**

| metric | A | B |
|---|---|---|
| history-trim events (history shrank between turns) | 210 | 235 |
| typical `history_messages` | 15–30 (about 8 calls, about 19 min of play) | same |
| turns where the carried world model was empty | 644 (47 %) | 652 (49 %) |
| turns where it was identical to the previous turn | 404 | 376 |
| assistant notes mentioning "world model" | 655 | 694 |
| …of which the extractor captured | 197 (30 %) | 193 (28 %) |
| "Cross-level notes" written (the only field that survives a level change) | 2 | 27 |

The context is small for four reasons:

* The estimator in `_estimate_tokens` is `len(json)/3` over the full payload, *including the base64 PNG* of every retained user image. A measured data-URL costs about 2 450–2 650 estimated tokens per image.
* The system prompt is about 4.2k estimated tokens.
* Every yield-retry appends another full user prompt of about 3.2k chars plus an image.
* Past reasoning is kept in history.

**Repeated work**

* **Exact duplicate code:** almost none, because code is sampled fresh each time.
* **Redefined helpers:** 71 % of calls define functions again (mean code length 606 / 656 chars), and 83 / 56 calls redefine a diff helper.
* **Full inventory dumps:** 288 / 259 calls print the whole segmentation node list, and 131 / 127 turns *start* with one. The model re-grounds after every yield because nothing persists.
* **Rebuilding history:** 56 (A, 18 games) and 71 (B, 20 games) calls rebuild per-level history to "find what won level N". The harness already knows exactly which action completed each level.

**Actions before the first level**

| metric | A | B |
|---|---|---|
| games that cleared L1 | 20 | 19 |
| actions used / human baseline on L1 | 546 / 604 | 396 / 574 |
| median LLM calls to clear L1 | 13.5 | 12 |
| median wall time to clear L1 | 1 155 s | 1 018 s |
| actions burned in games that never cleared L1 | 1 274 (dc22 299, g50t 116, sk48 118, sp80 430, tn36 311) | 1 066 |

**Harness bugs found**

1. After `_execute_auto_reset()`, `_last_step_summary` still has `game_over=True`, so every later prompt says "The game is over." `last_action_result['game_over']` is also still True. The system prompt tells the model to "stop acting immediately" on `game_over`. This produced 207 stale prompts.
2. `_extract_labeled_blocks` only matches `^<label>:`. It rejects `World model (revised):`, `World model v27:`, `World model update:`, `World model L2 (rev):` and `**World model (rev, L2):**`. These are the dominant forms in the transcripts.
3. `_update_summarized_knowledge_from_step_summary` clears everything except `cross_level_notes` on a level transition, and the model almost never writes that label.
4. `_grid_from_state` uses `state.frame` (the last frame only). `taaf.GameState.animation_frames` exists but is never exposed.

## 4. Prioritised harness and prompt changes

Impact estimates are rough. They are based on the call and time budgets above; "games" lists where the transcripts show a concrete loss.

| P | change (implementable) | expected impact | games helped |
|---|---|---|---|
| **P0-1** | **Fix game-over handling.** After an auto-reset, set `_last_step_summary = {game_over_reset: True, …}` and clear `last_action_result['game_over']`. Prompt text: "The previous attempt ended (game over #k); the level was reset. Keep playing — use what you learned." Limit "stop immediately" to `level_completed/run_complete` within a batch. | Removes 207 stale prompts and 16/38 idle turns; up to 65 min in dc22-B. Near-zero effort. | dc22, tu93, sp80, tn36, bp35, ft09, su15, wa30 |
| **P0-2** | **Expose animation frames.** Add `last_transition.animation_frames` (a list of frames) and an auto-summary in the action result: "animation: K frames; max N cells changed in bbox …; final board reverted: yes/no". | Makes sp80's pour and tn36's program execution observable. That is at least 2 zero-score games, and prevents false "action is a no-op" conclusions. | sp80, tn36, g50t, sb26, sc25, su15, bp35, r11l, lf52, cd82, ft09, ls20 |
| **P0-3** | **Make memory robust.** (a) Relax label parsing to `^\W*(world model|goal model|…)\b[^:\n]{0,40}:`. (b) If the content has no labels, fall back to the last 600 chars of assistant content, or a label found in the reasoning. (c) Better still, add a python-side `notes` dict with `note(key, text)` that persists across calls and turns and is rendered into every prompt (capped at about 1.5k tokens). | Carried knowledge goes from 30 % of updates to about 100 %. Removes the model's need to rebuild state after every trim or yield. | all; especially cd82-A, sc25-B, wa30-B, g50t-A (0–7 notes) |
| **P0-4** | **Cross-level memory by construction.** On a level transition, do not wipe the notes; move them to a "previous levels" block. The harness also appends a **win record**: level, #actions, the last 10 actions, and a compact diff of the winning transition. Keep a per-game "mechanics learned" list. | Cuts the 56–71 history-reconstruction calls and the rediscovery on every new level (about 3–6 calls per level). With about 55 calls per game, this is likely +0.5–1 level on games that clear L1. | ar25, s5i5, re86, vc33, ls20, lf52, ka59, cn04, m0r0 |
| **P1-5** | **Stuck detection and strategy reset.** Inject a reset prompt when any of these hold: 3 consecutive non-acting turns; 12 calls or 60 actions on a level without progress; or the same hypothesis family repeated (for example, more than 3 notes containing "sweep/enumerate/untested"). The reset prompt should include: the tested-hypothesis list from notes, an **untested interaction list** from P1-6, a hard "act within this turn" instruction, and a ban on re-analysing past levels for more than one call. | Targets the 11–21 % of time lost in gaps of more than 15 min and the brute-force loops. | re86, r11l, lp85, ls20, m0r0, tr87, cn04, tn36, ft09, sb26, dc22 |
| **P1-6** | **Interaction-coverage tracker (harness side).** For every executed action, record (action type, clicked object hash or colour, context) together with the effect class: none / HUD-only / gameplay / game-over. Show a compact table of untried object×action pairs. | Prevents tn36's "311 clicks on 33 cells", sp80's single SPACE test, and dc22-A never walking over the toggled bridge. | tn36, sp80, dc22, g50t, sk48 (ACTION7 = undo never used), su15 |
| **P1-7** | **Auto-diff with HUD masking in every action result.** Detect HUD cells as those that change monotonically on almost every step (bars, counters) and exclude them. Report moved, appeared and vanished objects by hash; add `gameplay_changed` alongside `board_changed`; include `time_remaining_s` and a calls-left estimate. | Saves about 1 inspection call per acting turn (45–49 % of calls are inspection-only) and removes HUD false positives. | all; dc22, r11l, lp85, ls20 especially |
| **P1-8** | **Persistent sandbox namespace plus a helper library.** Keep user globals between calls. Preload `diff(a, b, ignore_hud=True)`, `objects(frame)`, `crop()`, `bfs(start, goal, passable)`, `simulate_check(model_fn)` (replays history against a user transition model and reports the first divergence), and `level_start_frame(n)`. Allow `difflib` and `time`. | Removes the 71 % function redefinition, the NameErrors and disallowed imports (about 4 % of wall time), and many simulator-bug turns (lp85, sb26, m0r0). | lp85, sb26, m0r0, s5i5 (routing), ka59, tu93 |
| **P2-9** | **Context management.** Keep the image only on the newest user message. Estimate images at a fixed about 1.1k tokens instead of `len(base64)/3`. Drop `reasoning` from stored history. On a yield-retry send a short "continue; state unchanged" message instead of the full 3.2k-char prompt with a new image. | Roughly doubles retained history, from about 8 to about 16 calls. It also cuts prefill per call, which is shared across the 25 streams. | all |
| **P2-10** | **Latency and throughput.** Cap thinking per request (for example, a reasoning budget of about 1.5k tokens, with a larger budget only when stuck or on a new level). Run fewer games concurrently, or enable MTP/speculation consistently. Raise `yield_seconds` above one request (it is 60 s against about 142 s per request). | Each 2× in calls per game roughly doubles the exploration budget. The level-1 median is about 13 calls, and about 55 are available. | all |
| **P2-11** | **Prompt edits.** (a) "A game over is information; the level restarts; continue." (b) "An action that leaves the final frame unchanged may still have done something — check `animation_frames` before ruling it out; test each action type in at least two contexts." (c) "Look for a submit/execute/pour trigger when the board is a set-up puzzle." (d) Give a budget line: "~N LLM calls remain; prefer probes that act." | Cheap. Addresses sp80, tn36, dc22-B and the paralysis cases. | sp80, tn36, dc22, g50t, tr87, re86 |
| P3-12 | Minor. Report `executed_count` and `requested_actions` for batched MOUSE clicks explicitly (r11l-B hallucinated dropped clicks). Surface the 30 s python limit before long searches. | Small. | r11l, lp85 |

**Suggested order:** P0-1 and P0-3 first (a day, pure harness fixes), then P0-2 and P0-4, then P1-5 to P1-8, then the P2 tuning. Rough expectation: P0 alone should convert sp80/dc22-type blocks and cut about 15–25 % of wasted calls. The level-rate gains depend mostly on P1-5 and P1-6 reducing the 61–64 % of time spent stuck on the final level.
