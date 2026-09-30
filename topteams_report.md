# Top-team intelligence: ARC-AGI-3, ARC-AGI-2, Gemma 4 Developer Agent

Snapshot 2026-09-30 ~14:30 UTC. Raw material is in `/tmp/topteams/`:
- leaderboard CSVs per competition
- `hist_arc3.json` / `hist_arc2.json`: Tong Hui Kang's minutely LB monitor (https://arc3.huikang.dev/leaderboard), with per-day score and runtime in minutes
- `topics/*.txt`: forum threads
- `nb2/`, `nb3/`, `nbg/`: pulled notebooks, with `.txt` flattenings
- `models.txt`, `all_by_user.txt`: artefacts of every top-15 member

Nothing was pushed, submitted or created.

Legend:
- **[V]** verified from the LB, code, model card or host post.
- **[C]** a claim by a third party, not verified.
- **[S]** our speculation or inference.

Deadlines: ARC-AGI-2 and ARC-AGI-3 end Nov 2. The Gemma agent competition ends Dec 2. **The ARC-AGI-3 Milestone-2 open-source deadline is today, Sep 30.**

---
## 1. ARC-AGI-3 (us: 3.66, rank 477/3504; #10 = 15.68, #50 = 5.65, #100 = 4.91)

Top 15:
- Scores and the trend are [V] from the LB history.
- Every top team's runtime is 530-550 min per submission [V], the same as ours (535-541). **Nobody is ahead because they use more wall time.**

| # | Team (members) | Score | Subs | Sep 18 → 29 | What is public / known |
|---|---|---|---|---|---|
| 1 | Tufa Labs (dlorah, driessmit1, infinitecreativity, jeroencottaar, pressman1, stefano1283) | 45.33 | 150 | 18.8 → 27.3 (9/26) → 45.3 (9/28) | Authors of the Duck harness (M1 writeup, topic 717133). Declined to open-source for M2 (topic 742801). The only recent artefact is the dataset `driessmit1/qwen3-8-27b-fp8-hf` (Aug 18). Method unknown [V]. |
| 2 | Yi-Chia Chen (threerabbits) | 40.80 | **16** | 4.8 → 12.9 → 18.8 → 28.3 → 36.7 → 40.8 | Declined to share. Their Kaggle models (June, probably for ARC-2) are `opd-32b-v33` GPTQ-W4A16 and `dflash-32b-draft`, i.e. on-policy distillation plus DFlash speculative decoding [V exists / S use]. |
| 3 | Daniel Franzen (dfranzen) | 27.24 | 87 | 13.1 → 16.7 → 21.0 → 26.6 → 27.2 | **Will open-source if 1st among teams willing to share** (topic 742935) [V]. Nothing public yet. |
| 4 | the last dance (dwellement0baser, fses91, gklambauer, lukasaichberger) | 24.54 | 67 | 6.0 → 24.5, steady daily climb | Nothing public. |
| 5 | Lord Han Solo (lordhansolo) | 23.84 | 78 | 11.5 → 19.4 → 23.8 | **Public Kaggle model `lordhansolo/swift-1-5-qwen3-8-flash-next-nvfp4`** (uploaded 9/28), a mirror of UkisAI Swift 1.5 [V]. Will share if top 3 among sharers [V]. |
| 6 | Tong Hui Kang (huikang) | 20.53 | 88 | 8.7 → 15.0 (9/24) → 20.5 (9/25) | Runs the LB monitor. Will "very likely" share if not in the top 3 of sharers [V]. Private `nemotron-adapter` models (from AIMO). |
| 7 | face-of-agi (cmechevalier, richardcsaky) | 20.00 | 39 | 7.2 → 20.0 (9/29) | June notebook: Qwen3.6-35B-FP8 plus a "world-goal-agent" wheelhouse. Outdated. |
| 8 | Third Intelligence (joshkun, seele1917, takahashinaoki, yukiokumura1) | 18.29 | 61 | 8.2 → 18.3 (9/27) | Nothing public. |
| 9 | NVARC3 (cpmpml, darraghdog, eladsarafian, galkaplun, sorokin, yeyinzhu) | 16.07 | 24 | flat since 9/19 | Nothing. Open-sources only at the end, if gold. |
| 10 | Son Pham & Mark Barney (sonphamorg, markbarney) | 15.68 | 63 | 7.4 → 12.2 → 15.7 (9/28-29) | Datasets: an SGLang serving kit for Flash-Next (parts A-C) plus an **FP8-PLE shard set** that loads PLE in FP8 (51 GB less to read). They tried DeepSeek-Flash; its token efficiency was worse than Flash-Next [C, topic 742788]. |
| 11 | rellik13 (sirikilohit) | 14.49 | 41 | 6.4 → 14.5 (9/25-29) | Nothing relevant. |
| 12 | Skyfall AI (alykassem202 +5) | 13.19 | 40 | 7.7 → 13.2 (9/29) | Mirror of keithtyser's vLLM runtime, labelled "duckWM" (Duck + world model?) [S]. |
| 13 | Matija L & Zhongwei W & Fususu | 11.64 | 179 | flat | Kaggle model `phuongncn/arc3-dsv4-flash-weights` (DeepSeek-V4-Flash). Pinned Tufa v21 source. |
| 14 | keithtyser | 11.04 | 93 | **4.17 → 8.62 → 9.17 → 11.04 (9/27-29)** | Author of our public base. His current private version is 2.6x his public one [V LB]. |
| 15 | Chew Kok Wah | 9.98 | 110 | 7.0 → 10.0 (9/28) | Qwen3.8-27B-FP8 and TAAF forks. |
| 16 | Michael Poluektov | 9.40 | **7** | 2.9 → 5.1 (9/28) → 9.4 (9/29) | **Published `michaelpoluektov/qwen3-8-swift-nvfp4` on 9/29**: Swift 1.5 with FP8 PLE and NVIDIA's MTP head, 132.7 GB, the same footprint as our RadixArk checkpoint [V]. |

### Key findings
1. **A simultaneous jump happened on 9/24-9/29 and lines up with the release of Swift 1.5 Qwen3.8-Flash-Next.**
   - Timing [V]: the HF repo `ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4` was created 9/22 and finalised 9/24. At least 10 of the top-16 teams jumped 1.5-3x between 9/24 and 9/29.
   - The model [V]: an RL+OPD post-train of Flash-Next.
     - Its card reports 44-64% fewer thinking tokens and a ~1.8x speed-up.
     - Accuracy is about equal: GPQA 89.8 → 89.6, LiveCodeBench 88.4 → 90.4, Terminal-Bench 67.6 → 69.7.
     - Arch `qwen4_exp`, the same as our base.
   - Kaggle artefacts [V]:
     - #5 mirrored it (186 GB, BF16 PLE).
     - #16 converted it to a drop-in (FP8 PLE plus NVIDIA MTP, 132.7 GB) and went 2.9 → 9.4 within 48 h, with 7 submissions.
     - Other copies: `cihanatak/swift15-flashnext-nvfp4-nonple` (77 GB dataset); a Swift-1.5-27B W4A16 dataset.
   - Why it plausibly helps [S]: ARC-3 is throughput-bound. There are 110 games in ~9 h, and we see 28 concurrent games on 8 vLLM sequences. Halving the thinking tokens roughly doubles the actions and turns per game.
   - Which teams use it [S]: that the top 4 use it is inference. That #5 and #16 do is near-certain.
2. **A second candidate is NVIDIA's official `nvidia/Qwen3.8-Flash-Next-NVFP4`** (created 9/2, 238k downloads). Our base is the RadixArk ModelOpt checkpoint. One forum user attributes "much of the LB boost" to "correct vLLM installation + nvidia/qwen3.8-flash-next-nvfp4" [C, topic 743952]. Poluektov's package already takes the MTP head from the NVIDIA checkpoint [V].
3. **Some public notebooks do score above us, but only in the ~5-6 range.**

   | Notebook | Their LB | What it is |
   |---|---|---|
   | `shiiin9/affectify-arc-duck-plus-ours` | 5.75 [V team] | keithtyser Duck + a one-cell "attention lottery" scheduler (`OURS_SCHED=1`, width 8, anneal, stall knee 80) |
   | `sunchengyuu/...` | 5.23 [V] | — |
   | `scottlegrand/taaf-flashnext-sheetu12b-0922` | 5.19 [V] | agentfix: one image per request, HUD-band-aware change detection, no-op/loop guards, ACTION7 fix, 16k analyzer context |
   | `fantasy0312/6-02-tufa` | "6.02" in the title; the team shows 4.15 [V] | the claim does not match its own LB entry |

   - The same code scored 5.68, 4.09 and 2.30 [C, shiiin9]. These are lottery-level differences that no public notebook reproduces reliably.
   - **No public notebook comes anywhere near the 10-45 of the top 16.**
4. **Noise and the endgame.**
   - The same build spans 4.5-9.4 on public-25 [C, Makarov] and 2.3-5.7 on the LB [C].
   - Public-25 gains do not transfer: our own v5 scored 10.3 locally and 2.53 on the LB [V].
   - Host [C, topic 697944]: there is no rerun at the end. The private score of every submission is already computed on the other half of the ~110 games, and we pick 2 of them.
5. **Fine-tuning is unproven for the compute-poor.**
   - SFT of Flash-Next with Qwen-Max or Astra teachers gave no gains [C, blakewest, H200s].
   - STaR trajectory-LoRA on a 27B model moved the LB 1.25 → 1.94 in August [C, topic 739047].
   - An RL pipeline on Flash-Next gave nothing yet [C].
   - Distilling from Astra conflicts with OpenAI's terms and with prize open-sourcing.
6. **REAP-448E** (`boristown/qwen3-8-flash-next-nvfp4-reap-448e`, 124 GB) prunes the routed experts from 512 to 448 so that 128K context fits [V]. Team 暗黑AGI is at 7.43 (#30), so this is not a proven winner.
7. **Milestone-2 (today) may make 3 top-6 solutions public within ~24 h**: #3 Franzen, #5 Lord Han Solo and #6 Tong Hui Kang. Each is conditional on ranking among the teams willing to share. Tufa and Yi-Chia have declined [V].

---
## 2. ARC-AGI-2 (us: 28.61 from 1 sub, rank 1091/2268; #10 = 36.25, #20 = 33.47, #100 = 32.22, #300 = 31.39)

| # | Team (members) | Score | Subs | Runtime (min) | History / known |
|---|---|---|---|---|---|
| 1 | Tufa Labs (driessmit1, jeroencottaar, pressman1, stefano1283) | 83.06 | 39 | ~725 | 73.5 → 83.1 on 9/22. Private. |
| 2 | rabbithole (cookizesong, xiezejian, yuhuake240506) | 79.72 | 103 | ~735 | >70 since Aug. Private. |
| 3 | nvbanana (cpmpml, darraghdog) | 75.42 | 130 | 540-720 | "Different weights" than NVARC-2025 [C]. Open-source only at the end. |
| 4 | Yi-Chia Chen (threerabbits) | 55.14 | 11 | ~620-690 | 31 → 55 over 9/6-9/15. Kaggle models: OPD-32B GPTQ-W4A16 + DFlash 32B draft (June) [V exist, S use]. |
| 5 | Kha Vo (khahuras) | 37.50 | 98 | ~700 | flat |
| 6 | okiek (notnitsuj) | 37.36 | **8** | 330-715 | 28.9 → 30.1 → 37.4 (9/23-28). Nothing public. |
| 7 | Junhua Yang (junhuay) | 37.22 | 183 | ~700 | flat, 183 subs (seed lottery?) [S] |
| 8 | Nguyen (nguyennguyen599) | 36.94 | 64 | 70-780 | AIMO fine-tuner (gpt-oss, Qwen3-4B-thinking models). Nothing ARC-specific public. |
| 9 | Anil Thomas (anlthms) | 36.81 | 13 | 530-840 | 30.1 → 31.1 → 35.7 → 36.8 |
| 10 | AI Winter (aiwinter, stoneshek) | 36.25 | 78 | ~710 | 35.4 → 36.25 |
| 11 | Dawid Kopiczko (dawkopi) | 35.56 | 24 | ~700 | 32.4 → 35.6 (9/29) |
| 12 | Hiểu Vy (thangnm1) | 34.86 | 149 | ~690 | flat |
| 13 | iga (tangtang1999) | 34.03 | 74 | 500-660 | |
| 14 | Team BlackBox (koushikrudra, poby7722 +3) | 33.89 | 128 | ~720 | Authors of the public 33.89 lineage (`failed-in-aimo`, `arc-agi2-original-kg`) |
| 15 | Mithil A Vakde | 33.89 | 45 | ~230 | inactive since July (mdlARC model) |

### Key findings
1. **Ranks 5-15 have no public code, writeups or models.**
   - A few climbed with few submissions: okiek, Anil Thomas, Kopiczko. That points to real method changes rather than seed luck. What they did is unknown [S].
   - Places 1-4 use different model families or weights and are not reproducible [V/C].
2. **No public notebook is verified above 33.89.** Recent public ones sit below it:

   | Notebook | Team LB | Note |
   |---|---|---|
   | `junaid512/arc-agi-31-11` (139 votes) | 31.11 [V] | a refactor of perfpatch |
   | `finalsunflower/arc-agi-2-nvarc-plus` | 31.39 [V] | — |
   | `medvax/...symbolic-fallback` | 28.89 [V] | [C] on its own 120-task eval: selector `score_full_probmul_3` = 29.17 vs `score_kgmon` = 28.33; its train-exact program fill for empty attempt_2 gives a small gain |

3. **Our 28.61 is one draw** from a pipeline whose identical reruns span 26.9-32.2 [C, luxluxshan]. The median public fork sits at ~31.4-32.6 (ranks 100-300). One or two resubmits are likely worth +2-4 in expectation just from variance [S].
4. **The known paths above 34 are model-level work: new SFT data or new weights.** CPMP: "simple tweaks won't be enough". NVARC-lineage fine-tuning attempts on the forum have failed [C, topic 733930].

---
## 3. Gemma 4 Developer Agent (us: 0.12 = 7/58 tasks, rank 104/1079)

**Unit [V, dariushafshar]:** the public LB has **58 tasks**, so 1 task = 0.0172.
- 0.17 = 10 tasks, 0.15 = 9, 0.13 = 8, 0.12 = 7.
- 107 teams are tied at 0.12. #1 is 3 tasks above us; #2-6 are 2 above.
- One binomial SD at the cut ≈ 2.3 tasks.

| # | Team (members) | Score | Subs | Last sub | Known |
|---|---|---|---|---|---|
| 1 | Statistical Science by Lasme (sslasme) | 0.17 | 7 | 9/30 | nothing public |
| 2 | Makus (alexmartinez75) | 0.15 | 4 | 9/30 | nothing |
| 3 | sergiomalv | 0.15 | 6 | 9/30 | nothing |
| 4 | GopiPitchai | 0.15 | 2 | 9/29 | nothing |
| 5 | Gabriel (gabeshaps) | 0.15 | 5 | 9/29 | nothing |
| 6 | Baidalin Adilzhan [dsmlkz] (baidalinadilzhan, kozykappa) | 0.15 | 4 | 9/29 | **`kozykappa/top-2-gemma-eda-lb-score-0-15` is cell-for-cell identical to romanrozen's 0.12 notebook** (diff empty) [V]. Black Cat says the zip is byte-identical too [C]. |
| 7-15 | Chandan Ranjan, 暗黑AGI (boristown), Alperen ÖZ, Daniel Olajide, Jeki Wan Taufik, Daniel Du, I Zaznov, Howon Kang, DoRA | 0.13 | 1-7 | 9/26-30 | Nothing public. Alperen notes his 0.13 differs from a failed bundle "only in thinking settings and timeout_seconds" [C]. |

### Key findings
1. **The top of the board is statistically indistinguishable from the 0.12 bundle.**
   - The same romanrozen bundle has scored 0.12 and 0.15 [V].
   - Black Cat resubmitted their byte-identical anchor and scored 0.10 twice [C].
   - The LB spread (SD 1.72 tasks) is what one re-rolled agent would produce (1.74) [C, dariushafshar].
2. **Harness and platform changes are landing now** (host posts) [V]:

   | Issue | Status |
   |---|---|
   | Thinking-mode thoughts dropped between tool calls (vLLM ignored `reasoning_content`) | patch "deploy later today" (9/29) |
   | Tool results double-JSON-encoded | "addressing now". [C] measurement: 22% edit_file failures now, 0% with raw text |
   | LoRA zeroing | fixed in wheelhouse v23 |
   | With LoRA on 4xL4, the KV cache is ~7.6k tokens and prompts >7.6k hang | open |
   | 12 h overrun | still fails the whole submission; "unfinished = 0" planned, not live |
   | Task execution | tasks run **sequentially** |
   | Patch validation time | ~2 h, excluded |
   | Test set | ~120 tasks, public + private in the same run |
   | Distillation | allowed from models whose licence permits it (answered 9/29) |

3. **Short and generic beats long and rule-heavy.**
   - Black Cat's lab-rule prompts scored 0.05-0.06, while the short 450-word romanrozen prompt with default budgets scored 0.12-0.15 [C].
   - This matches our v1 (0.06).

---
## 4. Prioritised actions

GPU-hour estimates:
- ARC-3: 1 RTX-Pro-6000 run on the public-25 ≈ 2.3 h; a submission scoring run is ~9 h of Kaggle compute.
- ARC-2: 1 full 4xL4 eval ≈ 6-8 h, charged at 2x.

| P | Comp | Action | Evidence | Expected gain | Cost | Risk |
|---|---|---|---|---|---|---|
| **1** | ARC-3 | **Swap the model to Swift 1.5 Flash-Next.** Attach Kaggle model `michaelpoluektov/qwen3-8-swift-nvfp4` (FP8 PLE + NVIDIA MTP, 132.7 GB, the same format and footprint as our RadixArk checkpoint) in place of `keithtyser/qwen3-8-flash-next-nvfp4`. Keep the harness at v2. Smoke-test that vLLM loads it with our runtime (arch `qwen4_exp`, as in our base). Then run the public-25 once and submit. | [V] release timing, model card (-44-64% thinking tokens), top teams' artefacts, Poluektov 2.9 → 9.4 in 48 h. [S] that the top-4 use it. | Large if the causal story holds: the teams that jumped went 1.5-3x. [S] Plausibly 3.7 → 6-10. | ~1 h smoke + 2.3 h public-25 + 1 submission | Medium. Load or patch incompatibility is possible. Licence: Swift Open License + Qwen Community; check OSI-eligibility before relying on it for prizes. Fallback option: #5's BF16-PLE mirror (186 GB; may not fit without PLE FP8 conversion). |
| **2** | ARC-3 | **Check Milestone-2 releases on Oct 1 (00:00-03:00 UTC):** `kaggle kernels list --user dfranzen / lordhansolo / huikang --sort-by dateRun`, plus topic 742935. If a 20+ solution is published, diff it against v2 and adopt its model, serving and scheduling. | [V] their stated intentions | Potentially the largest available jump (20+ LB-verified code) | ~0 to read; 1 submission to test | Low. Releases are conditional and may not happen. |
| 3 | ARC-3 | A/B the plain `nvidia/Qwen3.8-Flash-Next-NVFP4` checkpoint (needs a Kaggle upload by someone; none found attached) only if P1 fails to load. | [C] forum only | small-to-medium, unknown | 2.3 h + upload | Medium |
| 4 | ARC-3 | **Submission policy:** stop judging harness tweaks on single LB runs. Submit the chosen config several times; for finals pick the best-public copy of the best config plus one *different* model/harness. | [V] private scores are pre-computed; our v1-v6 spread ±0.5; same-code spreads of 2.3-5.7 in public notebooks | Protects against lottery loss | 0 | Low |
| 5 | ARC-3 | Port only the cheap scheduler cell from shiiin9 (`OURS_SCHED`, one cell) after P1, if public-25 does not degrade. | [V] its best LB 5.75, but the same code also scored 4.09 and 2.30 | ~0-1 | 2.3 h | Noise-level |
| 6 | ARC-2 | **Resubmit perfpatch** with deterministic seeds (`PYTHONHASHSEED=0` + md5 stable hash). Use `score_full_probmul_3` as the attempt_1 selector if the saved-pickle eval agrees. Use a daily submission slot; commit with the 4-task smoke only. | [V] our 28.61 is 1 draw; public forks cluster 31-34; [C] medvax selector +0.8 on eval | +2-4 expected from variance alone | ~1 h commit/day | Low |
| 7 | ARC-2 | NVARC+ two-pass pooling (luxluxshan) + a train-exact program fill for empty attempt_2 (medvax/finalsunflower). Evaluate on the 120-task eval pool before any LB use. | [C] 32-36 on the runtime board for NVARC+ | +1-2 over perfpatch | 6-8 h 4xL4 eval | Low-medium |
| 8 | ARC-2 | Anything beyond 37 needs new SFT data or weights (continued SFT of `qwen3_4b_grids15_sft139` / the 2B model). No public evidence that the ranks 5-15 recipe is reachable. | [C] CPMP; failed forum SFT attempts | unknown | >50 GPU-h off-Kaggle | High; deprioritise |
| 9 | Gemma | **Resubmit the exact 0.12 bundle 2-3 times** before changing anything. The top 6 are within ~1 SD. | [V] kozykappa copy = 0.15 | +0-3 tasks (lottery), and it calibrates our noise | 1 submission/day, 0 of our GPU | Low |
| 10 | Gemma | After the host confirms the thinking-mode patch is live, test the base bundle with thinking on (`include_thoughts` true, a moderate `max_output_tokens`, plus an `eval_config` time cap so the whole run fits in 12 h). | [V] host patch; [C] Alperen's 0.13 varied thinking/timeout | unknown; the first real lever after the fix | local eval on dev tasks (4xL4, several hours) | Medium: a 12 h overrun fails the whole submission |
| 11 | Gemma | Keep prompts short. Do not stack harness rules. Do not ship LoRA until the 7.6k-token KV issue is fixed. Re-check tool-encoding behaviour after the double-JSON fix. | [V] host posts; [C] Black Cat ablations | avoids losses | 0 | Low |
| 12 | Gemma | For the 2 final picks (due Dec 2), use two *different* agents, not two near-copies. | [C] dariushafshar calculation (+1.02 vs +0.80 expected private tasks) | small but free | 0 | Low |

### Verified vs speculative, in one line each
- **Verified:**
  - LB scores, submission counts, dates and daily runtimes.
  - The Swift 1.5 release dates and model card.
  - The existence and contents of the Swift Kaggle mirrors by #5 and #16.
  - kozykappa = romanrozen code.
  - Public notebook LB claims (5.75, 5.19, 31.11, 31.39, 28.89).
  - Host statements on the harness.
- **Speculative:**
  - That the top ARC-3 teams' jumps come from Swift 1.5.
  - That Yi-Chia uses OPD/DFlash.
  - What ARC-2 ranks 5-15 do.
  - Expected gains.
