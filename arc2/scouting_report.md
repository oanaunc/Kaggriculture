# ARC Prize 2026 - ARC-AGI-2: scouting report
Snapshot: 2026-09-27. Our team has 0 submissions so far (1 remaining today).
Raw material is under /tmp/arc2/: `pages.txt` (competition pages), `data/`, `lb.txt`, `lb_runtime.json` (community runtime history),
`nb/*/` (pulled notebooks + `.txt` flattenings), `out/*/` (notebook outputs and logs), `topics/*.json` (forum threads).

## 1. Rules and limits
| Item | Value |
|---|---|
| Deadlines | Entry and team-merger deadline **Oct 26 2026**. Final submission deadline **Nov 2 2026, 23:59 UTC** (about 36 days / 36 submissions left). Winners are announced Dec 4. |
| Submissions | **1 per day**. We pick 2 final submissions. The private LB decides; ties go to the earlier submission. |
| Team | Max 5 members. Merging is allowed until Oct 26 if the combined submission count fits within the allowance. |
| Metric | Each test output gets 2 attempts and needs an exact grid match to score 1. Score = mean over all test outputs. The hidden set has 240 tasks (a few have 2 test inputs). Local eval set: 120 tasks, 172 outputs. |
| Format | `submission.json` with `{task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}`. Every task id and both attempts must be present. |
| Compute | Notebook only. CPU or GPU, **<= 12 h** runtime, **no internet**. Public external data and pretrained models are allowed. Hardware is **4x L4 (96 GB total)**; L4 quota is charged at 2x the T4/P100 rate. Runtimes are obfuscated (±10 min). |
| Placeholder test | `arc-agi_test_challenges.json` holds 240 *training* tasks. It is swapped for the hidden set at rerun. Use the `KAGGLE_IS_COMPETITION_RERUN` env var to switch modes. |
| Open source | Prize winners must open-source under **CC BY 4.0**, with an open-source system, model and weights (OSI Open Source AI checklist). A writeup is due within 7 days of the deadline. Only OSI-licensed code may be used. |
| Prizes | Progress prizes for places 1-8: $75k/50k/40k/35k/25k/20k/15k/15k. Innovation/"Grand" writeup prize: $275k. Bonus of $150k if a team reaches 85%. |
| Gotchas (forum) | Accelerators get switched off automatically: always "Save" (commit) before submitting. A Kaggle system error can refund the day's submission. Some people report CLI kernel submission failing, so keep the UI as a fallback. |

## 2. Leaderboard snapshot (public LB, 2026-09-27)
| # | Team | Score | Subs | Typical runtime |
|---|---|---|---|---|
| 1 | Tufa Labs | **83.06** | 36 | ~12 h |
| 2 | rabbithole | 76.94 | 100 | ~12 h |
| 3 | nvbanana (CPMP / Sorokin, the 2025 NVARC winners) | 74.17 | 127 | 8-10 h |
| 4 | Yi-Chia Chen | 55.14 | 11 | ~11.4 h |
| 5 | Kha Vo | 37.50 | 95 | ~11.5 h |
| 6 | Junhua Yang | 37.22 | 179 | ~11.5 h |
| 7 | Nguyen | 36.94 | 60 | 6-12 h |
| 8 | AI Winter | 36.25 | 75 | ~11.8 h |
| 9 | Hieu Vy | 34.86 | 146 | ~11.5 h |
| 10 | Team BlackBox | 33.89 | 126 | ~12 h |

The distribution of 2,214 teams: 4 teams above 50, 4 teams at 35-40, **759 teams at 30-35**, 476 at 25-30 and 807 below 5.
The 30-35 cluster is almost entirely forks of the public NVARC notebook. Places 5-10 differ by ~3.5 points, which is about the
**seed noise** of that notebook: identical reruns score 26.9 / 29.7 / 31.4 / 31.8 / 32.2 according to the luxluxshan notebook.

The top 3 have private methods that are far ahead. nvbanana (the NVARC team) has new weights; CPMP says "simple tweaks to our
last year winning solution won't be enough". A forum user suspects NVIDIA's `nvidia/Nemotron-SFT-ARC-AGI-v1` code-execution
SFT data is involved. Places 1-4 are out of reach within 5 weeks. **The realistic target is places 5-8** (currently 36.25-37.5 public).
Public-to-private shuffles of ±3 points are likely.

## 3. Top public notebooks
The NVARC 2025 winner lineage underlies all of them. The model is `sorokin/qwen3_4b_grids15_sft139`: a Qwen3-4B SFT'd on synthetic
ARC puzzles with a 16-token grid vocabulary, **Apache 2.0**, fine-tuning allowed. The pipeline is: per-task LoRA test-time training
(r=256, rsLoRA, lr 5e-5, 1 epoch over 128 augmented views) → batched "turbo DFS" decoding over 16 views (8 dihedral transforms x 2 colour permutations),
pruned at p>=0.2 → teacher-forced NLL re-scoring of every candidate under 8 augmentations → `score_kgmon` selection
(#views producing the grid − mean augmented NLL). It runs 4 workers, one per L4, from a shared queue, with a 1200 s per-task cap and a 12 h − 10 min global stop.
Every notebook also uses the utility script `sorokin/pip-install-unsloth-flash-patch` (unsloth offline install). Public code is deemed OSI-licensed (Rules 3.6b).

| Ref | Approach / delta | Reported score | Extra dependencies |
|---|---|---|---|
| `koushikrudra/failed-in-aimo` (566 votes) | Origin of the "LB 33.89" line. Plain NVARC 4xL4. The current version adds an optional Qwen2.5-Coder-7B DSL-induction pre-pass, but that model is not attached, so the pre-pass is inert. | 33.89 (per its forks) | qwen3_4b only |
| **`mikelou1/arc-agi2-lb33-89-minimal-perfpatch`** (116) | Same maths as failed-in-aimo. Keeps logits on GPU, gathers only the 12 ARC tokens and disables the KV cache when scoring, so more tasks finish in the budget. Cleanest base. | **33.89** (title) | qwen3_4b only |
| `qiuqiuh/arc-highscore-lb3389-replica` (53) | Byte-identical copy of mikelou1 apart from print lines. | ~33.9 | same |
| `koushikrudra/arc-agi2-original-kg` (105) | Perfpatch reverted. Adds a deterministic scoring seed (`PYTHONHASHSEED` plus a stable key hash) and model-path discovery. **Bug:** the newline token key is mojibake (`"ÄŠ"` instead of `"Ċ"`) and should be checked before reuse. gpt-oss-120b is attached but unused. | n/a | gpt-oss-120b (unused) |
| `luxluxshan/arc2-nvarc-v1` (79) | "NVARC+": two independent half-cost TTT passes (64 train seqs / 8 views each, different seeds, cheap-first order, 700-800 s cap) pooled by KGMoN, with pass-A top-1 forced into the attempts. Documents the 26.9-32.2 rerun spread. | not stated. The half-cost config reportedly scored 33.89-35.97 on the runtime board. | qwen3_4b only |
| `finalsunflower/arc-agi-2-nvarc-plus` (24) | Full NVARC plus stable seeds. attempt_2 is an independent probability-ranked candidate. The SOAR-7B program-synthesis sampler runs on GPU 3 and train-verified programs fill empty or duplicate slots. A train-exact geometric filler is also used. | "31.39 submission" (earlier version) | `pourceljulien/soar-qwen-7b`, dataset `yhay81/arc-agi-2-atlas-program-corpus-data` |
| `yusuketogashi/arc-baseline-rebuild` (83) | NVARC primary pass (<9 h), then a bounded retry (≤24 tasks, 75 min) on tasks with empty or single-candidate outputs. A retry may only add attempt_2. Symbolic fallback. Heavily defensive engineering. | n/a | qwen3_4b only |
| `nihilisticneuralnet/baseline-nvarc-...-t4x2` (210) | NVARC ported to 2xT4. Historic. | ~19 (T4) | qwen3_4b |

Other resources: `sorokin/qwen3_2b_grids15_sft141` (a smaller NVARC model, now fine-tunable). NVARC's GitHub repo includes TRM
checkpoints. `nvidia/Nemotron-SFT-ARC-AGI-v1` (HF). Public 35B ARC post-trains "Nex-N2-mini" and "Aquila-mini" (forum topic 732723).
The best LLM-agent submission reported so far scored 13.75.

Local evidence: the perfpatch and NVARC+ commit logs each solve **3/4** on the 4-task smoke subset. Per-task wall time on one L4
is 520-1280 s at full config, so 240 tasks / 4 GPUs is about 60 tasks per GPU in ~11.8 h, which means ~700 s per task on average.
**The budget is binding**: slow tasks hit the 1200 s cap, and whatever time a speed-up saves can go to more views.
CPMP reports that NVARC's public-eval score (~30%) tracked its public LB score (~27%) well, so the 120-task eval set is a usable proxy for the Qwen pipeline.
Other model families showed up to 50% drops from eval to LB.

## 4. Recommended base
**`mikelou1/arc-agi2-lb33-89-minimal-perfpatch`** with **deterministic seeding** added (the stable-hash fix from koushikrudra
without the mojibake bug, plus `PYTHONHASHSEED`). Reasons:
- It is the cleanest and most-forked 33.89 lineage.
- It needs only one model plus the unsloth utility script.
- The license is clean (Apache 2.0 weights, public OSI code), so a prize would stay eligible.
- It has zero dependence on unverified extra models.

Pull in two ideas from other notebooks: NVARC+ pooling (luxluxshan) and a verified-program overlay for attempt_2 (finalsunflower/SOAR).

## 5. Feasibility and prioritised improvement plan
Constraints: 1 submission/day (~35 left) and ±3 points of seed noise per submission. **A single LB result cannot confirm a
change smaller than about 3 points**, so decisions must come mostly from the local eval set. Kaggle GPU quota is ~30 h/week,
and L4x4 counts double, which leaves roughly **15 L4x4-hours per week**. A full 120-task local-eval run at production config takes ~6 h of 4xL4.
That allows at most ~2 full eval runs per week per account (each team member has their own quota, so parallelise across members).

What runs on CPU (locally, here) versus what needs Kaggle GPU:
- **CPU:** selection algorithms and ensembling over saved candidate pickles; submission validation; symbolic/DSL solvers and program verifiers; scheduling simulations using logged per-task times; checks on the eval-set proxy.
- **Kaggle GPU:** TTT, DFS decoding, re-scoring, SOAR/LLM program sampling, and any timing measurements.

Prioritised plan (expected gains are rough):
1. **Reproduce first (days 1-2).**
   - Submit perfpatch unchanged, with stable seeds, to get our own anchor.
   - In parallel, run one commit in non-rerun mode after widening `starter.py`'s 4-task filter to all 120 eval tasks, and **save `/kaggle/inference_outputs` pickles plus per-task timings as notebook output**.
   - This produces a candidate pool for CPU-side experiments. Budget: one 6-8 h L4x4 run.
2. **Selection and ensembling on CPU (days 2-7, 0 GPU).** Using the saved pickles:
   - Tune the kgmon weighting (count vs NLL, per-view normalisation).
   - Make attempt_2 = best *distinct* candidate (the finalsunflower idea).
   - Add a train-pair consistency filter (output-shape and colour-set priors from the train pairs).
   - Measure top-2 vs oracle-in-pool, which gives the ceiling for selection work.
   - Cheap, fully local, and monotone-safe.
3. **Throughput and scheduling (week 2).**
   - Order the queue longest-first or by size-estimated cost; the current order is sorted keys.
   - Replace the flat 1200 s cap with an adaptive one.
   - Early-exit DFS when ≥k views agree.
   - Put the time saved into more inference views or a second seeded pass on low-confidence tasks (NVARC+ pooling, or the yusuketogashi retry of empty/singleton outputs).
   - Two half-cost passes both reduce seed variance and add coverage. That is **the most credible +2-4 points.**
4. **Verified program overlay for empty or duplicate attempt_2 slots (week 2-3).**
   - Run a DSL/icecuber-style search on CPU cores during the GPU phase: the 4xL4 box has CPUs that sit idle.
   - Optionally add SOAR-7B sampling on one GPU for the last ~1-2 h.
   - Only programs that reproduce **all** train pairs may fill a slot, so there is no downside for attempt_1.
   - Test locally on CPU against eval solutions.
5. **Hyper-parameter sweep of TTT (weeks 2-4; GPU-heavy, use teammates' quotas).** Candidates: lr (5e-5 vs 1e-4), LoRA rank 64/128 (faster), train views 64 vs 128, n_eval_aug 2 vs 3. Only run these on the 120-task eval, never tune on LB.
6. **Stretch (only with outside GPUs, weeks 1-4):**
   - Continue SFT of qwen3_4b (or the 2b, which is faster and allows more TTT views) on RE-ARC plus ARC-AGI-2 training tasks and Nemotron-SFT-ARC-AGI data.
   - High upside, but risky in 5 weeks and it would need open-sourced data and weights.
   - Skip unless a teammate has ≥8xH100 for a week.

Final-submission policy:
- Pick one robust "best mean on local eval" config and one higher-variance variant.
- Do not chase a single lucky public score, because the public noise is about the size of the gap to 8th place.

## 6. Exact steps to reproduce the best public notebook under our account
The team runs these steps; this scouting session pushed and submitted nothing.
```bash
mkdir -p ~/arc2/perfpatch && cd ~/arc2/perfpatch
kaggle kernels pull mikelou1/arc-agi2-lb33-89-minimal-perfpatch -p . -m
# edit kernel-metadata.json:
#   "id": "<OUR_USERNAME>/arc-agi2-perfpatch-repro", "title": "arc-agi2-perfpatch-repro",
#   keep "docker_image": "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868",
#   keep: "enable_gpu": true, "machine_shape": "NvidiaL4", "enable_internet": false,
#         "model_sources": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"],
#         "kernel_sources": ["sorokin/pip-install-unsloth-flash-patch"],
#         "competition_sources": ["arc-prize-2026-arc-agi-2"], "is_private": true
# optional determinism: in the ipynb, prefix the starter command with PYTHONHASHSEED=0 and
#   replace `seed=hash(bk) % 1024**2` with a stable hash, e.g. int(hashlib.md5(bk.encode()).hexdigest(),16) % 1024**2
kaggle kernels push -p .            # commit run: non-rerun mode, 4 eval tasks, ~25-60 min on 4xL4, expect "Reload score: 3.0"
kaggle kernels status <OUR_USERNAME>/arc-agi2-perfpatch-repro
kaggle kernels output <OUR_USERNAME>/arc-agi2-perfpatch-repro -p ./out   # check the log shows 4 GPUs working and submission.json
# Submit the committed version (the UI "Submit to Competition" is the reliable path):
kaggle competitions submit arc-prize-2026-arc-agi-2 -k <OUR_USERNAME>/arc-agi2-perfpatch-repro -v <VERSION> -f submission.json -m "perfpatch repro"
```
Checks before submitting:
- In the notebook settings, verify that the accelerator is **"GPU L4 x4"**, internet is off, and the Docker image is the pinned py3.11/torch 2.8 image the original uses (from "Copy & Edit"). Copy & Edit in the UI is the safest way to inherit the environment pin.
- The model path used is `/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1` (new-style mount). If attaching yields a different mount, use koushikrudra's `resolve_model_dir()`.

Expected result: 30-34 public, with about 11.5-12 h scoring time.
