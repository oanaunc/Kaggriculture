# arc2-nvarc-ours

Our submission notebook for Kaggle "ARC Prize 2026 - ARC-AGI-2". It is derived from
`mikelou1/arc-agi2-lb33-89-minimal-perfpatch` (public LB 33.89), which is NVARC Qwen3-4B per-task
LoRA TTT, then turbo DFS over 16 views, then augmented-NLL rescoring, then `score_kgmon`, on 4 x L4.

**Unchanged from the base:** model, prompts, token ids, LoRA/TTT hyper-parameters, the number of
views, the DFS threshold (p >= 0.2), rescoring, the selection algorithm, the 1200 s per-task cap and
`global_end_time = start + 12 h - 10 min`.
**What we changed:** determinism, robustness, the time budget, scheduling, and how empty slots are filled.

## Files

| File | What it is |
|---|---|
| `arc2-nvarc-ours.ipynb` | The notebook to push. It is generated; do not hand-edit it. |
| `arc2-nvarc-ours.py` | A flat, readable view of every cell. Magics are commented out, and it passes `py_compile`. |
| `kernel-metadata.json` | id `oanaunciuleanu/arc2-nvarc-ours`, private. Same model_sources, kernel_sources, docker_image, machine_shape `NvidiaL4`, GPU on and internet off as the base. |
| `src/*.py`, `src/cells/*.py` | Source of truth. `build_notebook.py` turns these into the ipynb and py files. The `%%writefile` modules are `arc_loader`, `arc_decoder`, `arc_solver`, `starter`, `arc_dsl` and `arc_submit`. |
| `base_vs_ours.diff` | Unified diff of every base module against ours, plus the base driver cells for reference. |
| `tests/` | CPU unit tests, with fake unsloth/peft/datasets and a fake model. |

To rebuild and test:
```bash
cd /home/user/Kaggriculture/arc2 && python3 build_notebook.py
/tmp/arc2/venv/bin/python -m pytest -q -p no:cacheprovider tests      # 16 passed
```

To push (lead only):
```bash
kaggle kernels push -p /home/user/Kaggriculture/arc2
```
The commit run uses the base's 4-task smoke subset, and its log should show `Reload score: 3.0`.

## Changes versus the base (diff-style summary)

### Cell 1: setup
```diff
  global_end_time = time.time() + 12 * 3600 - 600          # kept: 10-minute buffer
+ ARC_ORDER = "cheap_first"; ARC_EVAL_SUBSET = "smoke"; DSL_TIME_LIMIT = 3600
+ RERUN_MODE / TEST_PATH / SOLUTIONS_PATH defined once
```
- Setting `ARC_EVAL_SUBSET="all"` in a commit run runs all 120 eval tasks and saves `inference_outputs.tar.gz`, which gives us candidate pools for CPU selection experiments.

### arc_loader.py
No change.

### arc_decoder.py
```diff
- for key in os.listdir(store):
-     with bz2.BZ2File(...) as f: outputs = pickle.load(f)
+ if not os.path.isdir(store): return
+ for key in sorted(os.listdir(store)):            # deterministic tie order
+     if key.endswith(".tmp"): continue            # partial atomic writes
+     try: ... pickle.load ... except: skip file    # truncated pickle (worker killed)
  benchmark: no crash when no candidate is correct (np.max of an empty list)
```

### arc_solver.py
The model code (`turbo_dfs`, `inference_turbo_dfs`, `calc_scores`, trainer and collator) is byte-identical. The worker loop changes as follows:
```diff
+ stable_seed(key, salt) = md5(key|salt) % 1024**2
- aug_dataset.augment(seed=hash(bk) % 1024**2)     # varies with PYTHONHASHSEED on every run
+ aug_dataset.augment(seed=stable_seed(bk))
+ seed_everything(stable_seed(key, pass_id)) at the start of every task
+ check_tokenizer(): logs whether "0".."9", "Ċ"(\n)=10, "<|im_end|>"=15 match the tokenizer
+ trainer.add_callback(DeadlineCallback(end_time))  # TTT stops at end_time
- while not queue.empty(): key = queue.get()      # None sentinels
+ while True: main queue (get_nowait), then the retry queue
+ do not start a new task with < 120 s left (it could not finish training anyway)
+ the whole per-task body is in try/except: log the traceback, free the trainer and CUDA cache, continue
+ a rank exits after 3 consecutive failures (broken GPU), so it does not drain the queue
+ pass 0 (the base seeds, train 1 / eval 2) that crashed, or that finished without timing out
+   but produced 0 candidates for some test, is re-queued once as pass 1 (seeds 1001/1002).
+   Pass 1 files get the suffix ".r1" and pool with pass 0 in kgmon.
+   It runs only after the main queue is empty and only if >= max(600 s, first-pass time) + 60 s remain.
+ candidate rescoring stops at end_time + 60 s
- bz2 pickle written in place
+ atomic write (".tmp" then os.replace)
+ ARC_TEST_PATH / ARC_OUTPUT_DIR env overrides (tests only; defaults = base paths)
```

### starter.py
```diff
- for key in sorted(data.keys()): queue.put(key)
+ order_keys(data, ARC_ORDER): "cheap_first" sorts by a cell-count cost proxy (training cells + decode size)
- queue.put(None) x4 ; mp.spawn(local_worker, nprocs=4)
+ 4 independent spawn-context Processes + a retry queue.
+   With mp.spawn, one rank that raised killed all four ranks. Now the survivors keep draining the queue.
+ the wait for the previous rank's unsloth-import marker is bounded (1800 s), and the marker is written even if the import fails
+ hard deadline: terminate, then kill, the workers at end_time + 180 s
```

### Driver cells (new: `src/cells/10_launch.py`, `11_run.py`, `12_submit.py`)
```diff
+ (launch) write a valid placeholder submission.json immediately
+ (launch) start arc_dsl.py (nice 19) in the background on the test file
- !UNSLOTH_DISABLE_STATISTICS=1 ... OMP_NUM_THREADS=12 python starter.py --end-time {global_end_time}
+ same env + PYTHONHASHSEED=0, started with Popen(start_new_session=True), output streamed;
+ the process group is killed at end_time + 240 s if it is still alive
- submission = data.get_submission(decoder.run_selection_algo())      # [[0]] for missing attempts
+ arc_submit.build_submission(challenges, kgmon ranking, dsl predictions):
+   attempt_1/2 = model candidates in kgmon order, de-duplicated and sanitised (the model's attempt_1 is never replaced)
+   -> empty slots: train-verified DSL predictions not already used
+   -> then the test input itself as a free guess -> then [[0]]
+ validate_submission(): every task, every test, both attempts, valid 1..30 grids with 0..9 values
+   (on failure, fall back to a construction-valid submission); atomic write
+ eval mode: score model-only vs final, the base benchmark, the base reload score, and inference_outputs.tar.gz
```

### arc_dsl.py (new): CPU program search
Program families:
- 8 dihedral transforms
- integer upscale
- downscale (uniform, majority or any-foreground block)
- tiling where each tile is a dihedral copy of the input, learned per tile
- fractal `kron(mask, grid)`
- crop to the bounding box of non-background or of a colour (optionally its interior)
- object extraction with 4- or 8-connectivity, mono- or multi-colour, selected by largest, smallest, largest or smallest bounding box, most or fewest colours, a position extreme, the unique colour signature or the unique shape (selection must be unambiguous, otherwise the rule refuses)
- two-part boolean ops (and, or, xor, nor, andnot, nand), learning the output colours
- overlays of 2 to 4 equal parts with a priority order
- symmetry-completion of a hole colour

Each base program is optionally followed by a dihedral transform and/or a learned colour map. A colour map refuses unseen colours.

A program is kept only if it reproduces **all** train pairs exactly. The two simplest distinct test predictions are returned. The search runs in a separate process with a 10 s per-task limit plus SIGALRM, and a total limit of 3600 s. It takes about 8 s on 240 tasks.

## Verification (all local, CPU)

1. **Compile.** `py_compile` passes for `arc2-nvarc-ours.py`, all `src/*.py` and the non-magic cells, under both Python 3.12 and **3.11.15** (the Kaggle image is 3.11).
2. **Unit tests** (`tests/test_pipeline.py`): **16 passed**. They exercise the real `arc_solver.worker` loop, `arc_decoder`, `arc_submit` and `starter.py`, using fake unsloth/peft/datasets and a fake "identity-task" model whose DFS returns the right grid plus a view-dependent wrong beam.
   - **attempt_2:** never duplicates attempt_1. The model's attempt_1/attempt_2 are never displaced. The DSL fills empty slots. The fallback order is test input, then `[[0]]`. Invalid grids (31 wide, ragged, value 10, negative, float, empty) are skipped.
   - **Submission:** covers all tasks and multi-test tasks. `validate_submission` detects missing or extra tasks and bad grids. The placeholder is written correctly, and the metric scorer is correct.
   - **Decoder:** skips truncated pickles and `.tmp` files, tolerates a missing store, and pools `.r1` retry files. kgmon ranking is as expected.
   - **Seeds:** `stable_seed` gives the same value under `PYTHONHASHSEED=1` and `=2`, while the base's `hash(bk)` differs. Augmentation keys are reproducible.
   - **Worker end-to-end with 4 tasks:**
     - A training crash (fake OOM) is retried and solved.
     - A zero-candidate task is retried with new seeds and solved.
     - A task that fails twice is retried only once and ends with the fallback.
     - Result: score 4/4 on the fake tasks. `attempt_1:model` = 4 and `attempt_2:model` = 4 (distinct), no `.tmp` files are left, and the tokenizer check reports ok.
   - **Failures and deadlines:**
     - After 3 consecutive failures the rank exits and leaves the rest of the queue for the others.
     - No task starts with < 120 s left.
     - A retry is skipped when there is not enough time.
     - `DeadlineCallback` stops training after the deadline.
   - **starter.py as a real subprocess** with 3 ranks: rank 0 hard-crashes (`os._exit`), rank 2 hangs, and rank 1 still processes **every** task. The hard deadline kills the hung rank and the starter exits cleanly. The cheap-first order is also checked.
3. **Driver-cell simulation.** The launch and submit cells were run on CPU against `/tmp/arc2/data`, in eval mode and in rerun mode (240 placeholder tasks, 259 outputs). The DSL ran in the background, and a valid `submission.json` was written with no GPU results at all.
4. **Program search** on `/tmp/arc2/data` (script `/tmp/arc2/work/eval_dsl.py`):

   | Set | Tasks | Outputs | Outputs with a train-consistent program | Correct (top-2) | Wrong (false positive) |
   |---|---|---|---|---|---|
   | training (ARC-AGI-2 train, 1000) | 1000 | 1076 | 81 (76 tasks) | **81** (81 top-1) | **0** |
   | evaluation (ARC-AGI-2 eval) | 120 | 172 | 0 | 0 | 0 |
   | placeholder test (240 training tasks) | 240 | 259 | 20 (18 tasks) | 20 | 0 |

   - The only false positive found during development was an "object densest" rule (task 72ca375d); it was removed.
   - A harsher leave-one-out check trains the programs on train[:-1] and predicts the held-out train pair (script `/tmp/arc2/work/loo_dsl.py`). It fired on 74 training tasks: 67 were right and 7 wrong (90.5% precision with one fewer pair); on eval it fired on 0.
   - Because the DSL only fills slots that would otherwise be empty or `[[0]]`, a wrong DSL answer cannot cost score.

## Expected impact (honest)

| Change | Expected LB effect | Why |
|---|---|---|
| Crash isolation, deadline callback, hard kills, placeholder and guarded assembly | Protects against a failed or 0 submission; ~0 on a clean run | The base could overrun 12 h: a task started just before `end_time` trains for another 200-870 s with no check. One rank raising, for example a CUDA OOM on a large task, killed all 4 ranks through `mp.spawn`. |
| Cheap-first ordering | 0 to about +1.5 points if the 11.8 h budget is binding (reported runtimes are 11.5-12 h), otherwise ~0 | The tasks dropped at the deadline become the most expensive ones, which have the lowest solve rate, instead of random ones. **Unverified:** a full-eval timing run is needed to confirm. |
| Retry of crashed or zero-candidate tasks | Small; it only uses time left after the main queue | This is the yusuketogashi idea, restricted to otherwise-empty outputs. |
| Deterministic seeds | No change in mean; slightly lower rerun variance | GPU kernel non-determinism remains. |
| Distinct attempt_2 | ~0 | The base `score_sum` already groups identical grids, so duplicates never occurred. The only change is that `[[0]]` is replaced by DSL/test-input guesses. |
| DSL fill | ~0 on the hidden ARC-AGI-2 set (0/120 on eval); non-zero only on ARC-1-style tasks | It is zero-risk by construction. |

Overall this is **the base's expected score (30-34 with ±3 seed noise) with much lower tail risk**, plus a possible small gain from scheduling.
It is not a big accuracy jump. The credible accuracy levers from the scouting report (NVARC+ pooling, hyper-parameter sweeps) need GPU runs on the eval set.

## Risks and open questions

- **Nothing here has run on a GPU.** Please do the commit smoke run first and check the log for:
  - `tokenizer check ok (newline id 10)`
  - `[starter] queued 4 tasks (order=cheap_first, gpus=4)`
  - `[Rank 0..3] start!`
  - `Reload score: 3.0`
  - `Score (model only…)` equal to `Score (final…)`, or higher.
- The runner is now `subprocess.Popen([shutil.which("python"), "-u", "starter.py", ...])` instead of `!python ...`, so it uses the same `python` on PATH.
  - Output is streamed through a thread. If streaming misbehaves on Kaggle, the log shows it immediately in the smoke run.
- The spawn-context `Process` replaces `mp.spawn`. It uses the same start method and the same `CUDA_VISIBLE_DEVICES` per rank, and it was tested locally only on CPU.
- New timing constants:
  - 120 s minimum to start a task
  - scoring cutoff at end_time + 60 s
  - starter kill at end_time + 180 s
  - cell kill at end_time + 240 s

  The worst case therefore finishes around 11 h 55 m, versus an unbounded overrun in the base.
- `ARC_ORDER="sorted"` reproduces the base's queue order exactly if the team wants a pure A/B on ordering.
- The newline token: the base (and ours) uses the correct `"Ċ": 10`.
  - The `"ÄŠ"` mojibake in koushikrudra's fork is a dict *key*, and only `ARC_VOCAB.values()` is used, so even that fork decodes correctly.
  - Our runtime check now verifies the ids against the tokenizer anyway.

## Run log

| Date | Version | Kaggle commit run (4-task check) | Submission ref | Public LB |
|---|---|---|---|---|
| 2026-09-27 | v1 | ran cleanly, exit codes [0,0,0,0], model-only = final = 2.5/4 (base 3.0/4; the difference is 36a08778 test 1, found by a single augmented beam in the base run: seed variance) | 56617135 | pending |
| 2026-09-30 | v2 | commit run clean (exit [0,0,0,0], selector score_full_probmul_3, 2.5/4 on the 4-task check) | submitted 16:0x UTC | pending |
| 2026-10-01 | v2 result | - | 56712633 | 25.97 (probmul_3 selector; v1 kgmon 28.61) -> within the ±3 rerun band, no evidence probmul_3 helps; default back to kgmon for future versions |
| 2026-10-01 | v1 resubmit | - | (see API) | pending (variance measurement) |
| 2026-10-01 | v1 resubmit (kgmon) | 28.06 | LB rerun noise ≈ ±0.5 (28.61 vs 28.06); selector v2 (25.97) is a real loss |
| 2026-10-04 | original perfpatch fork (oanaunciuleanu/arc2-perfpatch-ours v1), unchanged | pending | tests whether our v1 hardening cost ~3 pts (public forks of the original avg ~31.8). Smoke 2.5/4. |
| 2026-10-05 | original perfpatch fork result | **28.06** | identical to our v1 rerun (28.06); our hardening did NOT cost points. The 33.89 public score was a lucky draw. ARC2 pipeline is saturated at ~28-29 for us. |
