# Offline eval on the public tasks

This kernel runs the **official `swegemma.Evaluator`** on a fixed, deterministic subset of the 129 public tasks of `gemma-4-developer-agent`, for one submission bundle. It does the same steps as the host notebook `ryanholbrook/getting-started-gemma-4-developer-agent`:

1. Install the wheelhouse.
2. Start vLLM with `gemma-4-31b-it-qat-w4a16-ct` on 4x L4, with TP=4 and a 32k context.
3. Phase 1: the agent produces a patch in the subprocess sandbox.
4. Phase 2: run pytest.

Compaction and context-cache settings are the same as on the host. The per-task caps come from the bundle's own `eval_config.yaml`.

The point of this kernel is to compare configs with less noise than the public LB. The LB has about 58 tasks, a noise of about ±2 tasks, and allows 1 submission per day.

- Kernel: `oanaunciuleanu/gemma-offline-eval` (private, GPU `NvidiaL4`, internet off). Its sources are the competition, `metric/gemma-4-developer-agent-wheelhouse` and the model `google/gemma-4/Other/gemma-4-31b-it-qat-w4a16-ct/2`, with the host's docker image pinned.
- Files:
  - `make_notebook.py` generates `gemma-offline-eval.ipynb` with the bundle **embedded** as base64. Bundles are about 5 KB; `README.md` is excluded, as in the shipped zip.
  - `kernel-metadata.json`.

## Run with another bundle

```bash
cd /home/user/Kaggriculture
python3 gemma/offline_eval/make_notebook.py --bundle gemma/submission_r12base --n-tasks 20 --concurrency 2
kaggle kernels push -p gemma/offline_eval
kaggle kernels status oanaunciuleanu/gemma-offline-eval
kaggle kernels output oanaunciuleanu/gemma-offline-eval -p /tmp/gemma/oe_out   # results.json, results/
```

Options:

- `--n-tasks N` and `--offset K` take tasks `K..K+N` of the fixed ordering, which is `sha256("gemma-dev-v1:" + instance_id)`. The first 20 tasks are the default dev subset.
- `--task-ids a,b,c` gives an explicit list instead.
- `--concurrency C` sets how many tasks run in parallel against one vLLM server, using the Evaluator's official `concurrency`.
- `--bundle-dir /kaggle/input/...` uses an attached dataset instead of the embedded files. You need this for LoRA `.safetensors`; add the dataset to `dataset_sources`.

**Comparing two configs.** Run both bundles on the **same** subset, meaning the same seed and offset. Each push replaces the previous version of the same kernel, and Kaggle runs only one version of a kernel at a time. To run in parallel, copy the folder and change `id` in `kernel-metadata.json` (for example `oanaunciuleanu/gemma-offline-eval-b`). A 40-task comparison can be split into shards with `--offset 0` and `--offset 20` in two kernels. Compare per task with a paired test (McNemar on the discordant tasks), not on raw pass rates.

## Output: `/kaggle/working/results.json`

- `summary`:
  - which bundle ran: bundle name and sha256, the caps and the subset;
  - results: `n_resolved`, `pass_rate`, empty-patch and error counts;
  - wall seconds for each stage (install, vLLM start, eval);
  - `machine_minutes_per_task_*`.
- `tasks[]`: `instance_id`, `resolved`, `duration_seconds`, `tool_calls`, `llm_calls`, `test_exit_code`, `patch_chars`, `error`, the tail of the test output and the other scalar fields of `TaskResult`.

The harness also writes `results/task_results.jsonl`, `summary.json`, `patches/` and `test_outputs/`.

## Cost (measured)

Smoke run on 2026-10-05: version 1 of the kernel, `submission_cand1`, the first 5 tasks of the subset, C=2.

| Stage | Wall time |
|---|---|
| Wheel install | 60 s |
| vLLM start-up (4x L4, TP=4) | 430 s |
| Eval of 5 tasks (phase 1 + phase 2) | 231 s |
| Whole session | 758 s (12.6 min) |

- **Fixed overhead:** about 8.5 min per run.
- **Marginal cost:** about **0.8 machine-minutes per task at C=2**. The mean task wall time is 89 s, so C=1 would cost about 1.5 min per task.
- **Estimates:** a 20-task run is about 25 min and a 40-task run about 40 min of a 4x L4 session at C=2. These are for cand1's caps of 5 min / 60 calls. In this run the call cap ended a task before the time cap did: the longest task, which hit 60 tool calls, took 141 s. Bundles with larger caps, such as R12 with the 60-minute defaults, can take much longer per task.

Smoke result for cand1: **2/5 resolved**.

| Task | Result | Notes |
|---|---|---|
| fastapi_15589 | pass | |
| fastapi_14786 | pass | |
| fastapi_14479 | fail | tests failed |
| rich_3470 | fail | tests failed |
| fastapi_14482 | fail | hit the 60-tool-call cap with an empty patch |

There were no harness errors. The sandbox logs harmless "ensurepip unavailable" warnings.

## Caveats

- **Concurrency differs from the scorer.** The scorer runs tasks one after another. Here N tasks share the server, and the caps are wall-clock (`max_time_minutes`), so each task gets somewhat fewer tokens within its cap when C > 1. Use C=1 when absolute numbers matter; any fixed C is fine for A/B comparisons. Always compare at the same C.
- **The public tasks are not the LB tasks.** About half of the hidden tasks come from private repos. There are 129 public tasks (fastapi 67, rich 48, requests 13, httpx 1), so the default subset is weighted toward fastapi and rich. The models may also have seen these repos.
- **Run-to-run noise exists here too** (T=0.2, timing-dependent caps). Re-running the same bundle is the way to measure it.
- **GPU quota.** A 4x L4 session counts against the weekly GPU quota for its wall time. Install plus vLLM start-up is a fixed cost per run, so larger N per run is cheaper per task.
- The phase-2 pytest timeout is the bundle's `timeout_seconds`, as in the host notebook. The real scorer's phase-2 settings are not public.
