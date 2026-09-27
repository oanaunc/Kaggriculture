# Gemma 4 Developer Agent: submission v1 ("harness-debt fixes")

This is our first leaderboard submission for `gemma-4-developer-agent`. It keeps the architecture of the best public configuration: a coder agent plus a read-only `code_analyzer` sub-agent. This is the design behind romanrozen's 0.12 and zhukovoleksiy's v2 at 0.10.

What changes is the interface. It fixes failure modes that we confirmed by reading the released harness source (`swegemma 0.2.7`, `adk_submission 0.2.11`, `google-adk 1.36.1`). Several of these failure modes make a task score 0 even when the agent has already written a correct fix. No LoRA is included.

## Submit

```bash
python3 gemma/build.py        # or /tmp/gemma/venv/bin/python gemma/build.py to add the official harness checks
kaggle competitions submit gemma-4-developer-agent \
  -f /home/user/Kaggriculture/gemma/dist/submission.zip \
  -m "v1 harness-debt fixes: tail-safe outputs, search_similar_code banned, bounded analyzer, 4.5 min/45 calls, out 4096, thinking off"
```

**Why a direct upload is the right route.** The Kaggle API reports this competition with `is_kernels_submissions_only = False`. For comparison, AIMO-3, the Konwinski Prize and ARC-2025 report `True`. The overview says "You must provide a zip archive (`submission.zip`)…". The host's `swegemma/submission.py` extracts the uploaded archive and looks for `**/agent.yaml` in `/kaggle/working` and `/kaggle/tmp`, choosing the shallowest match. Public notebooks only *build* the zip on CPU; the host's scoring run then executes it. So no notebook or `kernel-metadata.json` is needed.

The limit is 1 submission per day. `kaggle competitions submissions gemma-4-developer-agent` shows the status.

## Layout

```
gemma/
├── submission/                   # the source tree; its contents become the zip root
│   ├── agent.yaml                # root coder (swe_coder)
│   ├── eval_config.yaml          # per-task budgets
│   ├── configs/sampling.yaml     # coder sampling
│   ├── configs/analyzer_sampling.yaml
│   ├── prompts/system.md         # coder instruction (~1.2k tokens)
│   ├── prompts/analyzer.md       # analyzer instruction (~0.45k tokens)
│   └── sub_agents/code_analyzer.yaml
├── build.py                      # validation + deterministic zip + round-trip re-validation
├── dist/submission.zip           # built artifact (7 files, ~5 KB)
├── dist/manifest.json            # sha256, file list, compiled-agent report, budget arithmetic
└── tools/dry_run_mock.py         # CPU end-to-end dry run through the real Evaluator with a scripted mock LLM
```

## Harness facts these changes rely on (verified in source)

| # | Fact | Where |
|---|---|---|
| F1 | `run_command` keeps the **first** 5,000 characters of stdout and of stderr (`result.stdout[:max_chars]`). On a non-zero exit it returns `error_message` (stderr, or stdout if stderr is empty) **plus** `details.stdout` and `details.stderr`. The same text therefore appears 2-3 times, up to about 15k characters (about 4-5k tokens) per call. | `swegemma/tools/execution.py` |
| F2 | Any exception inside the agent loop is caught as `Sandbox execution error` **before** `agent_patch` is assigned. The patch is lost even if `submit_patch` was already called, and even if the working tree holds a fix. Such exceptions include vLLM rejecting a prompt longer than 32,768 − `max_output_tokens`, and a call to an unknown tool name. | `swegemma/harness/agent_runner.py` (the `except Exception` at the end of `run_agent_sandbox`) |
| F3 | Calling a tool name that is not registered raises `ValueError` in ADK. No harness plugin implements `on_tool_error_callback`, so the error propagates to F2. The harness's own task prompt advertises `search_similar_code`, `get_code_neighbors` and `get_code_subgraph` whenever graph files exist. | `google/adk/flows/llm_flows/functions.py::_get_tool`, `adk_eval_core/plugins/` |
| F4 | `search_similar_code` returns each hit's full node source with no cap. Public measurements are 136k characters for `FastAPI` and 259k for `get_openapi`, which is more than the whole context. | `swegemma/tools/graph.py`; zhukovoleksiy walkthrough |
| F5 | Token-threshold compaction runs before each model call: at 14,336 prompt tokens, everything except the last 5 events is summarized. The summarizer keeps **only `part.text`**, so function calls and tool outputs vanish. | `google/adk/flows/llm_flows/compaction.py`, `google/adk/apps/llm_event_summarizer.py`, HARNESS_README §7.2 |
| F6 | An `agent_tool` sub-agent runs in its own `Runner` with no `App`. It therefore gets **no compaction and no LLM-call cap**. Its context grows until it overflows, which triggers F2. Session state, including `problem_description`, *is* copied into its session. | `google/adk/tools/agent_tool.py::run_async` |
| F7 | `include_thoughts: false` sets `enable_thinking=False`. `thinking_budget` and `seed` are never forwarded to vLLM, so thinking that is left on is bounded only by `max_output_tokens`. | `adk_submission/resolvers/generation.py::apply_thinking_config_to_model` |
| F8 | `eval_config.timeout_seconds` becomes `harness.command_timeout_seconds`, which the local `Evaluator` also uses as the **Phase-2 verification pytest timeout**. | `swegemma/config.py`, `swegemma/harness/verification.py` |
| F9 | Time, tool-call and turn exhaustion are caught, and the working-tree diff is then graded. Budget caps end a task but do not lose its patch. Only *uncaught* errors lose the patch (F2). | `agent_runner.py` fallback block |
| F10 | Tool calls and LLM calls made by the analyzer count toward the same per-task `max_tool_calls` and `max_turns`. Measured in the dry run: 10 LLM calls = 7 coder + 3 analyzer. | dry run below |
| F11 | Skill scripts (`run_skill_script`) have **untruncated** output and do **not** count toward `max_tool_calls`. | `google/adk/tools/skill_toolset.py`, `adk_eval_core/sandbox/base.py::AdkSandboxCodeExecutor` |

## Changes and rationale

The baselines are the sample submission (S) and the top public configurations: romanrozen at 0.12 (R12), zhukovoleksiy v2/v3 at 0.10 (Z), nihilisticneuralnet at 0.10 (N) and Black Cat (B).

| # | Change | S / top public | Ours | Rationale → expected effect |
|---|---|---|---|---|
| 1 | Tail-preserving command idiom | S: none. R12: none. Z v3/B: `> log; tail` | Every long command is `CMD > /tmp/out_N.log 2>&1; echo "exit=$?"; tail -n 30 …`. Searches are piped through `head -n 30`. | F1. pytest prints its verdict last. In a local test on `httpx_3672` with the test patch applied, plain `pytest -q` printed 7,951 characters; the first 5,000 end mid-traceback, and `5 failed, 19 passed` sits in the hidden part. The idiom also exits 0, so the harness returns the output **once** and not two or three times, which roughly halves the context each test run costs. → The agent sees pass/fail and loops less. |
| 2 | Ban `search_similar_code`, but keep it registered | S: recommends it first. R12: gives it to the analyzer. Z v3: same as ours. | Kept in `tools` (registered) for the coder; the prompt forbids it. It is removed from the analyzer, whose prompt never mentions it. | F3 + F4. Unregistering it would turn the harness-advertised call into a crash; calling it can overflow the context. Both paths end in F2. → No lost patches from this cause. |
| 3 | Exact tool list in both prompts ("any other name crashes the session") | none | Explicit whitelist; "use `run_command` for grep/ls/sed". | F3: a single hallucinated tool name loses the task. |
| 4 | Bounded analyzer | S: unbounded, uses all graph tools. R12/Z/B: unbounded prompt budget. | The analyzer is limited to ≤6 tool calls, `read_file` ≤60 lines, outputs cut with `head`, `max_output_tokens: 2048`, T=0.1. Tools are `run_command`, `read_file` and `get_code_neighbors` (names only). It answers in a fixed ≤200-word LOCATION / ROOT CAUSE / FIX PLAN / RELATED / TESTS / CONFIDENCE format. | F6. The analyzer has no compaction, so ~6 calls of at most ~4k tokens each plus ~3k of instruction stays under 32k − 2k. → Removes a silent overflow path and bounds how much time the analyzer takes. |
| 5 | Analyzer call is optional and made once | R12: always call it first | The coder reads the code directly when the issue names the file and function; otherwise it calls the analyzer once. `skip_summarization: false` (as in Z v3), so the coder keeps going in the same turn with no forced turn end or nudge. | Saves about 30-60 s on easy tasks under a 4.5-minute cap. |
| 6 | Notes that survive compaction | S/R12: none. Z v3: "write findings". | A `TASK:` block comes first. A `NOTE:` line goes before every tool call (file:line, fact, next step). A running `PLAN:` line is kept. | F5: only the model's text survives summarization. → Fewer re-reads after compaction and less drift. |
| 7 | The issue is repeated in both instructions via `{problem_description?}` | S: absent. Z v3: `{problem_description}` | The optional `?` form cannot raise `KeyError`. The issue is short (median 418 characters, p90 1.9k, max 10k), so this costs about 0.1-0.8k tokens per call. | F5 summarizes the original user message. The analyzer never sees that message and gets only the one-line request, so the issue must reach it through its instruction. Injection into both agents was verified in the dry run. |
| 8 | `max_output_tokens: 4096` and thinking off (`include_thoughts: false`) | S: 16,384 with thinking on. R12: 8,192, off. Z v3/B: 4,096, off. | 4,096 coder, 2,048 analyzer | F2 + F7. vLLM rejects prompt + max_tokens > 32,768, so dropping from 8,192 to 4,096 adds 4k tokens of prompt headroom. With thinking off, outputs are short tool calls, so 4,096 is enough for incremental edits. Every scoring public notebook runs with thinking off, and thinking-HIGH/16k (B "Second Strike") exceeded the 12 h limit. |
| 9 | Sampling | S: T=0.2, top_p=0.95 | Coder T=0.2, top_p=0.95, top_k=40 (the R12 values); analyzer T=0.1 | Kept from the best public run. `top_k` may be dropped by LiteLLM, which is harmless. |
| 10 | Budgets (`eval_config.yaml`) | S: 1 min / 10 calls / 50 turns / 60 s. R12: defaults (60 min / 100 / 500 / 300 s). N: 4.5 min / 40. B: 5 min / 40 / 100 / 120 s. Z v3: 4 min / 80 turns / 120 s. | `max_time_minutes: 4.5`, `max_tool_calls: 45`, `max_turns: 80`, `timeout_seconds: 300` | See "Budget arithmetic" below. The scores that settings flipped: 5 min with the sample prompt gave 0.00; defaults gave 0.12 and 0.10; a 12-minute cap exceeded 12 h (Z v3); thinking HIGH exceeded 12 h (B); 4.5 min / 40 gave 0.10 (N); 4 min / 24 gave 0.08 (B). 4.5 min is the largest cap whose *worst case* still fits. `timeout_seconds` stays at 300 because of F8: a lower value could also cut off slow hidden test files in Phase 2. The prompt bounds agent commands with `timeout 60/120` instead. |
| 11 | Time and loop discipline | S: "under 8-10 turns". R12: `get_status` every 8 calls. | `get_status` (free) is called after the analyzer and after every edit. When fewer than 90 s or fewer than 8 calls remain, the agent stops exploring, edits, runs `py_compile` and submits. Identical calls are never repeated. Two calls in a row with no new information force a change of approach. After two `edit_file` failures, the agent rereads and uses a smaller snippet. | F9: at timeout the tree is graded as it is. So the agent should never be mid-way through a multi-part edit or leave a syntax error; an early plausible edit is worth more than late exploration. |
| 12 | Submit semantics | S: "call submit_patch immediately" | `submit_patch` is the last tool call. If anything is edited after it, the agent resubmits. `git status` is checked and stray `/workspace` files are removed first. | `agent_patch = submitted_patch` when a patch was submitted, so edits made after the last submit are lost. Early "safety" submits cannot protect against F2 either, because the patch is dropped on an exception anyway. So we deliberately do **not** submit early. |
| 13 | Scratch files in `/tmp` via heredoc; a correct sandbox-facts list | S: none. R12: says `/tmp` but does not say how | `cat > /tmp/repro.py << 'EOF'`. No `rg` and no `tree`; use `git grep`. The file tools only reach `/workspace`. | The file tools reject `/tmp`, and a repro script left in `/workspace` ends up in the patch. |
| 14 | No-op LoRA adapters removed | S: `main_lora` / `tool_lora` (all `lora_B` are zero) | none | They make no difference to the model and add a failure point. The unmodified sample *errored* on the LB. |
| 15 | Graph-tool guidance | S: prefer graph tools | `get_code_neighbors` / `get_code_subgraph` are optional. They see synchronous `calls` edges only (no async functions), `edge_type` must never be passed, and the agent stops using them after an error. | The graphs contain only lowercase `calls` edges and no async nodes (walkthrough). |

**Deliberately not changed in v1.** These are next experiments, run one at a time:

- **Skills.** A `run_tests` skill script could print a capped summary for free (F11). The same property is a risk: uncapped output, and a code path no LB submission has exercised yet. The shell idiom (change 1) gets most of the benefit through the path we know works.
- **Single agent versus coder+analyzer.** This is the next ablation. F6 makes the analyzer the only context with no compaction.
- **A 5.0-minute cap.** Only if the runtime margin allows it; see below.

## Budget arithmetic (12 h includes vLLM start-up and sandbox setup, excludes Phase-2 validation)

Worst case: 125 tasks × (4.5 min agent + 0.5 min setup) + 25 min start-up = **650 min**, against a 720-minute limit, which leaves 70 minutes of margin.

The setup estimate comes from Z's full run with 5-minute caps, which finished in about 80 minutes. That is about 40 s per task including agent time, so setup is well under 0.5 min. `build.py` fails the build if the worst case leaves less than 30 minutes.

For comparison, a 5.0-minute cap gives a 713-minute worst case with these assumptions, which is too tight. The 12-minute cap is what failed in Z v3.

## Validation performed

1. **`build.py` local checks.** These cover: exactly one root `agent.yaml`; no symlinks or hidden files; only allowed extensions; under 3 GiB; every YAML parses; every `!include` and `config_path` resolves inside the bundle; every file is referenced; a single model equal to `gemma-4-31b-it-qat-w4a16-ct`; tool names within the 9 the harness registers; the root has `submit_patch`; the sub-agent is read-only; placeholders limited to `problem_description` (ADK's own regex); no `rg`; generation fields allowed, `max_output_tokens ≤ 8192`, `thinking_level` absent and `include_thoughts: false`; `eval_config` keys and types; the 12 h arithmetic. All pass.
2. **Official harness checks** (`/tmp/gemma/venv`, Python 3.12, harness wheels from `metric/gemma-4-developer-agent-wheelhouse`). `validate_directory(build_submission_limits())` passes. `validate_single_declared_model` returns the competition model. `compile_submission` with stand-in tools of the real signatures compiles `swe_coder` → `AgentTool(code_analyzer)`. ADK `inject_session_state` renders both instructions with a problem statement that contains braces. The allowed-extension set equals `swegemma.config.ALLOWED_SUBMISSION_EXTENSIONS`.
3. **Deterministic zip plus round trip.** The archive holds files only (7 entries, `agent.yaml` at the root), with a fixed timestamp, so the same sources always give the same SHA-256. The archive is extracted and every check is re-run. The SHA-256 is recorded in `dist/manifest.json`.
4. **CPU end-to-end dry run** (`tools/dry_run_mock.py`) through the **real `swegemma.Evaluator`**, using the subprocess sandbox, the scoring run's compaction settings and our `eval_config`, on `httpx_3672`. A scripted mock LLM plays the coder, the analyzer and the summarizer. Results:
   - The issue text was injected into both agents.
   - The `code_analyzer` round trip worked, and the analyzer received its 3 tools.
   - `get_status` reported the 4.5 min / 45 calls / 80 turns budget.
   - The prescribed pytest idiom returned `status: ok` with `exit=1` and the final `1 failed in 0.11s` line visible, and no duplicated output.
   - `submit_patch` produced 9,719 bytes across 7 files.
   - Phase 2 in a fresh sandbox reported `resolved: True` (24 passed).

   The mock applies the reference patch, so this validates the plumbing, not the agent's skill. Locally the subprocess sandbox inherits the harness venv, whose httpx shadows the repo; the `--clean-site-packages` flag in the script fixes this, and the problem does not exist on Kaggle's Docker scorer.

It was **not** possible to run the real model on CPU. The first real measurement will be the LB run itself, or a GPU notebook run of the host Evaluator on the public tasks.

## Risks

- **Prompt compliance is not enforced.** The analyzer's 6-call cap, the output trimming and the `search_similar_code` ban are all instructions. A disobedient trajectory can still overflow (F2 or F6). The harness gives the submission surface no way to cap output. An int4 31B model at T=0.1-0.2 usually follows explicit numeric limits, but not always.
- **Timing assumptions.** Start-up and per-task setup times on the scorer are inferred, not measured. If the hidden set is much larger than 125 tasks, or setup is slower than 0.5 min, the margin shrinks. `build.py` makes these assumptions explicit.
- **4.5 minutes may cut off solvable long tasks.** Under the 60-minute defaults, R12 scored 0.12. We trade those tail tasks for a guaranteed finish.
- **`timeout_seconds` in Phase 2.** Whether the Kaggle scorer's Phase 2 reads it is not public (its `metric.py` is not published). 300 equals the default, so this change is neutral in any case.
- **Noise.** About 60 public-LB tasks means about 0.017 per task, and v1 changes many things at once. A ±0.03 result is within noise; attribute effects only through ablations on the 129 public tasks.
- **Data freshness.** The hosts re-uploaded the graphs and embeddings on 2026-09-25. The prompt treats graph tools as optional, so broken graph files only cost a wasted call.
