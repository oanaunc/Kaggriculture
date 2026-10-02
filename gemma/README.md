# Gemma 4 Developer Agent: submission v1 ("harness-debt fixes")

> **Status:** v1 scored **0.06** on the public LB, below the 0.12 public reference it was built to improve on. **v2** (`submission_v2/`) goes back to that 0.12 bundle. It is described in [Submission v2](#submission-v2-reproduce-the-012-bundle-plus-three-small-deltas) at the end of this file. The v1 text below is kept unchanged as a record.

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

## Leaderboard log

| Date | Version | Ref | Public LB | Notes |
|---|---|---|---|---|
| 2026-09-27 | v1 | 56606876 | 0.06 | Below the 0.12 public base. Likely causes: 4.5 min / 45-call cap and 4096 output tokens cutting tasks short, plus heavier prompt rules (note: the 0.12 base also runs with thinking off). v2 rebuilds on the 0.12 config with minimal deltas. |

---

# Submission v2: reproduce the 0.12 bundle, plus three small deltas

v1 changed about 15 things at once and scored **0.06**. romanrozen's "GEMMA: EDA, Baseline for a start" bundle (R12) scored **0.12**. v2 goes back to R12 as a byte-exact base. On top of it, v2 adds only three deltas. Each delta targets a failure mode confirmed in the harness source (F1-F4, F9 above) and costs a few prompt lines or one config file.

## Build (no submission was made)

```bash
/tmp/gemma/venv/bin/python gemma/build.py submission_v2   # -> gemma/dist/submission_v2.zip + dist/submission_v2.manifest.json
/tmp/gemma/venv/bin/python gemma/build.py                 # v1, unchanged -> dist/submission.zip (same sha256 as before)
```

The zip contains `agent.yaml`, `configs/sampling.yaml`, `eval_config.yaml`, `prompts/analyzer.md`, `prompts/system.md` and `sub_agents/code_analyzer.yaml`. It is 6 files and 3,854 bytes, with sha256 `e156523b0f57bfe2052190d247b340eeee2c9a67e5597cfac68060ba86f9db60`.

## How the base was reproduced

R12's `.txt` code dump truncates cells, so the base comes from the full `.ipynb` instead. `/tmp/gemma/r12_repro/run_r12.py` runs R12's own cells locally with `GEMMA_AGENT_DATA=/tmp/gemma/data`. It runs cells 3, 5, 7, 10, 12, 38, 41 and 43: CFG, setup, data discovery, README facts, tool verification, the prompt builders, the sample-submission probe and the bundle writer. The plotting and IPython modules are stubbed out.

The run detected all 9 tools, `ARCH=analyzer+coder`, `!include` style and model `gemma-4-31b-it-qat-w4a16-ct`, which match what the notebook would detect on Kaggle. The generated bundle is kept untouched in `/tmp/gemma/r12_base/`. It has 5 files and no `eval_config.yaml`.

R12 as built:

- **Coder** `swe_coder`. Tools: `run_command, read_file, edit_file, write_file, get_status, submit_patch` and `agent_tool: sub_agents/code_analyzer.yaml` (`skip_summarization: true`). It has **no** graph tools.
- **Analyzer** `code_analyzer`. Tools: `run_command, read_file, search_similar_code, get_code_neighbors, get_code_subgraph`. It answers in a fixed format (at most 250 words): LOCATION / ROOT CAUSE / FIX PLAN / RELATED / TESTS / CONFIDENCE.
- **Shared `configs/sampling.yaml`.** Values: `temperature 0.2, top_p 0.95, top_k 40, max_output_tokens 8192, thinking_config: {thinking_budget: 4096, include_thoughts: false}`.
- **Thinking is effectively OFF.** The config is easy to misread as "thinking on with a 4k budget", but it is not. `adk_submission/resolvers/generation.py::apply_thinking_config_to_model` maps `include_thoughts: false` to `chat_template_kwargs.enable_thinking=False`. `thinking_budget` is never forwarded to vLLM (F7). Compiling the v2 tree confirms this for both agents: `extra_body = {'chat_template_kwargs': {'enable_thinking': False}}`. v2 keeps the file byte-identical, so whatever R12 actually ran, v2 runs too.
- **No `eval_config.yaml`.** The harness defaults therefore apply: 60 min, 100 tool calls, 500 turns and a 300 s command timeout.

## Exact diff vs the 0.12 base (`diff -ru /tmp/gemma/r12_base gemma/submission_v2`)

`agent.yaml`, `sub_agents/code_analyzer.yaml` and `configs/sampling.yaml` are **byte-identical**. That covers the tool lists, the sub-agent wiring, the sampling and the thinking settings. The changes are:

```diff
--- r12_base/prompts/system.md
+++ submission_v2/prompts/system.md
@@ Hard rules
 - Always finish by calling `submit_patch`. A careful best-effort fix beats no patch.
+- Never call `search_similar_code`, `get_code_neighbors` or `get_code_subgraph`, even if the task message lists them: they are not in your tool list, and calling them crashes the session and loses your patch. Search with `grep` via `run_command`, or ask `code_analyzer`.
@@ Budget discipline
 - Keep outputs short: pipe through `head`, use `grep -n`, `pytest -q`. Never print whole large files.
+- `run_command` returns only the FIRST 5000 characters of output, but pytest prints its verdict at the END. For test runs and other long output, write to a file and read the tail: `python -m pytest <tests/path> -x -q > /tmp/test.log 2>&1; tail -n 30 /tmp/test.log`.

--- r12_base/prompts/analyzer.md
+++ submission_v2/prompts/analyzer.md
@@ Tools
-- `search_similar_code` for concepts the issue describes without naming code
+- Never call `search_similar_code`: its output is unbounded, overflows your context and crashes the session

+++ submission_v2/eval_config.yaml   (new file)
+evaluation:
+  timeout_seconds: 300      # = default
+  max_time_minutes: 8       # default 60
+  max_tool_calls: 100       # = default
+  max_turns: 500            # = default
```

Why each delta exists:

- **(a) Tail idiom, coder only.** `run_command` keeps the *first* 5,000 characters (F1). Plain `pytest -q` on `httpx_3672` printed 7,951 characters, and the pass/fail line fell in the hidden part. R12's own advice to "pipe through `head`" hides the verdict as well. The analyzer never runs tests, so its prompt is unchanged here.
- **(b) The `search_similar_code` ban.**
  - **Analyzer.** R12 *recommends* this tool to the analyzer. Its output is uncapped: 136k to 259k characters were measured (F4). The analyzer also has no compaction (F6), so a single call overflows 32k and the exception loses the patch (F2).
  - **Keeping it registered.** The tool stays in the analyzer's tool list, so the tool schema is identical to R12's. A call to an unregistered name raises `ValueError` and crashes the session in the same way (F3). Removing it would therefore buy nothing and would change the tool surface.
  - **Coder.** The coder never had any graph tool. The harness's own task message still advertises all three whenever graph files exist (`agent_runner.py` "Code Intelligence Tools"), so a coder call is an unknown-tool crash (F3). The single coder line names all three tools for that reason.
- **(c) An 8-minute cap.** The reasoning is in the next section.

## Why an 8-minute cap (and not none, and not 4.5)

- **Evidence.** R12 finished inside 12 h with the 60 min / 100 calls / 500 turns defaults and thinking off. A second defaults run (zhukovoleksiy v2, 0.10) also finished. Thinking HIGH with 16k output and a 12-minute cap exceeded 12 h. That failure came from the thinking and sampling setup, not from the cap. 4.5 min / 40 calls scored 0.10 (N). 5 min with the starter prompt scored 0.00, which is a prompt effect. So with thinking off, R12-style trajectories are on average much shorter than the 60-minute cap. The real bound on each task was the 100 tool calls, which include the analyzer's calls (F10).
- **What v2 changes about runtime.** Delta (b) removes a crash path. A task that used to die quickly on a `search_similar_code` overflow now keeps running, possibly up to the 100-call cap. This is the only way v2 can run longer than R12. Its size is unknown because R12's crash rate is not public. Exceeding 12 h scores nothing for the whole submission.
- **Decision: `max_time_minutes: 8`, everything else at the defaults.** Container setup is excluded from the agent budget (HARNESS_README §7.1). A task that hits the cap is still graded on its working tree (F9), so the cap costs only fixes that would have landed after minute 8. Those are long, heavily compacted trajectories that rarely resolve. N at 4.5 min still scored 0.10, so even a much tighter cap does not wreck the score. 8 min bounds the extra tail from (b) while leaving R12's typical tasks untouched. 4.5 min (v1) would change R12 behaviour on more tasks and confound the comparison. `max_tool_calls` stays at 100 and `timeout_seconds` at 300, the defaults R12 ran with. `timeout_seconds` may also bound Phase-2 pytest (F8).
- **The 8-minute cap does not guarantee a finish.** The worst case is 125 × (8 + 0.5) + 25 = 1,088 min, which is more than 720. The defaults' worst case is far larger still, and R12 finished anyway, so the realised mean is what matters. That mean is ≤ R12's mean plus the time recovered from avoided crashes, and each such task is capped at 8 min. For this reason, `build.py` treats the worst-case arithmetic as advisory for `submission_v2` (`WORST_CASE_ADVISORY`) and records it in the manifest; for v1 it remains a hard failure.

## v2 vs v1

| | v1 (0.06) | v2 |
|---|---|---|
| Base | Our own rewrite (~1.2k-token prompt) | R12 as generated, byte-exact except the diff above |
| Coder tools | + `search_similar_code`, `get_code_neighbors`, `get_code_subgraph` (registered, banned by prompt) | R12's: no graph tools; prompt forbids calling them |
| Analyzer | Bounded: ≤6 calls, 3 tools, its own sampling (T=0.1, 2,048 output tokens), `skip_summarization: false`, called optionally | R12's: unbounded, 5 tools, shared sampling, `skip_summarization: true`, always called first |
| Sampling | Coder 4,096 output tokens, thinking off | 8,192 output tokens, `thinking_budget 4096` + `include_thoughts: false` (= off) |
| Prompt extras | `{problem_description?}` injection, TASK/NOTE/PLAN rules, strict submit/get_status rules, exact tool whitelist | None (only deltas a/b) |
| Budgets | 4.5 min / 45 calls / 80 turns / 300 s | 8 min / 100 calls / 500 turns / 300 s |

The v1 extras are not carried over. v1 bundled all of them and scored half of R12. No ablation shows that any one of them helps, and several of them (4.5 min / 45 calls, 80 turns, halved output tokens, extra per-call rules) shrink the budget R12 used. If they get re-tested, it should be one at a time on top of v2.

## Validation performed

- **Local checks** with `build.py submission_v2`. All local checks pass. The one advisory is the worst-case note explained above.
- **Official harness checks.** `validate_directory(build_submission_limits())` passes. `validate_single_declared_model` returns the competition model. `compile_submission` gives `swe_coder` → `AgentTool(code_analyzer)` with the tool lists above. ADK `inject_session_state` is a no-op on both instructions: there are no placeholders, and the issue comes from the harness user message, as in R12. The deterministic zip is round-tripped and re-validated.
- **v1 is unaffected.** `build.py` with no argument still produces the byte-identical `dist/submission.zip` (sha256 `db1a3a2d…`).
- **CPU dry run.** `tools/dry_run_mock.py` now supports trees whose analyzer prompt starts with "You are `code_analyzer`, a read-only code" and trees without an `eval_config.yaml`. It ran through the real `swegemma.Evaluator` with `--submission gemma/submission_v2` on `httpx_3672`:
  - The coder was offered its 7 tools and the analyzer its 5.
  - The `code_analyzer` round trip worked.
  - `get_status` reported 8.0 min / 100 calls / 500 turns.
  - The redirect-and-tail pytest command showed the verdict.
  - `submit_patch` produced 9,719 bytes across 7 files, and Phase 2 reported `resolved: True`.
  - The trace is in `/tmp/gemma/dryrun_v2/mock_trace.json`.
  - This tests the plumbing only; the mock applies the gold patch.

## Risks

- **Instructions are not enforcement.** The bans in (b) are prompt lines. An analyzer that ignores them can still overflow, exactly as in R12. The analyzer also keeps R12's other unbounded paths: large `read_file` or grep output, and no compaction (F6).
- **Noise.** About 60 public-LB tasks means about 0.017 per task. A v2 result anywhere in roughly 0.09-0.15 is consistent with "same as R12". Neither a small gain nor a small loss can be attributed to the deltas from one run.
- **Runtime.** See the cap reasoning above. If avoided crashes turn into many 8-minute tasks, total time grows. The hard ceiling is 125 × 8 min of agent time plus setup, and the realised mean is expected to be far lower.
- **Unknown differences from R12's actual run.** The base was regenerated from R12's code on our copy of the data. If Kaggle's copy of the files let R12 detect a different tool set, its uploaded bundle could differ. That is unlikely: all 9 tools appear in `HARNESS_README.md`, which the notebook greps.
| 2026-09-28 | v2 | 56623486 | 0.08 | Submitted 00:16 UTC. NOTE: upload must be named exactly `submission.zip` (submission_v2.zip returned HTTP 400). |
| 2026-09-29 | r12base | 56656630 | 0.12 | Exact public base reproduces 0.12, so v2's deltas (0.08) hurt. | Exact reproduction of the public 0.12 bundle (gemma/submission_r12base) to calibrate LB noise vs our deltas. |
| 2026-09-30 | v3 | 56691693 | 0.08 | Cap is not the cause: the prompt lines cost ~0.04 (likely the analyzer search_similar_code ban, which the 0.12 base relies on). |
| 2026-10-01 | v4 | 56732279 | ERROR | 'notebook hit an unhandled error while rerunning'; only a 1-line prompt diff vs the 0.12 base, host is changing the harness. Resubmit to check. | 0.12 base + only the coder tail-output line (analyzer untouched, no bans). |
- 2026-10-02: resubmitted v4 unchanged to see whether the "unhandled error while rerunning" repeats.
