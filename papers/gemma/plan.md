# Gemma 4 Developer Agent: Paper Track plan

Status on 2026-09-27. Raw material is in `/tmp/gemma/`: `main_pages.txt` and pulled notebooks in `nb_*/`, with text dumps in `nb_*/*.txt`.

## 1. Competition facts (verified)

**Paper track** (`gemma-4-developer-agent-paper`; we are entered; 44 teams)
- The deadline is **Nov 12 2026, 23:59 UTC**. Drafts that have not been submitted are not judged. The limit is 3,000 words, and the work must be original and unpublished. The track is non-archival.
- Five criteria, each scored 0-5 and averaged: Novelty, Quality/generality, Relevance to SWE and agentic learning, Verifiability, Clarity. The top 3 papers win a share of $35k and get a Google event slot (a NeurIPS expo workshop, for example).
- Required sections: title and subtitle, abstract, introduction, methods and experiments, related work and citations. An optional public notebook can be linked, and an optional arXiv-style PDF can be added through the Project Link.
- Topics: PEFT/RL for SWE agents, code comprehension and code graphs, tasks and benchmarks, and graph reasoning. Papers that use the released graph and embedding data are "highly encouraged".
- Entering the main competition is **not required** for the paper.

**Main competition** (`gemma-4-developer-agent`; we are **not entered**; entry deadline Nov 25, final deadline Dec 2)
- Task: SWE-bench-style Python issue resolution. The score is the % of tasks whose hidden tests pass after the agent's `git diff` is applied.
- A submission is a declarative `submission.zip`: an ADK `agent.yaml`, prompts, sub-agents (`agent_tool`), skills (scripts run in the sandbox), optional LoRA `.safetensors` adapters, and `eval_config.yaml`. It must be under 3 GiB and cannot contain arbitrary harness code. Limits are 1 submission per day and 2 final submissions.
- Model: **only `gemma-4-31b-it-qat-w4a16-ct`** (31B, QAT int4 weights), used for every agent and sub-agent. It is served by vLLM on **4x L4** with **max_model_len 32,768**. Different LoRA adapters are allowed per agent.
- Tools: run_command, read_file (at most 150 lines), edit_file, write_file, submit_patch, get_status, get_code_neighbors, search_similar_code, get_code_subgraph.
- Budget: 12 h for about 120 hidden tasks, run sequentially and including sandbox setup, so about 5.5 min per task on average. Per-task caps (`max_time_minutes`, `max_turns`, `max_tool_calls`, `timeout_seconds`) go in `eval_config.yaml`.
- Data: **129 public tasks** from fastapi, rich, requests and httpx. Each task comes with the gold patch, the test_patch, a repo snapshot `.tgz`, an AST call/dependency graph (NetworkX JSON with full node source), 256-d node embeddings, and 124 offline wheels. The hidden test set is about 120 tasks from **private repos**, split 50/50 public/private, and was filtered so that "a frontier model passes or is within one test".
- Harness: the libraries `swegemma 0.2.7`, `adk_submission 0.2.11` and `google-adk 1.36.1` are public in the dataset `metric/gemma-4-developer-agent-wheelhouse`. The host notebook `ryanholbrook/getting-started-gemma-4-developer-agent` runs the real `Evaluator` (phase 1 generates the patch, phase 2 runs pytest) in a Kaggle notebook on an L4 machine. **This means we can reproduce the scorer offline on the 129 public tasks.**
- Leaderboard (Sep 27): the top score is **0.15**, and about 10 teams are at 0.12-0.13. All of them tune prompts and budgets; none has trained a LoRA yet.
- We could not download the competition files (HTTP 403, because we are not entered). To get the data we either accept the main-competition rules (an action for the user) or read it inside a Kaggle notebook with the competition attached.
- We could not reach forum topics: there is no CLI command for them, and the Kaggle MCP is unauthorized.

**Harness pathologies already reported in public notebooks** (zhukovoleksiy walkthrough, Black Cat, Provorov). These are our hook:
1. `run_command` keeps only the **first** 5,000 characters, but pytest prints its verdict last. The agent never sees pass/fail.
2. `search_similar_code` returns full class bodies with no cap: 136k characters for `FastAPI`, 259k for `get_openapi`. This overflows the 32k context, the error is uncaught, and **the task's patch is lost even after submit_patch**.
3. vLLM rejects any prompt longer than 32,768 - max_output_tokens, with the same effect of losing the patch.
4. Event compaction drops tool calls and outputs and keeps only the model's text. Findings that live in tool outputs vanish.
5. `write_file` cannot write to `/tmp`. `rg` is not installed. `skip_summarization: true` on the analyzer ends the coder's turn and triggers a nudge.
6. An 80-minute run with 5-minute caps scored 0.00, and a run with 12-minute caps exceeded 12 h. Budget settings flip the result.

These map almost one-to-one onto our ARC-AGI-3 findings (`arc3/analysis/failure_report.md`; small model with a 32k context):

| ARC3 finding | SWE analogue here |
|---|---|
| Memory-label parser dropped 70% of notes | Compaction keeps only text, so findings in tool outputs are lost |
| Stale "game over" message after auto-reset | Status and nudge messages that no longer match the state (the analyzer turn-end nudge) |
| Image tokens over-counted, so only about 8 calls fit in context | Uncapped tool outputs (graph and pytest) consume or overflow the 32k context |
| Re-sending the full prompt on each yield | Instructions re-sent every turn (`{problem_description}` repetition) |
| 61-64% of time on the last unsolved level | Time spent on unresolvable tasks under a shared 12 h budget |
| Idle analysis loops | Read/grep loops without edits, and repeated identical calls |

## 2. Compute reality

- We have about 30 GPU-h per week of Kaggle quota and nothing else, which is roughly 180 GPU-h over 6 weeks. The competition model needs about 17-20 GB of weights plus KV cache for a 32k context. That fits on 4x L4 (as in the host notebook) and on a single RTX Pro 6000 (96 GB). **To verify in week 1:** which accelerator shapes our account can select, and whether vLLM concurrency lets us run 4-8 tasks in parallel against one server. The Evaluator runs tasks one at a time, so parallelism means one Evaluator per task in a thread or process pool.
- Internet: the competition's scoring run is offline, but our own development notebooks can turn on internet. The pulled notebooks all have it off by choice, and the offline wheelhouse makes internet unnecessary. **To verify** by switching it on in one interactive session.
- Cost model: one full pass over the 129 tasks at an average of 3-4 min per task, sequential, takes about 7-8 GPU-h. With 4 tasks running concurrently it takes about 2-2.5 h (**assumption; measure in week 1**). Plan for about 40 full-pass equivalents at most, and use a fixed 43-task dev subset (1/3) for iteration.
- CPU-only work costs no GPU quota: localization metrics over graphs and embeddings, parsing trajectory logs, failure detectors, and graph-view ablations.

## 3. Candidate directions (ranked)

### Rank 1 (recommended): "Harness debt". A cross-domain failure taxonomy for small local agents, with targeted harness and interface fixes

- **Research question.** How much of a small local model's failure rate on agentic SWE comes from the harness and the agent-computer interface rather than from the model's own capability? Which interface fixes recover it? Do those fixes transfer across domains (ARC-AGI-3 games and SWE repair)?
- **Method.**
  1. Instrument the public Evaluator: log every event, tool call, output length, truncation, compaction, prompt-token count, error and final patch.
  2. Write about 12 **deterministic failure detectors** that run over the trajectories. Examples: verdict truncated; context overflow lost the patch; graph tool overflow; compaction lost a finding (a symbol or line cited in a tool output, then re-read after compaction); edit_file old_string mismatch loop; no-edit read loop (N calls without a write); repeated identical call; timeout with a plausible diff; test-file edits; wrong-file localization (compare against the files in the gold patch); budget exhaustion. Validate the detectors on about 60 hand-labelled trajectories (precision and recall).
  3. Build the same taxonomy for our ARC3 runs (already about 80% done). This is the cross-domain evidence.
  4. Apply **interventions** using only the allowed submission surface, so that they are deployable. Each one targets a detector:
     - I1: output shaping. Skill scripts `run_tests` and `grep_ctx` return the tail plus a summary, with a byte budget.
     - I2: budgeted graph views. A skill wraps the graph tools with signature-only nodes, top-k ranking by personalized PageRank from symbols named in the issue, and a hard 3k-token cap.
     - I3: a compaction-proof scratchpad of structured notes in the model's text (LOCATION, ROOT CAUSE, TRIED).
     - I4: a stuck/loop breaker: prompt rules plus `max_turns`.
     - I5: token-budget accounting: max_output_tokens and thinking off versus a small thinking budget.
     - I6: a time-allocation policy that gives up early on low-progress tasks and submits the best diff.
  5. Optionally, as an upper bound: apply the same fixes inside the harness itself (the wheels are open source). This separates "fixable by users" from "fixable by harness authors".
- **Experiments.** Data: the 129 public tasks, with a dev subset of 43 and 86 for the report, or all 129 with 3 seeds. Configurations: B0 (the host sample submission), B1 (the strongest public prompt, about 0.12 on the LB, rebuilt from the Black Cat and Roman notebooks), B1 plus each single intervention (6 ablations), and B1 plus all interventions. Each configuration runs 3 seeds at temperature 0.2. Metrics:
  - resolve rate, with a 95% bootstrap CI and paired McNemar tests per task;
  - failure-class rates per detector;
  - "lost-patch" rate;
  - tokens and wall-time per task;
  - localization Hit@1 and Hit@5 at the file and function level;
  - the "harness-attributable share" of failures, measured as tasks flipped by interface-only fixes.
  If we enter the main competition, 3-5 LB submissions on the ~60 hidden private-repo tasks serve as held-out generalization.
- **Compute.** About 9 configurations x 3 seeds x 129 tasks is about 27 full passes, or 55-70 GPU-h at 4-way concurrency. Adding the dev iterations and the detector-labelling runs brings the total to about 110 GPU-h, which fits.
- **Expected contribution.** (a) The first quantified harness-debt taxonomy for a 31B int4 local SWE agent. (b) Evidence that the same interface failures appear in a very different agentic domain (ARC-AGI-3), which argues for generality. (c) Open detectors and skills that others can reuse. (d) A practical recipe that is deployable within the competition's rules.
- **Risks.** Resolve rates are low (about 12%, so about 15 tasks out of 129), which makes CIs wide. Mitigations: seeds, paired tests, and reporting failure-class shifts as the primary outcome. Other risks: parallel evaluation may not work (then use a 43-task subset and 2 seeds); contamination on the public repos (the LB private repos give a held-out check); the harness version could change mid-competition (pin the wheel versions).

### Rank 2: budget-aware code-graph views for localization by a 32k-context agent

- **Research question.** The released graph and embedding tools are harmful as shipped, because their output overflows the context. Can a *budgeted* graph interface improve localization and resolution for a small agent, and which view (callers/callees, induced subgraph, embedding top-k, PageRank seeded from the issue) carries the signal?
- **Method.** This is CPU-first. Compute localization Hit@k and MRR on the 129 tasks for BM25/TF-IDF, embeddings, graph PPR, and hybrids, all under token budgets of 1k, 3k and 8k. Then run the best 2 views end to end as skills (3 seeds).
- **Compute.** Offline work needs no GPU. End to end is 4 configurations x 3 seeds, about 25 GPU-h.
- **Contribution and risks.** It fits the track's graph topics and uses the data the organizers want used. The risk is that retrieval gains don't turn into resolves, a known finding in Agentless and LocAgent. Mitigation: report localization and resolution separately. **This fits inside Rank 1 as intervention I2 and a full section.**

### Rank 3: trajectory-distilled LoRA for protocol compliance

- **Research question.** Can a small LoRA trained on filtered successful trajectories reduce protocol failures (malformed edits, loops, test edits) on the frozen int4 base?
- **Method.** Rejection-sample trajectories from the Rank-1 runs and from external SWE-Gym/SWE-smith-style tasks on other repos, then train a LoRA with SFT.
- **Risks (high).** Training on a compressed-tensors W4A16 base with 32k-token sequences on L4 or RTX Pro 6000 is unproven. There are few successes (about 20 per pass). Quota is tight. Put this under Future Work, or run a small pilot only if weeks 1-3 finish early.

## 4. Recommended: Rank 1 with Rank 2 as its graph section

Working title: **"Harness Debt: Most Failures of a Small Local SWE Agent Are Interface Failures"**. Subtitle: *A failure taxonomy, deterministic detectors, and deployable fixes, with Gemma 4 31B on repository repair and ARC-AGI-3.* The title's claim is a hypothesis. It must be re-worded to whatever the data shows.

### Week-by-week plan (Sep 28 to Nov 12)

| Week | Dates | Deliverables |
|---|---|---|
| 1 | Sep 28-Oct 4 | Get the data (user decision: enter the main comp, or attach it in a notebook). Run the host Evaluator end to end on 5 tasks in a Kaggle notebook. Measure seconds per task and test 4-way concurrency. Add full trajectory logging (events, token counts, truncation flags). Freeze the wheel versions. Pick the 43-task dev split, stratified by repo. |
| 2 | Oct 5-11 | Run baselines B0 and B1 on all 129 tasks (1 seed). Write detectors v1 on CPU. Hand-label about 60 trajectories. Run the CPU localization study (Rank 2): Hit@k under budgets. Port the ARC3 taxonomy to the shared schema. |
| 3 | Oct 12-18 | Implement interventions I1-I6 as skills, prompts and eval_config. Run single-intervention ablations on the dev split (1 seed). Pick the final set. Report detector precision and recall. |
| 4 | Oct 19-25 | Main runs: B1, the 6 single interventions, and all interventions, 3 seeds x 129 tasks (about 60 GPU-h, spread over 2 weeks). Optionally make LB submissions of B1 and B1+all as a held-out check. |
| 5 | Oct 26-Nov 1 | Finish the main runs and the optional harness-side upper-bound run. Analysis: CIs, McNemar tests, failure-class shift tables, harness-attributable share, cross-domain table. Write the first full draft. |
| 6 | Nov 2-8 | Freeze numbers. Build the public reproduction notebook (log parser, detectors, skills bundle). Make figures. Cut the draft to 3,000 words. Get an internal review. |
| buffer | Nov 9-12 | Final edits. **Submit the writeup by Nov 11** (one day of slack). Check that the linked notebook and assets are public. |

Go/no-go checks:
- End of week 1: if concurrency fails, drop to 2 seeds on 86 tasks.
- End of week 3: if no intervention moves resolve rate or failure classes on the dev split, pivot the headline to the taxonomy and detectors plus the Rank-2 localization study.

### Paper outline (target at most 3,000 words)

| # | Section | Words |
|---|---|---|
| - | Title, subtitle, abstract | 180 |
| 1 | Introduction: small local agents; "harness debt"; contributions (taxonomy, detectors, fixes, cross-domain evidence) | 330 |
| 2 | Setting: Gemma 4 31B W4A16, 32k context, ADK harness, tools, 12 h budget, 129 public tasks; ARC-AGI-3 agent setup | 280 |
| 3 | Failure taxonomy and deterministic detectors: definitions, detector rules, precision and recall against hand labels | 450 |
| 4 | Where the failures are: baseline measurement, share attributable to the harness, lost-patch rate, token and time sinks | 400 |
| 5 | Interventions: output shaping, budgeted graph views (localization study), compaction-proof notes, loop breaker, token and time budgeting | 450 |
| 6 | Results: ablation table (resolve rate with CI, paired tests), failure-class shifts, LB held-out check, harness-side upper bound | 450 |
| 7 | Cross-domain: the same failure classes in the ARC-AGI-3 small-model agent, and fixes that transferred | 220 |
| 8 | Related work: SWE-agent ACI, Agentless, OpenHands, SWE-bench(-Lite/Verified), SWE-Gym, SWE-smith, R2E-Gym, LocAgent, RepoGraph, CodeRAG, context compaction | 150 |
| 9 | Limitations, reproducibility (seeds, versions, notebook), conclusion | 90 |
| | **Total** | **3,000** |

Figures and tables (these do not count toward the word limit):
1. Failure-class Sankey or bar chart, baseline versus fixed.
2. Ablation table.
3. Localization Hit@k versus token budget.
4. Cross-domain taxonomy table.

## 5. Open items and user decisions
- Whether to **enter the main competition**. That gives data download, LB held-out checks, and the option to publish the method there. Entering is not needed for the paper.
- Whether to confirm that 4x L4 or RTX Pro 6000 can be selected, and the remaining weekly GPU quota (`get_accelerator_quota` via the Kaggle MCP once it is authorized).
- Data terms: the competition data is under Apache 2.0, but the rules forbid redistributing it to non-participants. Publish detectors, code and aggregate statistics only, not task snapshots.
- Public-notebook release near the deadline (optional in the rules, but it strongly helps Verifiability).
