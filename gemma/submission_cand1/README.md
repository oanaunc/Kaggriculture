# submission_cand1: hsiaosuan "0.13" bundle, byte-exact

Credit: **hsiaosuan**, Kaggle notebook `hsiaosuan/gemma-developer-agent-0-13-submission`
("Gemma Developer Agent: 0.13 Submission", run 2026-10-04). The author reports 0.13 public LB on 2026-10-03.
That run was after the Sep 30 wheelhouse update (adk_submission 0.2.12).

The files are the notebook's embedded base64 zip, sha256
`6b3ebb33f2688515a5493ed299248ca169501d33e283facd0453d4af0a6a92f6` (10 files), extracted unchanged.
This README is ours and is NOT part of the original archive. Leave it out of the zip.

Design: a `SequentialAgent` runs a read-only `explore` stage (T=0.1, max out 4096; tools run_command/read_file/
get_code_neighbors/get_code_subgraph; no search_similar_code; output_key `repair_plan`), then the `swe_agent` coder
(T=0.2, max out 8192; read_file/edit_file/run_command/get_status/submit_patch + 2 instruction-only skills).
Thinking is off (include_thoughts: false). Budgets: 5 min / 60 calls / 100 turns / timeout 60 s.
See gemma/scout_2026-10-04.md.
