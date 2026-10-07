# submission_cand6: cand1 + one test-based verify-and-retry loop

Base: **hsiaosuan**, Kaggle notebook `hsiaosuan/gemma-developer-agent-0-13-submission` ("Gemma Developer Agent:
0.13 Submission"), shipped unchanged as `gemma/submission_cand1`. This README is ours and is NOT part of the zip.

Root `SequentialAgent swe_workflow`:

1. `explore` - unchanged from cand1.
2. `swe_agent` (coder) - unchanged except that it no longer submits: step 4 of `prompts/system.md` and step 4 of
   `skills/verify-before-submit` now end the turn with a one-sentence report. `submit_patch` stays registered,
   because an unregistered tool call crashes the task (F3); if the coder submits anyway, the harness ends the
   session as in cand1 (graceful fallback to cand1 behaviour).
3. `verify_loop` - `LoopAgent`, `max_iterations: 2`, over `verifier` (`include_contents: none`, `output_key:
   verify_report`, tools run_command/read_file/edit_file/get_status/submit_patch). It imports the changed module and
   runs the closest existing test file (`pytest -x -q`, tail 25). Pass -> `submit_patch`. Failure -> a `git stash`
   baseline run decides pre-existing vs caused by the patch; a caused failure gets one source fix and a re-check.
   A pass that submits and replies ends the whole session (swegemma `agent_runner.py` breaks on a final text event
   once `patch_submitted`), which is the loop's escalation; an unsuccessful pass replies `FAIL: ...` and the second
   iteration retries with that note in `{verify_report?}`.
4. `finalize` - submits the current tree when both passes failed, so the harness does not nudge and re-run the
   whole SequentialAgent from `explore`.

Budgets unchanged from cand1 (5 min / 60 calls / 100 turns / 60 s), so the 12 h worst case is the same.
