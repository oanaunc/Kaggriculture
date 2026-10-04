# submission_cand2: hsiaosuan "0.13" bundle + thinking ON

Credit: the base is **hsiaosuan**'s bundle (`hsiaosuan/gemma-developer-agent-0-13-submission`, reported 0.13 public LB on
2026-10-03). See submission_cand1/README.md. This README is ours. Leave it out of the zip.

The only change from cand1 is `include_thoughts: false -> true` in `configs/sampling.yaml` (coder, thinking_budget 2048)
and in `configs/explore_sampling.yaml` (explorer, thinking_budget 1024).

Why: the scorer's wheelhouse was updated on Sep 30 (adk_submission 0.2.12). Two things changed:
- reasoning_content is now mirrored to `reasoning`, so thoughts persist between tool calls (discussion 744354).
- `thinking_budget` is forwarded as `extra_body.thinking_token_budget`, but only when include_thoughts is not false
  (discussion 745059; source `adk_submission/resolvers/generation.py`).
Before the update, thinking-on runs were crippled (thoughts dropped every step) and HIGH/16k ran past 12 h. Every top
public bundle still runs with thinking off. The 0.17-0.24 scores appeared after the fix (Oct 3-4), but their method is
not public. This tree is an untested hypothesis with no LB evidence. Runtime is bounded by the 5-min per-task cap.
