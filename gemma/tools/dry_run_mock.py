#!/usr/bin/env python3
"""CPU dry run of the submission through the REAL harness (swegemma Evaluator, subprocess sandbox),
with a scripted mock LLM in place of Gemma. No GPU, no model.

It checks the plumbing a real run depends on:
  * compile_submission of our bundle, eval_config budgets, instruction state injection
    ({problem_description?} reaches both the coder and the code_analyzer AgentTool session)
  * the code_analyzer AgentTool round trip (its tools, its answer returned to the coder)
  * the shell idioms our prompt prescribes: `... > /tmp/x.log 2>&1; echo "exit=$?"; tail -n 25 /tmp/x.log`
    returns status ok with pytest's final summary line visible (the harness keeps only the FIRST 5,000 chars)
  * submit_patch -> Phase 2 verification in a fresh sandbox -> resolved flag
The mock "solves" the task by applying the reference patch; this validates plumbing, not skill.

Usage (needs the harness wheels in a py3.12 venv and one snapshot):
  /tmp/gemma/venv/bin/python gemma/tools/dry_run_mock.py \
      --tasks /tmp/gemma/data/tasks.jsonl --snapshots /tmp/gemma/data/snapshots \
      --task-id httpx_3672 --submission gemma/submission --out /tmp/gemma/dryrun \
      (or --submission gemma/submission_v2; issue_injected is expected False for v2, whose
       prompts, like the 0.12 base, rely on the harness user message for the issue text) \
      --clean-site-packages /tmp/gemma/sbxenv/lib/python3.12/site-packages
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import AsyncGenerator

import yaml
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from adk_submission import ModelRegistry
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
from swegemma.models import load_tasks

MODEL = "gemma-4-31b-it-qat-w4a16-ct"
LOG: list[dict] = []


def _fn_responses(req: LlmRequest) -> list[dict]:
    out = []
    for c in req.contents or []:
        for p in c.parts or []:
            if p.function_response is not None:
                out.append({"name": p.function_response.name, "response": p.function_response.response})
    return out


def _sys_text(req: LlmRequest) -> str:
    si = req.config.system_instruction if req.config else None
    if si is None:
        return ""
    if isinstance(si, str):
        return si
    return "".join(p.text or "" for p in getattr(si, "parts", []) or [])


def _call(name: str, **args) -> types.Part:
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


class ScriptedGemma(BaseLlm):
    """Plays a fixed trajectory for the coder, the analyzer and the compaction summarizer."""

    model: str = MODEL
    gold_patch: str = ""

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False) -> AsyncGenerator[LlmResponse, None]:
        sys_text = _sys_text(llm_request)
        responses = _fn_responses(llm_request)
        tools = sorted((llm_request.tools_dict or {}).keys())
        if not sys_text and not tools:  # compaction summarizer
            role, parts = "summarizer", [types.Part(text="SUMMARY: coder located the bug and applied a fix.")]
        elif "You are `code_analyzer`, a read-only code" in sys_text:  # v1 "locator", v2 "navigation specialist"
            role = "analyzer"
            step = len(responses)
            parts = [
                [_call("run_command", command="git grep -n 'def complete' -- '*.py' | head -n 30")],
                [_call("read_file", filepath="src/httpx/_parsers.py", start_line=370, end_line=420)],
                [types.Part(text="LOCATION: src/httpx/_parsers.py:378-400 (HTTPParser.complete)\nROOT CAUSE: rename.\n"
                                 "FIX PLAN: rename complete->reset.\nRELATED: src/ahttpx/_parsers.py\nTESTS: tests/test_parsers.py\n"
                                 "CONFIDENCE: high")],
            ][min(step, 2)]
        else:
            role = "coder"
            step = len(responses)
            patch = self.gold_patch if self.gold_patch.endswith("\n") else self.gold_patch + "\n"
            script = [
                [types.Part(text="TASK: rename HTTPParser.complete to reset.\nNOTE: asking analyzer."),
                 _call("code_analyzer", request="Locate HTTPParser.complete and its callers.")],
                [types.Part(text="NOTE: analyzer says src/httpx/_parsers.py:378."), _call("get_status")],
                [types.Part(text="NOTE: apply fix."),
                 _call("run_command", command="cat > /tmp/fix.patch << 'EOF'\n" + patch + "EOF\ngit apply /tmp/fix.patch; echo \"exit=$?\"")],
                [types.Part(text="NOTE: run targeted tests (expected to fail: old tests call complete())."),
                 _call("run_command", command="timeout 120 python3 -m pytest tests/test_parsers.py -q -x -p no:cacheprovider > /tmp/t_1.log 2>&1; echo \"exit=$?\"; tail -n 25 /tmp/t_1.log")],
                [types.Part(text="NOTE: review diff."), _call("run_command", command="git status --short; git diff | head -n 80")],
                [_call("submit_patch")],
                [types.Part(text="Renamed complete() to reset() and closed streams on server exit.")],
            ]
            parts = script[min(step, len(script) - 1)]
        LOG.append({"role": role, "n_function_responses": len(responses), "tools_offered": tools,
                    "system_chars": len(sys_text),
                    "issue_injected": "Server connection handling." in sys_text,
                    "last_response": responses[-1] if responses else None,
                    "emits": [p.text[:80] if p.text else f"call:{p.function_call.name}" for p in parts]})
        yield LlmResponse(content=types.Content(role="model", parts=parts),
                          usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=1000,
                                                                                    candidates_token_count=50,
                                                                                    total_token_count=1050))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--task-id", default="httpx_3672")
    ap.add_argument("--submission", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clean-site-packages", default=None,
                    help="site-packages dir of a venv holding only pytest + build backends; the subprocess "
                         "sandbox otherwise inherits the harness venv, whose httpx/rich/requests/fastapi "
                         "shadow the repo under test (local-only artefact)")
    a = ap.parse_args()
    if a.clean_site_packages:
        from swegemma.sandbox.subprocess import SubprocessManager
        _orig_start = SubprocessManager.start

        def _start(self):
            sid = _orig_start(self)
            for pth in self._sandboxes[sid]["venv"].glob("lib/python*/site-packages/_host_env.pth"):
                ws = self._sandboxes[sid]["workspace"]
                # subprocess mode skips the editable install, so expose the repo source directly
                pth.write_text(f"{a.clean_site_packages}\n{ws / 'src'}\n{ws}\n", encoding="utf-8")
            return sid

        SubprocessManager.start = _start

    sub = Path(a.submission).resolve()
    # harness defaults (scripts/inference.py) apply when a tree ships no eval_config.yaml
    ev = {"timeout_seconds": 300, "max_time_minutes": 60, "max_tool_calls": 100, "max_turns": 500}
    if (sub / "eval_config.yaml").exists():
        ev.update(yaml.safe_load((sub / "eval_config.yaml").read_text())["evaluation"])
    tasks = [t for t in load_tasks(Path(a.tasks)) if t.instance_id == a.task_id]
    assert tasks, a.task_id
    task = tasks[0]
    gold = next(json.loads(l)["patch"] for l in open(a.tasks) if json.loads(l)["instance_id"] == a.task_id)

    models = ModelRegistry()
    models.register(MODEL, ScriptedGemma(gold_patch=gold))
    limits, gen = build_submission_limits()
    cfg = EvalConfig(
        tasks_path=Path(a.tasks), snapshots_dir=Path(a.snapshots), results_dir=Path(a.out), submission_dir=sub,
        models=models, sandbox="subprocess",
        timeout_seconds=int(ev["timeout_seconds"]), max_time_minutes=float(ev["max_time_minutes"]),
        max_tool_calls=int(ev["max_tool_calls"]), max_turns=int(ev["max_turns"]),
        limits=limits, generation_constraints=gen,
        # same values as the scoring run (HARNESS_README section 7.2)
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(compaction_interval=5, overlap_size=2,
                                                        token_threshold=14336, event_retention_size=5),
        graph_dir="/nonexistent", embeddings_dir="/nonexistent", display_mode="quiet", verbose=False,
    )
    result = asyncio.run(Evaluator(cfg).evaluate_task(task=task, task_index=1, total_tasks=1))
    d = result.model_dump() if hasattr(result, "model_dump") else vars(result)
    keep = {k: d.get(k) for k in ("instance_id", "resolved", "error", "agent_error", "patch_size", "exit_code")}
    print(json.dumps(keep, indent=2, default=str))
    for row in LOG:
        lr = row.pop("last_response")
        row["last_response_preview"] = json.dumps(lr, default=str)[:400] if lr else None
        print(json.dumps(row, default=str))
    (Path(a.out) / "mock_trace.json").write_text(json.dumps({"result": d, "llm_calls": LOG}, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
