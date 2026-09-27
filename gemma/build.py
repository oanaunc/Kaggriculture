#!/usr/bin/env python3
"""Validate gemma/submission/ and package it as the exact submission.zip Kaggle expects.

Usage:
    python3 gemma/build.py                      # local checks + zip
    /tmp/gemma/venv/bin/python gemma/build.py   # + official harness checks (adk_submission/swegemma)

Output: gemma/dist/submission.zip (files only, agent.yaml at the archive root, deterministic bytes)
plus gemma/dist/manifest.json (sha256, file list, budget arithmetic).

Checks (local, no harness needed):
  * exactly one root config (agent.yaml) at the root; no symlinks; allowed extensions only; < 3 GiB
  * every YAML parses; every !include and agent_tool config_path resolves inside the bundle
  * exactly one model, and it is the competition model
  * every tool name is one the harness registers; root has submit_patch; sub-agents are read-only
  * instruction placeholders: only {problem_description} / {problem_description?} (ADK raises KeyError otherwise)
  * prompts never tell the agent to run `rg` (not installed in the sandbox)
  * generation config within the harness GenerationConstraints; thinking disabled explicitly
  * eval_config.yaml keys/types, and the worst-case 12 h runtime arithmetic
Checks (official, when the harness wheels are importable):
  * adk_submission.validate_directory with swegemma.config.build_submission_limits()
  * swegemma.models.discovery.validate_single_declared_model
  * adk_submission.compile_submission with stand-in tools of the real signatures
  * ADK inject_session_state on every compiled instruction with a sample problem statement
Round trip: the zip is extracted to a temp dir and all checks are re-run on the extracted copy.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SRC = HERE / "submission"
DIST = HERE / "dist"
ZIP_PATH = DIST / "submission.zip"

MODEL = "gemma-4-31b-it-qat-w4a16-ct"
ROOT_NAMES = ["agent.yaml", "agent.yml", "root_agent.yaml", "root_agent.yml"]
ALLOWED_EXT = {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}
MAX_BYTES = 3 * 1024**3
HARNESS_TOOLS = {
    "run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch",
    "get_code_neighbors", "search_similar_code", "get_code_subgraph",
}
WRITE_TOOLS = {"edit_file", "write_file", "submit_patch"}
ALLOWED_STATE = {"problem_description"}
EVAL_KEYS = {"timeout_seconds": int, "max_tool_calls": int, "max_time_minutes": (int, float), "max_turns": int}

# 12 h budget arithmetic (hidden test set is "about 120" tasks; assume a little more).
N_TASKS_ASSUMED = 125
SETUP_MIN_PER_TASK = 0.5      # container setup per task (80-min full run with 5-min caps bounds it)
STARTUP_MIN = 25              # vLLM start-up on 4x L4 (engine-ready timeout is 20 min)
LIMIT_MIN = 12 * 60


class IncludeLoader(yaml.SafeLoader):
    pass


IncludeLoader.add_constructor("!include", lambda loader, node: {"__include__": loader.construct_scalar(node)})


def fail(msg: str) -> None:
    raise SystemExit(f"BUILD FAILED: {msg}")


def load(path: Path):
    return yaml.load(path.read_text(encoding="utf-8"), Loader=IncludeLoader)


def resolve_include(root: Path, owner: Path, value):
    """Return the included content: str for .md/.txt, parsed YAML for .yaml."""
    if not (isinstance(value, dict) and "__include__" in value):
        return value
    ref = value["__include__"]
    target = (owner.parent / ref).resolve()
    if root.resolve() not in target.parents or not target.is_file():
        fail(f"{owner.relative_to(root)}: !include {ref} does not resolve to a file inside the bundle")
    if target.suffix in (".yaml", ".yml"):
        return load(target)
    return target.read_text(encoding="utf-8")


def check_instruction(owner: str, text: str) -> None:
    for m in re.finditer(r"{+[^{}]*}+", text):  # the regex ADK uses for state injection
        var = m.group().strip("{}").strip().removesuffix("?")
        if var.isidentifier() and var not in ALLOWED_STATE:
            fail(f"{owner}: instruction placeholder {m.group()} is not in session state (ADK KeyError)")
    if re.search(r"(^|[`\s(;|&])rg\s", text):
        fail(f"{owner}: instruction tells the agent to run rg, which is not installed")


def check_generation(owner: str, cfg: dict) -> None:
    if not isinstance(cfg, dict):
        fail(f"{owner}: generate_content_config must be a mapping")
    allowed = {"temperature", "top_p", "top_k", "max_output_tokens", "presence_penalty",
               "frequency_penalty", "stop_sequences", "response_mime_type", "seed", "thinking_config"}
    extra = set(cfg) - allowed
    if extra:
        fail(f"{owner}: generation fields not allowed: {sorted(extra)}")
    mot = cfg.get("max_output_tokens", 16384)
    if not 1 <= mot <= 32768:
        fail(f"{owner}: max_output_tokens {mot} out of range")
    if mot > 8192:
        fail(f"{owner}: max_output_tokens {mot} leaves < 24.5k tokens of prompt room (policy: <= 8192)")
    tc = cfg.get("thinking_config") or {}
    if "thinking_level" in tc:
        fail(f"{owner}: thinking_level maps to reasoning_effort on vLLM; do not set it")
    if tc.get("include_thoughts") is not False:
        fail(f"{owner}: thinking must be disabled explicitly (include_thoughts: false)")
    if cfg.get("temperature", 0.0) < 0.0 or not 0.0 <= cfg.get("top_p", 1.0) <= 1.0 or cfg.get("top_k", 1) < 1:
        fail(f"{owner}: bad sampling values")


def check_agent(root: Path, path: Path, is_root: bool, seen: set, report: list) -> None:
    if path in seen:
        return
    seen.add(path)
    rel = path.relative_to(root).as_posix()
    data = load(path)
    if not isinstance(data, dict):
        fail(f"{rel}: not a mapping")
    if data.get("model") != MODEL:
        fail(f"{rel}: model must be {MODEL}, got {data.get('model')!r}")
    if not str(data.get("name", "")).isidentifier():
        fail(f"{rel}: name must be an identifier")
    if data.get("adapter"):
        ad = root / "adapters" / data["adapter"]
        if not (ad / "adapter_model.safetensors").is_file() or not (ad / "adapter_config.json").is_file():
            fail(f"{rel}: adapter {data['adapter']} not shipped")
    instr = resolve_include(root, path, data.get("instruction", ""))
    if not isinstance(instr, str) or not instr.strip():
        fail(f"{rel}: empty instruction")
    check_instruction(rel, instr)
    gen = resolve_include(root, path, data.get("generate_content_config") or {})
    check_generation(rel, gen)
    names, subs = [], []
    for t in data.get("tools") or []:
        if isinstance(t, str):
            names.append(t)
        elif isinstance(t, dict) and "agent_tool" in t:
            cp = t["agent_tool"]["config_path"]
            # config_path is resolved relative to the including file's directory, then the root
            cand = [(path.parent / cp).resolve(), (root / cp).resolve()]
            target = next((c for c in cand if c.is_file() and root.resolve() in c.parents), None)
            if target is None:
                fail(f"{rel}: agent_tool config_path {cp} not found inside the bundle")
            subs.append(target)
        else:
            fail(f"{rel}: unsupported tools entry {t!r}")
    unknown = set(names) - HARNESS_TOOLS
    if unknown:
        fail(f"{rel}: tools not registered by the harness: {sorted(unknown)}")
    if is_root and "submit_patch" not in names:
        fail(f"{rel}: root agent must have submit_patch")
    if not is_root and set(names) & WRITE_TOOLS:
        fail(f"{rel}: sub-agent must stay read-only, has {sorted(set(names) & WRITE_TOOLS)}")
    if data.get("skills"):
        for s in data["skills"]:
            if not (root / s / "SKILL.md").is_file():
                fail(f"{rel}: skill {s} has no SKILL.md")
    report.append({"agent": data["name"], "file": rel, "tools": names + [f"agent_tool:{s.stem}" for s in subs],
                   "instruction_chars": len(instr), "max_output_tokens": gen.get("max_output_tokens"),
                   "temperature": gen.get("temperature")})
    for s in subs:
        check_agent(root, s, False, seen, report)


def check_eval_config(root: Path) -> dict:
    p = root / "eval_config.yaml"
    if not p.exists():
        return {"eval_config": "absent (harness defaults: 60 min / 100 calls / 500 turns)"}
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    sec = raw.get("evaluation", raw)
    for k, v in sec.items():
        if k not in EVAL_KEYS:
            fail(f"eval_config.yaml: unknown key {k}")
        if not isinstance(v, EVAL_KEYS[k]) or isinstance(v, bool) or v <= 0:
            fail(f"eval_config.yaml: {k}={v!r} has the wrong type/value")
    t = float(sec.get("max_time_minutes", 60))
    worst = STARTUP_MIN + N_TASKS_ASSUMED * (t + SETUP_MIN_PER_TASK)
    if worst > LIMIT_MIN - 30:
        fail(f"eval_config.yaml: worst-case runtime {worst:.0f} min leaves < 30 min of the 12 h limit")
    if sec.get("timeout_seconds", 300) < 300:
        print("WARNING: timeout_seconds < 300 may also bound Phase-2 verification pytest")
    return {"evaluation": sec, "worst_case_runtime_min": round(worst), "limit_min": LIMIT_MIN,
            "assumptions": {"tasks": N_TASKS_ASSUMED, "setup_min_per_task": SETUP_MIN_PER_TASK,
                            "startup_min": STARTUP_MIN}}


def local_checks(root: Path) -> dict:
    roots = [n for n in ROOT_NAMES if (root / n).exists()]
    if roots != ["agent.yaml"]:
        fail(f"need exactly agent.yaml at the root, found {roots}")
    files = [p for p in root.rglob("*") if p.is_file() or p.is_symlink()]
    if any(p.is_symlink() for p in root.rglob("*")):
        fail("symlinks are rejected")
    bad = [p.relative_to(root).as_posix() for p in files if p.suffix.lower() not in ALLOWED_EXT]
    if bad:
        fail(f"disallowed extensions: {bad}")
    hidden = [p.relative_to(root).as_posix() for p in files if any(part.startswith(".") for part in p.relative_to(root).parts)]
    if hidden:
        fail(f"hidden files: {hidden}")
    total = sum(p.stat().st_size for p in files)
    if total >= MAX_BYTES:
        fail("over 3 GiB")
    for p in files:
        if p.suffix in (".yaml", ".yml"):
            load(p)  # parses
    report: list = []
    check_agent(root, root / "agent.yaml", True, set(), report)
    # every file should be referenced (catches stale leftovers)
    referenced = {"agent.yaml", "eval_config.yaml"}
    for p in files:
        if p.suffix in (".yaml", ".yml"):
            for ref in re.findall(r"!include\s+(\S+)", p.read_text(encoding="utf-8")):
                referenced.add(posixpath.normpath(posixpath.join(p.parent.relative_to(root).as_posix(), ref)))
            for ref in re.findall(r"config_path:\s*(\S+)", p.read_text(encoding="utf-8")):
                referenced.add(posixpath.normpath(ref))
    unref = [p.relative_to(root).as_posix() for p in files if p.relative_to(root).as_posix() not in referenced]
    if unref:
        fail(f"files not referenced by any config: {unref}")
    return {"files": sorted(p.relative_to(root).as_posix() for p in files), "bytes": total,
            "agents": report, **check_eval_config(root)}


# Stand-ins with the real tool signatures (swegemma.tools): compiling only checks names/schemas.
def run_command(command: str) -> str: ...
def read_file(filepath: str, start_line: int | None = None, end_line: int | None = None) -> str: ...
def write_file(filepath: str, content: str) -> str: ...
def edit_file(filepath: str, old_string: str, new_string: str, allow_multiple: bool = False) -> str: ...
def submit_patch() -> str: ...
def get_status() -> str: ...
def get_code_neighbors(node: str, edge_type: str | None = None, max_neighbors: int = 50) -> str: ...
def search_similar_code(query: str, k: int = 10) -> str: ...
def get_code_subgraph(nodes: list[str]) -> str: ...


STAND_INS = {f.__name__: f for f in [run_command, read_file, write_file, edit_file, submit_patch, get_status,
                                     get_code_neighbors, search_similar_code, get_code_subgraph]}


def official_checks(root: Path) -> dict:
    try:
        from adk_submission import ModelRegistry, compile_submission, validate_directory
        from google.adk.models.lite_llm import LiteLlm
        from swegemma.config import ALLOWED_SUBMISSION_EXTENSIONS, build_submission_limits
        from swegemma.models.discovery import validate_single_declared_model
    except Exception as e:  # noqa: BLE001
        return {"official": f"skipped (harness not importable: {type(e).__name__}: {e})"}
    import asyncio

    from google.adk.agents.llm_agent import LlmAgent
    from google.adk.tools.agent_tool import AgentTool
    from google.adk.utils.instructions_utils import inject_session_state

    assert ALLOWED_EXT == set(ALLOWED_SUBMISSION_EXTENSIONS), sorted(ALLOWED_SUBMISSION_EXTENSIONS)
    limits, constraints = build_submission_limits()
    validate_directory(root, limits)
    declared = validate_single_declared_model(root)
    if declared != MODEL:
        fail(f"validate_single_declared_model returned {declared}")
    registry = ModelRegistry()
    registry.register(MODEL, LiteLlm(model=f"openai/{MODEL}", api_base="http://127.0.0.1:9/v1", api_key="EMPTY"))
    agent = compile_submission(root, STAND_INS, registry, limits=limits, generation_constraints=constraints)

    agents = []

    def walk(a):
        agents.append(a)
        for t in getattr(a, "tools", []) or []:
            if isinstance(t, AgentTool):
                walk(t.agent)

    walk(agent)

    class _Ctx:  # minimal ReadonlyContext stand-in for inject_session_state
        class _Inv:
            class session:  # noqa: N801
                state = {"problem_description": "Sample issue with {braces} and `code`."}
                app_name, user_id, id = "a", "u", "s"
            artifact_service = None
        _invocation_context = _Inv()

    rendered = {}
    for a in agents:
        if isinstance(a, LlmAgent) and isinstance(a.instruction, str):
            text = asyncio.run(inject_session_state(a.instruction, _Ctx()))
            if "Sample issue with {braces}" not in text:
                fail(f"{a.name}: problem_description was not injected")
            rendered[a.name] = len(text)
        gc = getattr(a, "generate_content_config", None)
        if gc is not None and gc.max_output_tokens is not None and gc.max_output_tokens > 8192:
            fail(f"{a.name}: compiled max_output_tokens {gc.max_output_tokens}")
    tool_names = sorted(getattr(t, "__name__", None) or getattr(t, "name", type(t).__name__) for t in agent.tools)
    return {"official": "ok", "root_agent": agent.name, "compiled_agents": [a.name for a in agents],
            "root_tools": tool_names, "rendered_instruction_chars": rendered}


def package(src: Path, zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_path.unlink(missing_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(p for p in src.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo(p.relative_to(src).as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, p.read_bytes())


def main() -> None:
    print(f"source: {SRC}")
    local = local_checks(SRC)
    print("local checks: ok")
    official = official_checks(SRC)
    print(f"official checks: {official['official']}")
    package(SRC, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH) as z:
        names = z.namelist()
        if "agent.yaml" not in names or any(n.endswith("/") or n.startswith("/") or ".." in n for n in names):
            fail(f"bad archive layout: {names}")
        if z.testzip() is not None:
            fail("corrupt zip")
        with tempfile.TemporaryDirectory() as td:
            z.extractall(td)
            rt_local = local_checks(Path(td))
            rt_official = official_checks(Path(td))
    if rt_local["files"] != local["files"] or rt_official["official"] != official["official"]:
        fail("round-trip mismatch")
    print("round trip: ok")
    sha = hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()
    manifest = {"zip": str(ZIP_PATH), "sha256": sha, "zip_bytes": ZIP_PATH.stat().st_size,
                "entries": names, **local, **official}
    (DIST / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    print(json.dumps({k: manifest[k] for k in ("zip", "sha256", "zip_bytes", "entries", "worst_case_runtime_min")}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
