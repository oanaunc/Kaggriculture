"""[ours] Submission assembly: attempt selection, verified-program fill, format validation.

Pure json + numpy (no torch / transformers), so it can be unit-tested on CPU and can
always run, even if every GPU worker crashed.

Slot policy for every test output (task_id, i):
  1. model candidates, in the base selection order (score_kgmon), de-duplicated and
     sanitised -> attempt_1, attempt_2   (the model's attempt_1 is NEVER replaced)
  2. still-empty slots are filled with CPU program-search candidates (programs that
     reproduce every train pair), skipping grids already used
  3. anything still empty gets a cheap guess (the test input itself), then [[0]]
"""
import json
import os

import numpy as np

PLACEHOLDER = [[0]]


def sanitize_grid(g):
    """Return g as a list of lists of ints in 0..9, rectangular, 1..30 per side; else None."""
    try:
        a = np.asarray(g)
        if a.dtype == object or a.ndim != 2:
            return None
        if not (1 <= a.shape[0] <= 30 and 1 <= a.shape[1] <= 30):
            return None
        if a.dtype.kind == "f":
            if not np.all(np.equal(np.mod(a, 1), 0)):
                return None
        elif a.dtype.kind not in "iub":
            return None
        a = a.astype(np.int64)
        if a.min() < 0 or a.max() > 9:
            return None
        return a.tolist()
    except Exception:
        return None


def _key(g):
    return json.dumps(g, separators=(",", ":"))


def choose_attempts(model_cands, dsl_cands=(), fallback_cands=(), n=2):
    """Pick n distinct grids. Returns (attempts, sources)."""
    chosen, sources, seen = [], [], set()
    for src, cands in (("model", model_cands), ("dsl", dsl_cands), ("fallback", fallback_cands)):
        for c in cands or ():
            if len(chosen) >= n:
                break
            g = sanitize_grid(c)
            if g is None:
                continue
            k = _key(g)
            if k in seen:
                continue
            seen.add(k)
            chosen.append(g)
            sources.append(src)
    while len(chosen) < n:
        chosen.append(PLACEHOLDER)
        sources.append("placeholder")
    return chosen, sources


def build_submission(challenges, model_ranked=None, dsl_preds=None):
    """challenges: {task_id: task}; model_ranked: {f"{tid}_{i}": [grid, ...]} ranked;
    dsl_preds: {tid: [[grid, ...] per test input]}. Returns (submission, stats)."""
    model_ranked = model_ranked or {}
    dsl_preds = dsl_preds or {}
    submission, stats = {}, {}
    for tid in sorted(challenges):
        tests = challenges[tid].get("test", [])
        entries = []
        for i, t in enumerate(tests):
            model_c = model_ranked.get(f"{tid}_{i}", [])
            try:
                dsl_c = dsl_preds.get(tid, [])[i]
            except Exception:
                dsl_c = []
            fb = [t.get("input")] if isinstance(t, dict) else []
            attempts, sources = choose_attempts(model_c, dsl_c, fb)
            entries.append({"attempt_1": attempts[0], "attempt_2": attempts[1]})
            for slot, s in zip(("attempt_1", "attempt_2"), sources):
                stats[f"{slot}:{s}"] = stats.get(f"{slot}:{s}", 0) + 1
        if not entries:  # defensive: a task without test inputs still needs an entry
            entries = [{"attempt_1": PLACEHOLDER, "attempt_2": PLACEHOLDER}]
        submission[tid] = entries
    return submission, stats


def validate_submission(submission, challenges):
    """Return a list of problems (empty list == valid)."""
    problems = []
    for tid, task in challenges.items():
        if tid not in submission:
            problems.append(f"missing task {tid}")
            continue
        entries = submission[tid]
        n = max(1, len(task.get("test", [])))
        if not isinstance(entries, list) or len(entries) != n:
            problems.append(f"{tid}: expected {n} entries")
            continue
        for i, e in enumerate(entries):
            for slot in ("attempt_1", "attempt_2"):
                if not isinstance(e, dict) or slot not in e:
                    problems.append(f"{tid}[{i}] missing {slot}")
                elif sanitize_grid(e[slot]) != e[slot]:
                    problems.append(f"{tid}[{i}] {slot} is not a valid grid")
    for tid in submission:
        if tid not in challenges:
            problems.append(f"extra task {tid}")
    return problems


def write_json_atomic(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def load_dsl_predictions(path):
    try:
        with open(path) as f:
            d = json.load(f)
        return d.get("predictions", {}) or {}
    except Exception as e:
        print(f"*** No program-search predictions ({e!r})")
        return {}


def write_placeholder(challenges_path, out_path="submission.json"):
    """Valid submission before any GPU work (test input as guess, [[0]] otherwise)."""
    with open(challenges_path) as f:
        challenges = json.load(f)
    sub, _ = build_submission(challenges)
    assert not validate_submission(sub, challenges)
    write_json_atomic(sub, out_path)
    return sub


def score_submission(submission, solutions):
    """Official metric: per task, mean over test outputs of any-attempt exact match."""
    total = 0.0
    for tid, sols in solutions.items():
        for i, s in enumerate(sols):
            e = submission.get(tid, [])
            if i < len(e) and (e[i].get("attempt_1") == s or e[i].get("attempt_2") == s):
                total += 1.0 / len(sols)
    return total
