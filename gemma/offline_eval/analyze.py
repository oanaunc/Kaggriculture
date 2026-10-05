"""Categorise offline-eval failures and compare runs task by task.

usage: python3 analyze.py RUN.json [OTHER_RUN.json]
"""
import json
import sys
from collections import Counter

TOOL_CAP = 60
TIME_CAP_S = 300


def category(t):
    if t.get("resolved"):
        return "resolved"
    err = (t.get("error") or t.get("error_message") or "") or ""
    if err:
        low = err.lower()
        if "context" in low or "maximum" in low or "too long" in low:
            return "context_overflow"
        if "timeout" in low or "time" in low:
            return "timeout"
        return "harness_error"
    if not t.get("patch_chars"):
        if (t.get("tool_calls") or 0) >= TOOL_CAP:
            return "no_patch_tool_cap"
        if (t.get("duration_seconds") or 0) >= TIME_CAP_S * 0.95:
            return "no_patch_time_cap"
        return "no_patch_other"
    tail = t.get("test_output_tail") or ""
    if "error" in tail.lower() and "passed" not in tail.lower():
        return "patch_breaks_tests"
    return "patch_wrong"


def load(path):
    return {t["instance_id"]: t for t in json.load(open(path))["tasks"]}


def summarize(path):
    tasks = load(path)
    cats = Counter(category(t) for t in tasks.values())
    n = len(tasks)
    print(f"{path}: {cats['resolved']}/{n} resolved")
    for c, k in cats.most_common():
        print(f"  {c:22s} {k:3d}  {k / n:.0%}")
    return tasks


def compare(a, b):
    common = sorted(set(a) & set(b))
    only_a = [i for i in common if a[i].get("resolved") and not b[i].get("resolved")]
    only_b = [i for i in common if b[i].get("resolved") and not a[i].get("resolved")]
    print(f"paired on {len(common)} tasks: A-only {len(only_a)}, B-only {len(only_b)}")
    # exact McNemar (binomial) on discordant pairs
    from math import comb
    n, k = len(only_a) + len(only_b), min(len(only_a), len(only_b))
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    print(f"  McNemar exact p = {p:.3f}")
    print("  A-only:", only_a)
    print("  B-only:", only_b)


if __name__ == "__main__":
    runs = [summarize(p) for p in sys.argv[1:]]
    if len(runs) == 2:
        compare(*runs)
