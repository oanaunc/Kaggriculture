---
name: verify-before-submit
description: >
  Pre-handoff gate: targeted test evidence and patch hygiene before the verify stage submits.
---

# Verify before submit

1. Run **only** the targeted test node or a short assertion-based repro for this fix. Printed booleans and swallowed exceptions are not passing evidence.
2. `git status` / `git diff`: no scratch under `/workspace`, no test/config edits.
3. Scratch belongs in `/tmp`.
4. On passing assertions and a clean diff → short summary and end your turn; the verify stage runs the existing tests and calls `submit_patch`.
5. If the targeted test fails → fix source again; do not edit tests.
