Verify the uncommitted fix in `/workspace` for this issue: {problem_description}

You are the VERIFY stage. The coder has edited source but has not submitted. Your previous verify note (empty on the first pass): {verify_report?}

1. Run `cd /workspace && git status --short && git diff --stat`. If no source file changed, call submit_patch and reply `NO CHANGE`.
2. Find the existing test file closest to the changed source with one search, e.g. `ls tests | grep -i <module stem>`. Run ONE check: `cd /workspace && python -c "import <changed module>" && timeout 50 python -m pytest -x -q -p no:cacheprovider <test file> 2>&1 | tail -n 25`. If no test file matches, run the import only.
3. If the check passes, call submit_patch, then reply `VERIFIED` in one line.
4. If a test fails, decide whether the patch caused it: `cd /workspace && git stash -q && timeout 40 python -m pytest -x -q -p no:cacheprovider <same test file> 2>&1 | tail -n 5; git stash pop -q`. If it also fails without the patch, or the failing assertion checks the old behavior that the issue asks to change, call submit_patch and reply `EXPECTED`.
5. Otherwise the patch broke existing behavior: read the failing source lines and make ONE minimal edit_file fix in non-test source that keeps the requested behavior. Re-run the step-2 check. If it passes, call submit_patch and reply `FIXED`. If it still fails, do not submit; reply one line `FAIL: <test id> <error>`.
Never edit tests, never revert the whole patch, never run the full test suite. Keep every command output short.
