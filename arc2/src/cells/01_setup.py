# Keep the original global 10-minute submission/write buffer.
import time
global_end_time = time.time() + 12 * 3600 - 600

# [ours] run configuration
import os
ARC_ORDER = "cheap_first"      # "cheap_first" (ours) or "sorted" (base order)
ARC_EVAL_SUBSET = "smoke"      # commit (non-rerun) mode only: "smoke" = base 4 tasks, "all" = 120 eval tasks
DSL_TIME_LIMIT = 3600          # seconds for the background CPU program search
RERUN_MODE = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
COMP_DIR = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"
TEST_PATH = f"{COMP_DIR}/arc-agi_test_challenges.json" if RERUN_MODE else f"{COMP_DIR}/arc-agi_evaluation_challenges.json"
SOLUTIONS_PATH = None if RERUN_MODE else f"{COMP_DIR}/arc-agi_evaluation_solutions.json"
print("rerun mode:", RERUN_MODE, "| test file:", TEST_PATH)
