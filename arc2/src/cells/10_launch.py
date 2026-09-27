# [ours] 1) write a valid placeholder submission.json immediately, so a later failure
#    still leaves a well-formed file; 2) start the CPU program search in the background
#    (niced, uses otherwise idle CPU cores while the GPUs train).
import sys, subprocess, traceback
from arc_submit import write_placeholder

try:
    write_placeholder(TEST_PATH, "submission.json")
    print("placeholder submission.json written")
except Exception:
    traceback.print_exc()

dsl_proc = None
try:
    dsl_cmd = [sys.executable, "arc_dsl.py", TEST_PATH, "dsl_predictions.json",
               "--time-limit", str(DSL_TIME_LIMIT), "--per-task", "10"]
    dsl_log = open("dsl.log", "w")
    try:
        dsl_proc = subprocess.Popen(["nice", "-n", "19"] + dsl_cmd, stdout=dsl_log, stderr=subprocess.STDOUT)
    except FileNotFoundError:
        dsl_proc = subprocess.Popen(dsl_cmd, stdout=dsl_log, stderr=subprocess.STDOUT)
    print("program search started, pid", dsl_proc.pid)
except Exception:
    traceback.print_exc()
