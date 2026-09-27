# [ours] Same launch as the base (`!... python starter.py --end-time ...`) plus:
#  - PYTHONHASHSEED=0 for reproducible runs,
#  - the starter terminates its workers at end_time + 180 s,
#  - and this cell kills the whole process group at end_time + 240 s as a last resort,
#    so the final cell always gets to write submission.json within the 12 h limit.
import os, sys, shutil, signal, subprocess, threading, time

run_env = dict(
    os.environ,
    PYTHONHASHSEED="0",
    UNSLOTH_DISABLE_STATISTICS="1",
    TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas",
    OMP_NUM_THREADS="12",
    ARC_ORDER=ARC_ORDER,
    ARC_EVAL_SUBSET=ARC_EVAL_SUBSET,
)
python_bin = shutil.which("python") or sys.executable  # same interpreter as the base's `!python`
starter = subprocess.Popen(
    [python_bin, "-u", "starter.py", "--end-time", str(global_end_time),
     "--hard-deadline", str(global_end_time + 180)],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    env=run_env, start_new_session=True,
)

def _pump(stream):
    for line in stream:
        print(line, end="", flush=True)

pump = threading.Thread(target=_pump, args=(starter.stdout,), daemon=True)
pump.start()

kill_at = global_end_time + 240
while starter.poll() is None and time.time() < kill_at:
    time.sleep(10)
if starter.poll() is None:
    print("*** starter still running at the kill deadline: killing its process group")
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(starter.pid, sig)
        except Exception as e:
            print("killpg:", e)
        time.sleep(10)
pump.join(timeout=10)
print("starter exit code:", starter.poll())
