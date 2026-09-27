import os
import sys
import time
import json
import argparse
import traceback
import torch
import torch.multiprocessing as mp


def local_worker(rank, queue, retry_queue, end_time):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    # Fix Unsloth patching issue: ranks import unsloth one after the other.
    # [ours] bounded wait, so one rank dying during import cannot block the others forever.
    marker_dir = os.getenv("ARC_MARKER_DIR", "/kaggle")
    if rank > 0:
        waited = 0
        while not os.path.exists(f"{marker_dir}/worker{rank-1}") and waited < 1800:
            time.sleep(5)
            waited += 5

    try:
        from arc_solver import worker
    finally:
        with open(f"{marker_dir}/worker{rank}", "w") as f:
            f.write("Ok")

    print(f"[Rank {rank}] start!", flush=True)

    try:
        worker(rank, queue, end_time, retry_queue)
    except Exception:
        print(f"[Rank {rank}] worker crashed:\n{traceback.format_exc()}", flush=True)

    print(f"[Rank {rank}] done!", flush=True)


def task_cost(task):
    """[ours] Cheap proxy for per-task GPU time: cells seen in training (+ newlines) plus
    decode length per test output. Used to run cheap tasks first."""
    def cells(g):
        return len(g) * (len(g[0]) + 1) if len(g) else 0
    train = sum(cells(p["input"]) + cells(p["output"]) for p in task["train"])
    test = sum(cells(t["input"]) for t in task["test"])
    return train + test + len(task["test"]) * 2 * max((cells(t["input"]) for t in task["test"]), default=0)


def order_keys(data, mode):
    keys = sorted(data.keys())
    if mode == "cheap_first":
        keys = sorted(keys, key=lambda k: (task_cost(data[k]), k))
    return keys


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    parser.add_argument("--hard-deadline", type=float, default=0.0)
    args = parser.parse_args()

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    if rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    test_path = os.getenv("ARC_TEST_PATH") or test_path  # [ours] override for local tests

    with open(test_path, "r") as f:
        data = json.load(f)

    order = os.getenv("ARC_ORDER", "cheap_first")
    eval_subset = os.getenv("ARC_EVAL_SUBSET", "smoke")   # commit mode only: smoke | all
    num_gpus = int(os.getenv("ARC_NUM_GPUS", "4"))

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    queue = manager.Queue()
    retry_queue = manager.Queue()

    n_put = 0
    for key in order_keys(data, order):
        if not rerun_mode and eval_subset == "smoke":
            if key not in ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]:
                continue
        queue.put(key)
        n_put += 1
    print(f"[starter] queued {n_put} tasks (order={order}, gpus={num_gpus})", flush=True)

    # [ours] independent processes instead of mp.spawn: with mp.spawn one crashed rank
    # terminates all the other ranks; here the survivors keep draining the queue.
    procs = []
    for rank in range(num_gpus):
        p = ctx.Process(target=local_worker, args=(rank, queue, retry_queue, args.end_time))
        p.start()
        procs.append(p)

    hard_deadline = args.hard_deadline or (args.end_time + 180 if args.end_time else 0)
    while any(p.is_alive() for p in procs):
        for p in procs:
            p.join(timeout=5)
        if hard_deadline and time.time() > hard_deadline:
            print("[starter] hard deadline reached, terminating workers", flush=True)
            for p in procs:
                if p.is_alive():
                    p.terminate()
            time.sleep(10)
            for p in procs:
                if p.is_alive():
                    p.kill()
            break

    print(f"[starter] exit codes: {[p.exitcode for p in procs]}", flush=True)
    try:
        manager.shutdown()
    except Exception:
        pass
