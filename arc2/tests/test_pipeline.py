"""CPU tests for our selection / fallback / submission-writing / worker-loop code paths.

Run:  /tmp/arc2/venv/bin/python -m pytest -q /home/user/Kaggriculture/arc2/tests
"""
import bz2
import json
import os
import pickle
import queue as queue_mod
import subprocess
import sys
import time

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, SRC)
sys.path.insert(0, HERE)

import fakes  # noqa: E402

fakes.install_stubs()

import arc_submit  # noqa: E402
from arc_decoder import ArcDecoder  # noqa: E402
import arc_solver  # noqa: E402
from arc_loader import ArcDataset  # noqa: E402


# ----------------------------------------------------------------------------- arc_submit

G1 = [[1, 2], [3, 4]]
G2 = [[5]]
G3 = [[6, 6]]


def test_attempt2_never_duplicates_attempt1():
    att, src = arc_submit.choose_attempts([G1, np.array(G1), G1], [G1, G2])
    assert att == [G1, G2] and src == ["model", "dsl"]


def test_model_attempt1_never_replaced_and_model_top2_kept():
    att, src = arc_submit.choose_attempts([G1, G2], [G3])
    assert att == [G1, G2] and src == ["model", "model"]


def test_dsl_fills_both_when_model_empty():
    att, src = arc_submit.choose_attempts([], [G2, G3])
    assert att == [G2, G3] and src == ["dsl", "dsl"]


def test_fallback_then_placeholder():
    att, src = arc_submit.choose_attempts([], [], [G1])
    assert att == [G1, [[0]]] and src == ["fallback", "placeholder"]
    att, src = arc_submit.choose_attempts([G1], [], [G1])  # fallback equal to attempt_1 -> skipped
    assert att == [G1, [[0]]]


def test_invalid_grids_are_skipped():
    bad = [np.zeros((31, 2), int), [[1, 2], [3]], [[10]], [[-1]], [[0.5]], "x", None, np.zeros((0, 3), int)]
    att, src = arc_submit.choose_attempts(bad + [G1], [], [])
    assert att[0] == G1 and src[0] == "model"
    assert arc_submit.sanitize_grid(np.array([[1.0, 2.0]])) == [[1, 2]]


def _challenges():
    return {
        "aaaa0001": {"train": [], "test": [{"input": G1}]},
        "aaaa0002": {"train": [], "test": [{"input": G2}, {"input": G3}]},
        "aaaa0003": {"train": [], "test": [{"input": [[7] * 31]}]},  # invalid input -> placeholder
    }


def test_build_submission_covers_everything():
    ch = _challenges()
    model = {"aaaa0002_1": [np.array(G1)]}
    dsl = {"aaaa0002": [[G3], [G2]]}
    sub, stats = arc_submit.build_submission(ch, model, dsl)
    assert arc_submit.validate_submission(sub, ch) == []
    assert set(sub) == set(ch)
    assert len(sub["aaaa0002"]) == 2
    assert sub["aaaa0002"][1] == {"attempt_1": G1, "attempt_2": G2}
    assert sub["aaaa0002"][0] == {"attempt_1": G3, "attempt_2": G2}  # dsl, then test input
    assert sub["aaaa0003"][0] == {"attempt_1": [[0]], "attempt_2": [[0]]}
    json.loads(json.dumps(sub))  # JSON serialisable (no numpy types)


def test_validate_detects_problems():
    ch = _challenges()
    sub, _ = arc_submit.build_submission(ch)
    del sub["aaaa0001"]
    sub["aaaa0002"][0]["attempt_2"] = [[11]]
    sub["zzzz"] = []
    p = arc_submit.validate_submission(sub, ch)
    assert any("missing task aaaa0001" in x for x in p)
    assert any("aaaa0002[0] attempt_2" in x for x in p)
    assert any("extra task zzzz" in x for x in p)


def test_placeholder_and_score(tmp_path):
    ch = _challenges()
    cp = tmp_path / "c.json"
    cp.write_text(json.dumps(ch))
    out = tmp_path / "submission.json"
    sub = arc_submit.write_placeholder(str(cp), str(out))
    assert json.loads(out.read_text()) == sub
    sol = {"aaaa0001": [G1], "aaaa0002": [G1, G3]}
    # identity guess is right for aaaa0001 and the second output of aaaa0002
    assert arc_submit.score_submission(sub, sol) == pytest.approx(1.5)


# ----------------------------------------------------------------------------- decoder loading + selection

def _write(store, name, results):
    with bz2.BZ2File(os.path.join(store, name), "w") as f:
        pickle.dump(results, f)


def test_decoder_loading_is_robust_and_selection_ranks(tmp_path):
    store = tmp_path / "out"
    store.mkdir()
    good, bad = np.array(G1), np.array(G2)
    for v in range(3):
        _write(str(store), f"t0000001_0.v{v}", [
            {"beam_score": 0.1, "score_aug": [0.2] * 8, "solution": good},
            {"beam_score": 0.5, "score_aug": [1.0] * 8, "solution": bad},
        ])
    _write(str(store), "t0000001_0.v9.r1", [{"beam_score": 0.3, "score_aug": [3.0] * 8, "solution": np.array(G3)}])
    (store / "t0000001_0.v5").write_bytes(b"BZh91AY&SY-truncated")  # corrupt
    (store / "t0000001_0.v6.tmp").write_bytes(b"partial")
    dec = ArcDecoder(dataset=None, n_guesses=2)
    dec.load_decoded_results(str(store))
    dec.load_decoded_results(str(tmp_path / "does-not-exist"))
    ranked = dec.run_selection_algo()
    assert list(ranked) == ["t0000001_0"]
    assert np.array_equal(ranked["t0000001_0"][0], good)
    assert np.array_equal(ranked["t0000001_0"][1], bad)
    assert len(ranked["t0000001_0"]) == 3  # retry-pass candidate pooled in


# ----------------------------------------------------------------------------- determinism

def test_stable_seed_independent_of_pythonhashseed():
    code = ("import sys; sys.path.insert(0, %r); sys.path.insert(0, %r); import fakes; fakes.install_stubs(); "
            "import arc_solver; print(arc_solver.stable_seed('0934a4d8_0'), hash('0934a4d8_0') %% 1024**2)") % (SRC, HERE)
    outs = []
    for hs in ("1", "2"):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env=dict(os.environ, PYTHONHASHSEED=hs))
        assert r.returncode == 0, r.stderr
        outs.append(r.stdout.split())
    assert outs[0][0] == outs[1][0]          # ours: identical
    assert outs[0][1] != outs[1][1]          # base hash(bk): differs between runs


def test_augment_with_stable_seed_is_reproducible():
    q = {"k_0": {"train": [{"input": [[1, 2]], "output": [[2, 1]]}] * 3, "test": [{"input": [[3, 4]]}]}}
    a = ArcDataset(queries=q, replies={"k_0": [[[4, 3]]]}, keys=["k_0"]).augment(seed=arc_solver.stable_seed("k_0"))
    b = ArcDataset(queries=q, replies={"k_0": [[[4, 3]]]}, keys=["k_0"]).augment(seed=arc_solver.stable_seed("k_0"))
    assert a.keys == b.keys


# ----------------------------------------------------------------------------- worker loop (fake model)

TASKS = {
    "a0000001": {"train": [{"input": [[1, 2], [3, 4]], "output": [[1, 2], [3, 4]]}] * 2,
                 "test": [{"input": [[5, 6], [7, 8]]}]},
    "a0000002": {"train": [{"input": [[1, 0, 2]], "output": [[1, 0, 2]]}] * 2,
                 "test": [{"input": [[3, 0, 3]]}, {"input": [[4], [4], [0]]}]},
    "a0000003": {"train": [{"input": [[9]], "output": [[9]]}] * 2, "test": [{"input": [[2, 2]]}]},
    "a0000004": {"train": [{"input": [[5, 5], [0, 5]], "output": [[5, 5], [0, 5]]}] * 2,
                 "test": [{"input": [[1, 0], [0, 1]]}]},
}
SOLUTIONS = {k: [t["input"] for t in v["test"]] for k, v in TASKS.items()}


@pytest.fixture
def env(tmp_path, monkeypatch):
    tp = tmp_path / "tasks.json"
    tp.write_text(json.dumps(TASKS))
    out = tmp_path / "outputs"
    monkeypatch.setenv("ARC_TEST_PATH", str(tp))
    monkeypatch.setenv("ARC_OUTPUT_DIR", str(out))
    monkeypatch.setattr(arc_solver, "inference_turbo_dfs", fakes.fake_inference_turbo_dfs)
    monkeypatch.setattr(arc_solver, "calc_scores", fakes.fake_calc_scores)
    fakes.SCRIPT.clear()
    fakes.STATE.update(current=None, attempts={}, callbacks_seen=0, stopped_by_deadline=0)
    return tp, out


def _assemble(out):
    dec = ArcDecoder(dataset=None, n_guesses=2)
    dec.load_decoded_results(str(out))
    sub, stats = arc_submit.build_submission(TASKS, dec.run_selection_algo(), None)
    assert arc_submit.validate_submission(sub, TASKS) == []
    return sub, stats


def _queues(keys):
    q, rq = queue_mod.Queue(), queue_mod.Queue()
    for k in keys:
        q.put(k)
    return q, rq


def test_worker_end_to_end_with_crash_empty_and_retry(env, capsys):
    tp, out = env
    fakes.SCRIPT.update({
        "a0000002": ["raise_train", "ok"],   # crash (e.g. OOM) then succeed on retry
        "a0000003": ["empty", "ok"],         # no candidates, retry with new seeds
        "a0000004": ["raise_dfs", "raise_dfs"],  # fails twice -> only one retry
    })
    q, rq = _queues(sorted(TASKS))
    arc_solver.worker(0, q, time.time() + 3600, rq)
    log = capsys.readouterr().out
    assert fakes.STATE["attempts"] == {"a0000001": 1, "a0000002": 2, "a0000003": 2, "a0000004": 2}
    assert "RETRY a0000002" in log and "RETRY a0000003" in log
    files = os.listdir(out)
    assert any(f.endswith(".r1") for f in files) and not any(f.endswith(".tmp") for f in files)
    sub, stats = _assemble(out)
    score = arc_submit.score_submission(sub, SOLUTIONS)
    # a0000001..3 solved by the (fake) model, a0000004 only via the test-input fallback
    assert score == pytest.approx(4.0)
    assert sub["a0000004"][0]["attempt_1"] == TASKS["a0000004"]["test"][0]["input"]
    assert stats["attempt_1:model"] == 4   # a0000001, a0000002 (x2), a0000003 from the model
    assert stats["attempt_1:fallback"] == 1
    assert stats["attempt_2:model"] == 4   # distinct wrong beams give a distinct attempt_2
    assert "tokenizer check ok" in log
    for k, entries in sub.items():
        for e in entries:
            assert e["attempt_1"] != e["attempt_2"] or e["attempt_2"] == [[0]]


def test_worker_stops_after_consecutive_failures(env):
    tp, out = env
    for k in TASKS:
        fakes.SCRIPT[k] = ["raise_train", "raise_train"]
    q, rq = _queues(sorted(TASKS))
    arc_solver.worker(0, q, time.time() + 3600, rq)
    # 3 consecutive failures -> the rank exits; the 4th task is left for other ranks
    assert sum(fakes.STATE["attempts"].values()) == 3
    assert q.qsize() == 1 and rq.qsize() == 3


def test_worker_respects_deadline_and_skips_retry(env, monkeypatch):
    tp, out = env
    q, rq = _queues(["a0000001"])
    arc_solver.worker(0, q, time.time() + 60, rq)  # < MIN_TASK_START_SECONDS left
    assert fakes.STATE["attempts"] == {} and q.qsize() == 1
    # retry needs RETRY_MIN_SECONDS (600) + 60 remaining
    fakes.SCRIPT["a0000003"] = ["empty", "ok"]
    q, rq = _queues(["a0000003"])
    arc_solver.worker(0, q, time.time() + 500, rq)
    assert fakes.STATE["attempts"] == {"a0000003": 1}


def test_deadline_callback_stops_training():
    cb = arc_solver.DeadlineCallback(time.time() - 1)
    c = fakes._Control()
    cb.on_step_end(None, None, c)
    assert c.should_training_stop
    cb = arc_solver.DeadlineCallback(time.time() + 100)
    c = fakes._Control()
    cb.on_step_end(None, None, c)
    assert not c.should_training_stop


# ----------------------------------------------------------------------------- starter process management

FAKE_SOLVER = '''
import os, time, json
def worker(rank, queue, end_time, retry_queue=None):
    if rank == 0:
        os._exit(3)                      # hard crash of one rank
    if rank == 2:
        while True: time.sleep(1)        # a hung rank, must be killed by the hard deadline
    done = []
    while True:
        try:
            k = queue.get_nowait()
        except Exception:
            break
        done.append(k)
    with open(os.path.join(os.environ["ARC_MARKER_DIR"], f"done{rank}.json"), "w") as f:
        json.dump(done, f)
'''


def test_starter_survives_crashed_and_hung_ranks(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "arc_solver.py").write_text(FAKE_SOLVER)
    (work / "starter.py").write_text(open(os.path.join(SRC, "starter.py")).read())
    tp = tmp_path / "tasks.json"
    tp.write_text(json.dumps(TASKS))
    end = time.time() + 5
    t0 = time.time()
    r = subprocess.run([sys.executable, "starter.py", "--end-time", str(end), "--hard-deadline", str(end + 5)],
                       cwd=work, capture_output=True, text=True, timeout=120,
                       env=dict(os.environ, ARC_TEST_PATH=str(tp), ARC_MARKER_DIR=str(work),
                                ARC_NUM_GPUS="3", KAGGLE_IS_COMPETITION_RERUN="1", PYTHONPATH=str(work)))
    assert r.returncode == 0, r.stderr
    assert time.time() - t0 < 60
    done = json.loads((work / "done1.json").read_text())
    assert sorted(done) == sorted(TASKS)   # the surviving rank processed every task
    assert "hard deadline reached" in r.stdout
    # cheap-first order
    import starter as st  # noqa: F401  (import from src for ordering check)
    order = st.order_keys(TASKS, "cheap_first")
    assert order[0] == "a0000003" and set(order) == set(TASKS)
    assert st.order_keys(TASKS, "sorted") == sorted(TASKS)
