# [ours] Assemble submission.json. Every step is guarded: whatever fails, a valid file
# covering every test output with 2 attempts is written.
import json, traceback
from arc_submit import (build_submission, validate_submission, write_json_atomic,
                        load_dsl_predictions, score_submission)

with open(TEST_PATH) as f:
    challenges = json.load(f)

model_ranked, data, decoder = {}, None, None
try:
    from arc_loader import ArcDataset
    from arc_decoder import ArcDecoder
    data = ArcDataset.from_file(TEST_PATH)
    if not RERUN_MODE:
        data = data.load_replies(SOLUTIONS_PATH)
    decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)
    decoder.load_decoded_results("/kaggle/inference_outputs")
    model_ranked = decoder.run_selection_algo()   # base selection: score_kgmon
except Exception:
    traceback.print_exc()

_dsl = globals().get("dsl_proc")
if _dsl is not None and _dsl.poll() is None:
    print("*** program search still running: stopping it")
    try:
        _dsl.kill()
    except Exception:
        pass
dsl_preds = load_dsl_predictions("dsl_predictions.json")
print(f"*** program search produced candidates for {len(dsl_preds)} tasks")

submission, stats = build_submission(challenges, model_ranked, dsl_preds)
problems = validate_submission(submission, challenges)
print("*** slot sources:", json.dumps(stats, sort_keys=True))
if problems:
    print("*** VALIDATION PROBLEMS:", problems[:20])
    submission, _ = build_submission(challenges)  # last resort, valid by construction
write_json_atomic(submission, "submission.json")
print(f"*** wrote submission.json with {len(submission)} tasks")

if not RERUN_MODE:
    with open(SOLUTIONS_PATH) as f:
        solutions = json.load(f)
    model_only, _ = build_submission(challenges, model_ranked, None)
    print("*** Score (model only, all eval tasks):", score_submission(model_only, solutions))
    print("*** Score (final, all eval tasks):", score_submission(submission, solutions))
    try:
        decoder.benchmark_selection_algos()
        with open("submission.json", "r") as f:
            reload_submission = json.load(f)
        print("*** Reload score:", data.validate_submission(reload_submission))
    except Exception:
        traceback.print_exc()
    # keep the raw candidate pools for CPU-side selection experiments
    try:
        import shutil
        shutil.make_archive("inference_outputs", "gztar", "/kaggle/inference_outputs")
    except Exception:
        traceback.print_exc()
