"""Offline smoke test: run the Duck HarnessSolver on public games against the mock LLM.

usage: python smoke_run.py <solver_src_root> <bundle_dir> <env_files_dir> [n_games] [max_actions]
"""
import asyncio
import os
import pickle
import sys
from pathlib import Path

src_root, bundle, env_dir = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
n_games = int(sys.argv[4]) if len(sys.argv) > 4 else 2
max_actions = int(sys.argv[5]) if len(sys.argv) > 5 else 12
for p in (src_root / "tufa-arc-agi-framework" / "src", src_root / "ARC3-Inference"):
    sys.path.insert(0, str(p))
os.environ.setdefault("LOCAL_ANALYZER_BASE_URL", "http://127.0.0.1:18234/v1")
os.environ.setdefault("LOCAL_ANALYZER_MODEL_ID", "mock")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ["ONLY_RESET_LEVELS"] = "true"

import arc_agi  # noqa: E402
import taaf.game_api  # noqa: E402

work = Path(os.environ.get("SMOKE_WORK", "/tmp/arc3/smoke"))
work.mkdir(parents=True, exist_ok=True)
bm = pickle.load(open(bundle / "benchmark_initial.pkl", "rb"))
bm.job_dir = work
spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
ids = sorted(e.game_id for e in arcade.available_environments)[:n_games]
bm.games = [taaf.game_api.GameAPI(env_name=g, arcade_spec=spec) for g in ids]
bm.n_passes = 1
bm.game_weights = None
bm.solver.concurrency = n_games
bm.solver.max_actions_per_game = max_actions
bm.solver.max_runtime_s_per_game = 300.0
asyncio.run(bm.run(soft_end_time=None, runtime_environment=None, minimal_diagnostics=True))
