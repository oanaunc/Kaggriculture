"""Unit checks for our solver modifications (run with the arc3 venv).

PYTHONPATH=arc3/solver/tufa-arc-agi-framework/src:arc3/solver/ARC3-Inference python arc3/tools/test_solver_changes.py
"""
import os

os.environ.setdefault("MPLBACKEND", "Agg")
import arc_agi  # noqa: E402
import arcengine  # noqa: E402

from inference.agent.tool_agent import ToolAgent, _extract_scientist_note  # noqa: E402
from inference.framework import solver as S  # noqa: E402

# 1. world-model label extraction tolerates the headers the model actually writes
text = """Some reasoning.
**World model (revised):** player is the 2x2 green block
World model v12: extra line
### Goal model update: reach the yellow core
- Action model: UP/DOWN move player; clicks toggle bridges
Plan for next level: walk right
Planned moves are fine but not a label
"""
note = _extract_scientist_note(text)
assert "player is the 2x2 green block" in note["world_model"], note
assert "extra line" in note["world_model"], note
assert note["goal_model"] == "reach the yellow core", note
assert note["action_model"].startswith("UP/DOWN"), note
assert note["current_plan"].startswith("walk right"), note
assert _extract_scientist_note("World model: a\nPlan: b")["current_plan"] == "b"
print("label extraction ok")

# 2. auto-reset clears the stale game-over state
a = ToolAgent.__new__(ToolAgent)
a._last_step_summary = {"game_over": True, "executed_count": 3, "executed_actions": ["UP"]}
a._last_action_result = {"game_over": True, "state": "GAME_OVER"}
a.note_auto_reset(2)
assert a._last_step_summary["game_over"] is False and a._last_step_summary["auto_reset"]
assert a._last_action_result["game_over"] is False
print("auto-reset notice ok")

# 3. animation summary on a real game: find an action with transient animation frames
arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir="/tmp/arc3/data/environment_files")
import taaf.game_api  # noqa: E402

spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir="/tmp/arc3/data/environment_files")
found = 0
for gid in ["sp80-589a99af", "sb26-7fbdac44", "tn36-ef4dde99"]:
    g = taaf.game_api.GameAPI(env_name=gid, arcade_spec=spec)
    g.start_game()
    for act in g.current_state.available_actions:
        if act in (0, 6):
            continue
        prev = S._grid_from_state(g.current_state)
        st = g.execute_action(arcengine.ActionInput(id=arcengine.GameAction.from_id(act), data={}), generated_tokens=0, uncached_input_tokens=0)
        summ = S._animation_summary(prev, st)
        if summ:
            found += 1
            print(gid, arcengine.GameAction.from_id(act).name, summ)
print("animation summaries found:", found)
assert found > 0
