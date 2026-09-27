"""Build arc2-nvarc-ours.ipynb (and a flat, compilable .py view) from src/.

    python build_notebook.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
NAME = "arc2-nvarc-ours"

INTRO = """# ARC-AGI-2: NVARC Qwen3-4B TTT + DFS (our hardened variant)

Derived from `mikelou1/arc-agi2-lb33-89-minimal-perfpatch` (NVARC per-task LoRA test-time
training, turbo DFS decoding over 16 augmented views, augmented-NLL rescoring, `score_kgmon`
selection, 4 x L4 workers). Model maths, hyper-parameters, prompts, token ids and the
selection algorithm are unchanged. Our changes (see README for the full diff):

1. **Determinism**: stable md5-based seeds instead of `hash(bk)`, `PYTHONHASHSEED=0`, per-task seeding, sorted result loading.
2. **Robust attempts**: `attempt_2` is never a duplicate of `attempt_1`, nor `[[0]]` when any other candidate exists.
3. **CPU program search** (background, niced): small DSL; only programs that reproduce *all* train pairs may fill empty/duplicate slots; it never replaces a model `attempt_1`.
4. **Time/crash safety**: placeholder submission written first; training stops at the deadline; hard kill of workers; a crashed rank no longer kills the other ranks (no `mp.spawn`); per-task exception handling; atomic pickle writes; tolerant loading; guarded final assembly and format validation.
5. **Scheduling**: cheap tasks first; tasks that crashed or produced no candidate get one retry with new augmentation seeds when time remains.
"""


def read(name):
    with open(os.path.join(SRC, name)) as f:
        return f.read().rstrip("\n") + "\n"


def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {"trusted": True},
            "outputs": [], "source": src.splitlines(keepends=True)}


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


def main():
    cells = [md_cell(INTRO), code_cell(read("cells/01_setup.py")), code_cell(read("cells/02_pip.py"))]
    for fn in ["arc_loader.py", "arc_decoder.py", "arc_solver.py", "starter.py", "arc_dsl.py", "arc_submit.py"]:
        cells.append(code_cell(f"%%writefile {fn}\n" + read(fn)))
    for fn in ["cells/10_launch.py", "cells/11_run.py", "cells/12_submit.py"]:
        cells.append(code_cell(read(fn)))

    base_meta_path = "/tmp/arc2/nb/mikelou1_arc-agi2-lb33-89-minimal-perfpatch/arc-agi2-lb33-89-minimal-perfpatch.ipynb"
    with open(base_meta_path) as f:
        meta = json.load(f)["metadata"]
    meta.pop("papermill", None)
    nb = {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 4}
    with open(os.path.join(HERE, f"{NAME}.ipynb"), "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    # Flat readable/compilable view: IPython magics are commented out, %%writefile
    # bodies are kept inline (they compile as module-level code).
    out = []
    for i, c in enumerate(cells):
        s = "".join(c["source"])
        if c["cell_type"] == "markdown":
            out.append(f"# %% [markdown] cell {i}\n" + "\n".join("# " + l for l in s.splitlines()) + "\n")
            continue
        lines = []
        for l in s.splitlines():
            if l.startswith("%%") or l.lstrip().startswith("!"):
                lines.append("# " + l)
            else:
                lines.append(l)
        out.append(f"# %% cell {i}\n" + "\n".join(lines) + "\n")
    with open(os.path.join(HERE, f"{NAME}.py"), "w") as f:
        f.write("\n".join(out))
    print(f"built {NAME}.ipynb with {len(cells)} cells")


if __name__ == "__main__":
    main()
