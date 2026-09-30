"""Build the Kaggle source-bundle dataset and notebook for our modified Duck solver.

usage: python build_kaggle.py <version_tag>
Produces:
  /tmp/arc3/build/dataset/   -> kaggle dataset  oanaunciuleanu/arc3-duck-ours-src
  /tmp/arc3/build/notebook/  -> kaggle kernel   oanaunciuleanu/arc3-duck-ours
The baseline bundle (serving setup, pickled benchmark, vLLM runtime/model refs)
comes from keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1; only the solver
sources are replaced with arc3/solver/.
"""
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASE_BUNDLE = Path("/tmp/arc3/src/keith")
BASE_NOTEBOOK = Path("/tmp/arc3/nb/wuliao0_duck-qwen3-8-anim-base/duck-qwen3-8-anim-base.ipynb")
OUT = Path("/tmp/arc3/build")
USER = "oanaunciuleanu"
import os
DATASET_SLUG = os.environ.get("ARC3_DATASET_SLUG", "arc3-duck-ours-src")
KERNEL_SLUG = os.environ.get("ARC3_KERNEL_SLUG", "arc3-duck-ours")
RUNTIME_DATASET = "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"
MODEL_SOURCE = "keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"

tag = sys.argv[1] if len(sys.argv) > 1 else "dev"
shutil.rmtree(OUT, ignore_errors=True)
ds = OUT / "dataset"
shutil.copytree(BASE_BUNDLE, ds, ignore=shutil.ignore_patterns("__pycache__"))
for repo in ("ARC3-Inference", "tufa-arc-agi-framework"):
    shutil.rmtree(ds / "src" / repo)
    shutil.copytree(REPO / "arc3" / "solver" / repo, ds / "src" / repo, ignore=shutil.ignore_patterns("__pycache__"))
(ds / "OURS_VERSION.txt").write_text(tag + "\n")
(ds / "dataset-metadata.json").write_text(json.dumps({
    "title": DATASET_SLUG.replace("-", " "),
    "id": f"{USER}/{DATASET_SLUG}",
    "licenses": [{"name": "MIT"}],
}, indent=1))

nb = json.loads(BASE_NOTEBOOK.read_text())
for cell in nb["cells"]:
    src = "".join(cell["source"])
    if "DATASET_SOURCES = [" in src:
        start = src.index("DATASET_SOURCES = [")
        end = src.index("]", start) + 1
        src = src[:start] + f'DATASET_SOURCES = ["{USER}/{DATASET_SLUG}", "{RUNTIME_DATASET}"]' + src[end:]
    if os.environ.get("ARC3_QUICK_COMMIT") == "1" and "bm.games = [offline_by_id" in src and "\nbm.n_passes = 1\n" in src:
        # Interactive (non-submission) run: play 2 games for 2 minutes only, so the
        # commit needed before submitting costs minutes of GPU instead of ~2.3 h.
        src = src.replace(
            "\nbm.n_passes = 1\n",
            "\nif not TRUE_SUBMISSION:\n    bm.games = bm.games[:2]\n    bm.solver.max_runtime_s_per_game = 120.0\nbm.n_passes = 1\n",
            1,
        )
        print("quick-commit patch applied")
    if os.environ.get("ARC3_QUICK_COMMIT") == "1":
        # The public-eval summary cell insists on all 25 games; relax it for quick commits.
        src = src.replace(
            "if len(public_runs) != 25 or public_run_ids != list(PUBLIC_GAME_IDS):",
            "if (len(public_runs) != 25 or public_run_ids != list(PUBLIC_GAME_IDS)) and TRUE_SUBMISSION:",
        )
    cell["source"] = src
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": f"# ARC3 Duck (ours) - {tag}\nModified Tufa Labs Duck harness; see OURS_VERSION.txt in the source dataset."})
nbdir = OUT / "notebook"
nbdir.mkdir(parents=True)
(nbdir / f"{KERNEL_SLUG}.ipynb").write_text(json.dumps(nb, indent=1))
(nbdir / "kernel-metadata.json").write_text(json.dumps({
    "id": f"{USER}/{KERNEL_SLUG}",
    "title": KERNEL_SLUG,
    "code_file": f"{KERNEL_SLUG}.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "dataset_sources": [f"{USER}/{DATASET_SLUG}", RUNTIME_DATASET],
    "kernel_sources": [],
    "competition_sources": ["arc-prize-2026-arc-agi-3"],
    "model_sources": [MODEL_SOURCE],
    "machine_shape": "NvidiaRtxPro6000",
}, indent=1))
print("built", ds, nbdir)
