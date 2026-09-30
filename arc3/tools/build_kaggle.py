"""Build the Kaggle source-bundle dataset and notebook for our modified Duck solver.

usage: python build_kaggle.py <version_tag>
Produces:
  /tmp/arc3/build/dataset/   -> kaggle dataset  oanaunciuleanu/arc3-duck-ours-src
  /tmp/arc3/build/notebook/  -> kaggle kernel   oanaunciuleanu/arc3-duck-ours
The baseline bundle (serving setup, pickled benchmark, vLLM runtime/model refs)
comes from keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1; only the solver
sources are replaced with arc3/solver/.

ARC3_MODEL=swift serves the Swift 1.5 Kaggle model
michaelpoluektov/qwen3-8-swift-nvfp4/Transformers/default/1 instead of the
RadixArk checkpoint: arc3/serving/serving_setup_swift.py becomes the bundle's
serving_setup.py, arc3/serving/swift-staging/ is added, SOURCE_IDENTITY.json is
re-derived (serving_setup_sha256 of the copied file) and the kernel's
model_sources points at the Swift model.  See arc3/serving/README.md.
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
SWIFT_MODEL_SOURCE = "michaelpoluektov/qwen3-8-swift-nvfp4/Transformers/default/1"
MODEL = os.environ.get("ARC3_MODEL", "radixark").strip().lower()
if MODEL not in ("radixark", "swift"):
    raise SystemExit(f"ARC3_MODEL must be radixark or swift, got {MODEL!r}")
if MODEL == "swift":
    MODEL_SOURCE = SWIFT_MODEL_SOURCE

tag = sys.argv[1] if len(sys.argv) > 1 else "dev"
shutil.rmtree(OUT, ignore_errors=True)
ds = OUT / "dataset"
shutil.copytree(BASE_BUNDLE, ds, ignore=shutil.ignore_patterns("__pycache__"))
for repo in ("ARC3-Inference", "tufa-arc-agi-framework"):
    shutil.rmtree(ds / "src" / repo)
    shutil.copytree(REPO / "arc3" / "solver" / repo, ds / "src" / repo, ignore=shutil.ignore_patterns("__pycache__"))
if MODEL == "swift":
    serving = REPO / "arc3" / "serving"
    sys.path.insert(0, str(serving))
    import make_identity

    shutil.copyfile(serving / "serving_setup_swift.py", ds / "serving_setup.py")
    shutil.copytree(serving / "swift-staging", ds / "swift-staging")
    identity = make_identity.build(json.loads((BASE_BUNDLE / "SOURCE_IDENTITY.json").read_text()), ds / "serving_setup.py")
    (ds / "SOURCE_IDENTITY.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    print("swift model: serving_setup sha256", identity["serving_setup_sha256"])
(ds / "OURS_VERSION.txt").write_text(tag + (" model=swift" if MODEL == "swift" else "") + "\n")
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
nb["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": f"# ARC3 Duck (ours) - {tag}\nModified Tufa Labs Duck harness; see OURS_VERSION.txt in the source dataset." + (f"\n\nModel: `{SWIFT_MODEL_SOURCE}` (Swift 1.5 Qwen3.8-Flash-Next by UkisAI, Swift Open License v1.0 + Qwen Community License 1.0), staged by serving_setup.py." if MODEL == "swift" else "")})
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
