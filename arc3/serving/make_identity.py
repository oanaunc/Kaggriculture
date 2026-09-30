"""Write SOURCE_IDENTITY_swift.json from keith's identity + serving_setup_swift.py.

usage: python make_identity.py [base_identity] [setup] [out]
The serving setup refuses to start unless SOURCE_IDENTITY.json carries the
sha256 of the exact serving_setup.py next to it, so rerun this after every edit
(build_kaggle.py with ARC3_MODEL=swift also re-derives it at build time).
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/arc3/src/keith/SOURCE_IDENTITY.json")
SETUP = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "serving_setup_swift.py"
OUT = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "SOURCE_IDENTITY_swift.json"
STAGING = HERE / "swift-staging"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(base: dict, setup: Path) -> dict:
    ident = json.loads(json.dumps(base))
    setup_sha = sha(setup)
    ident["serving_setup_sha256"] = setup_sha
    ident["replacement_records"]["serving_setup.py"] = {
        "bytes": setup.stat().st_size,
        "sha256": setup_sha,
    }
    ident["model_manifest_sha256"] = None
    source_files = json.loads((STAGING / "SWIFT_SOURCE_FILES.json").read_text())
    ident["model"] = {
        "kaggle_model_source": source_files["kaggle_model_source"],
        "hf_source_repo": source_files["hf_source_repo"],
        "config_sha256": source_files["small_file_sha256"]["config.json"],
        "file_count": source_files["file_count"],
        "total_bytes": source_files["total_bytes"],
        "staged_config_sha256": sha(STAGING / "staged_config.json"),
        "staged_hf_quant_config_sha256": sha(STAGING / "staged_hf_quant_config.json"),
        "staging": "symlink tree + RadixArk config + bf16 MTP experts (see serving_setup.py)",
        "replaced_model": base.get("model"),
    }
    ident["swift_staging_artifacts"] = {
        f"swift-staging/{p.name}": {"bytes": p.stat().st_size, "sha256": sha(p)}
        for p in sorted(STAGING.iterdir())
    }
    ident["purpose"] = base.get("purpose", "") + " (Swift 1.5 model variant)"
    return ident


if __name__ == "__main__":
    identity = build(json.loads(BASE.read_text()), SETUP)
    OUT.write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n")
    print("wrote", OUT, identity["serving_setup_sha256"])
