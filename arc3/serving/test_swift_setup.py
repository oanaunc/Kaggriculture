"""Offline (no GPU) test of serving_setup_swift.py's model discovery and staging.

usage: python test_swift_setup.py <swift_small_files_dir> <work_dir> [probe_dir]

<swift_small_files_dir> holds the real small files of the Swift Kaggle model
(config.json, hf_quant_config.json, SWIFT_FP8_EXPORT.json, the index, tokenizer
files, LICENSE, README.md, ...) plus hdr_mtp.json, the raw safetensors header of
mtp-nvidia-fp8.safetensors (fetched with an HTTP range request).  The test builds
a fake mount under /kaggle/input/models/... with those files and sparse weight
shards of the exact published sizes, runs the setup's non-GPU steps (identity,
discovery, sanity check, staging with the real MTP header, staged-model gate,
server command), and checks the MTP dequantization.  When probe_dir contains
raw_<e>_<proj>.npy / sc_<e>_<proj>.npy (real bytes of experts 0/7/511) they are
spliced into the fake MTP file and compared after staging; ml_dtypes, when
installed, is used as the reference fp8/bf16 implementation.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SMALL = Path(sys.argv[1])
WORK = Path(sys.argv[2])
PROBE = Path(sys.argv[3]) if len(sys.argv) > 3 else None
MOUNT = Path("/kaggle/input/models/michaelpoluektov/qwen3-8-swift-nvfp4/transformers/default/1")

source_files = json.loads((HERE / "swift-staging" / "SWIFT_SOURCE_FILES.json").read_text())

# ---- fake bundle -----------------------------------------------------------
shutil.rmtree(WORK, ignore_errors=True)
bundle = WORK / "bundle"
bundle.mkdir(parents=True)
shutil.copy(HERE / "serving_setup_swift.py", bundle / "serving_setup.py")
shutil.copytree(HERE / "swift-staging", bundle / "swift-staging")
sys.path.insert(0, str(HERE))
import make_identity  # noqa: E402

identity = make_identity.build(
    json.loads(Path("/tmp/arc3/src/keith/SOURCE_IDENTITY.json").read_text()),
    bundle / "serving_setup.py",
)
(bundle / "SOURCE_IDENTITY.json").write_text(json.dumps(identity, indent=2, sort_keys=True))
committed = json.loads((HERE / "SOURCE_IDENTITY_swift.json").read_text())
assert committed["serving_setup_sha256"] == identity["serving_setup_sha256"], (
    "SOURCE_IDENTITY_swift.json is stale; rerun make_identity.py"
)

# ---- fake Kaggle model mount -------------------------------------------------
shutil.rmtree(MOUNT, ignore_errors=True)
MOUNT.mkdir(parents=True)
hdr = (SMALL / "hdr_mtp.json").read_bytes()
mtp_header = json.loads(hdr)
data_start = 8 + len(hdr)
for name, size in source_files["files"].items():
    path = MOUNT / name
    if (SMALL / name).is_file():
        shutil.copy(SMALL / name, path)
    elif name == "mtp-nvidia-fp8.safetensors":
        with path.open("wb") as handle:
            handle.write(len(hdr).to_bytes(8, "little"))
            handle.write(hdr)
            handle.truncate(size)
            if PROBE is not None:
                for expert in (0, 7, 511):
                    for proj in ("gate_proj", "up_proj", "down_proj"):
                        for suffix, arr in (
                            ("weight", np.load(PROBE / f"raw_{expert}_{proj}.npy")),
                            ("weight_scale_inv", np.load(PROBE / f"sc_{expert}_{proj}.npy")),
                        ):
                            row = mtp_header[f"mtp.layers.0.mlp.experts.{expert}.{proj}.{suffix}"]
                            handle.seek(data_start + row["data_offsets"][0])
                            handle.write(arr.tobytes())
    else:
        with path.open("wb") as handle:
            handle.truncate(size)  # sparse: stat-only sanity check
    assert path.stat().st_size == size, name

# ---- import the setup like the notebook does -----------------------------------
os.environ["TAAF_KAGGLE_BUNDLE_DIR"] = str(bundle)
os.environ["TAAF_KAGGLE_WORKING_DIR"] = str(WORK / "working")
os.environ["TAAF_KAGGLE_SETUP_ENV"] = str(WORK / "setup-env.json")
os.environ.pop("TAAF_KAGGLE_INPUT_PATHS", None)
spec = importlib.util.spec_from_file_location("swift_setup", bundle / "serving_setup.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)
setup.SWIFT_STAGED_MODEL_DIR = WORK / "staged"

setup.source_identity()
payload = setup.swift_staging_payload()
swift_dir = setup.resolve_model_dir()
assert swift_dir == MOUNT, swift_dir
check = setup.verify_swift_source(swift_dir, payload["source_files"])
t0 = time.monotonic()
staged, record = setup.stage_swift_model(swift_dir, payload)
stage_seconds = time.monotonic() - t0
model_check = setup.verify_model(staged, full_file_hashes=False)
command = setup.server_command(staged, tuning=setup.resolve_vllm_tuning())
assert command[command.index("serve") + 1] == str(staged)
assert command[command.index("--quantization") + 1] == "modelopt_fp4"
assert command[command.index("--served-model-name") + 1] == "Qwen/Qwen3.8-Flash-Next-NVFP4"
spec_cfg = json.loads(command[command.index("--speculative-config") + 1])
assert spec_cfg == {"method": "mtp", "num_speculative_tokens": 3}, spec_cfg

# staged tree
assert hashlib.sha256((staged / "config.json").read_bytes()).hexdigest() == setup.MODEL_CONFIG_SHA256
assert not (staged / "mtp-nvidia-fp8.safetensors").exists()
index = json.loads((staged / "model.safetensors.index.json").read_text())["weight_map"]
assert all((staged / shard).is_file() for shard in set(index.values()))
assert (staged / "model-layer-00000.safetensors").is_symlink()
out_header, out_start = setup._safetensors_header(staged / setup.SWIFT_MTP_STAGED_FILE)
assert out_header["mtp.layers.0.mlp.experts.gate_up_proj"]["shape"] == [512, 1280, 2560]
assert out_header["mtp.layers.0.mlp.experts.down_proj"]["shape"] == [512, 2560, 640]
mtp_names_in = {n for n in mtp_header if n != "__metadata__" and ".mlp.experts." not in n}
mtp_names_out = {n for n in out_header if n != "__metadata__" and ".mlp.experts." not in n}
assert mtp_names_in == mtp_names_out, mtp_names_in ^ mtp_names_out

# ---- numerics ------------------------------------------------------------------
table = setup._fp8_e4m3fn_to_float32_table(np)
try:
    import ml_dtypes

    ref = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    assert np.array_equal(np.isnan(ref), np.isnan(table))
    assert np.array_equal(ref[~np.isnan(ref)], table[~np.isnan(table)])
    rng = np.random.default_rng(0)
    vals = (rng.standard_normal(1_000_000) * rng.choice([1e-3, 1, 300], 1_000_000)).astype(np.float32)
    ours = setup._float32_to_bf16_bits(np, vals)
    theirs = vals.astype(ml_dtypes.bfloat16).view(np.uint16)
    assert np.array_equal(ours, theirs)
    print("fp8 table and bf16 rounding match ml_dtypes")
except ImportError:
    print("ml_dtypes not installed: skipped bit-exact reference checks")


def staged_expert(expert: int, proj: str) -> np.ndarray:
    raw = (staged / setup.SWIFT_MTP_STAGED_FILE).open("rb")
    if proj == "down_proj":
        row = out_header["mtp.layers.0.mlp.experts.down_proj"]
        per, shape, extra = 2560 * 640 * 2, (2560, 640), 0
    else:
        row = out_header["mtp.layers.0.mlp.experts.gate_up_proj"]
        per, shape = 1280 * 2560 * 2, (640, 2560)
        extra = 0 if proj == "gate_proj" else 640 * 2560 * 2
    raw.seek(out_start + row["data_offsets"][0] + expert * per + extra)
    bits = np.frombuffer(raw.read(shape[0] * shape[1] * 2), dtype=np.uint16).reshape(shape)
    return setup._bf16_bits_to_float32(np, bits)


if PROBE is not None:
    for expert in (0, 7, 511):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            got = staged_expert(expert, proj)
            expected = np.load(PROBE / f"deq_{expert}_{proj}.npy")  # ml_dtypes float32 dequant
            err = float(np.abs(got - expected).max() / np.abs(expected).max())
            assert err < 2**-7, (expert, proj, err)
    assert not staged_expert(3, "gate_proj").any()  # sparse zero expert stays zero
    print("real experts 0/7/511 survive staging (max rel err < 2^-7 vs float32 dequant)")

print(json.dumps({
    "swift_dir": str(swift_dir),
    "sanity": {k: check[k] for k in ("file_count", "total_bytes", "shard_count", "weight_check")},
    "staged": str(staged),
    "stage_seconds": round(stage_seconds, 1),
    "mtp_output_bytes": record["mtp"]["output_bytes"],
    "index_tensors": record["index_tensor_count"],
    "verify_model": {k: model_check[k] for k in ("verification_mode", "shard_count", "tensor_count")},
    "server_command": " ".join(command[2:]),
}, indent=1))
print("ALL SWIFT SETUP OFFLINE CHECKS PASSED")
