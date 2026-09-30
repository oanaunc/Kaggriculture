# Swift 1.5 model for the ARC3 notebook

This directory lets the notebook serve the Kaggle model
`michaelpoluektov/qwen3-8-swift-nvfp4/Transformers/default/1` in place of
`keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1`. That model is
Swift 1.5 Qwen3.8-Flash-Next with NVFP4 experts, FP8 PLE and NVIDIA's MTP head, about 132.7 GB.

Nothing else changes:

- the vLLM runtime dataset `keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1`;
- the RadixArk PLE patch in `vllm-patches/`;
- every vLLM flag: MTP with 3 speculative tokens, 28 sequences, 32k context, KV cache and memory settings;
- the served model name `Qwen/Qwen3.8-Flash-Next-NVFP4`, so `LOCAL_ANALYZER_MODEL_ID` and the solver need no changes;
- the sampling parameters.

## Files

| File | What it is |
|---|---|
| `serving_setup_swift.py` | Derived from keith's `serving_setup.py` (`/tmp/arc3/src/keith`). The bundle ships it as `serving_setup.py`. |
| `serving_setup_swift.diff` | Unified diff against the original, for review. |
| `swift-staging/` | Goes into the bundle. It holds `staged_config.json` and `staged_hf_quant_config.json`, which are byte-exact RadixArk files (sha `e765…`, `7e69…`). It also holds `SWIFT_SOURCE_FILES.json`, which pins the 196 Swift files, their sizes and the sha256 of the 14 small files. |
| `make_identity.py` | Writes `SOURCE_IDENTITY_swift.json`: keith's identity with the new `serving_setup_sha256` and a Swift `model` block. `model_manifest_sha256` is set to null. |
| `test_swift_setup.py` | Offline test of the non-GPU path. See below. |

## Why a staging step is needed

These findings come from the small files, from safetensors headers read with HTTP range requests, and from the pinned vLLM sources in runtime layer 24.

**What already matches RadixArk:**

- The text/vision config: `qwen4_exp`, 48 layers, 512 experts, `ple_embedding_dtype=float8_e4m3fn`, and one MTP layer.
- Every body tensor name, dtype and shape.
- The NVFP4 experts, including `input_scale`, `weight_scale` and `weight_scale_2`.
- The 128 PLE FP8 shards and their global `weight_scale`.
- The chat template and generation config.

**What does not match:**

- **Quantization config.** Swift's `config.json` uses ModelOpt `MIXED_PRECISION`, which vLLM maps to `modelopt_mixed`.
  - The pinned vLLM PLE loader selects FP8 only for `Fp8Config`, or for the exact RadixArk config through the patch.
  - Loading Swift's config as is would give a bf16 PLE of about 102 GB with the FP8 global scale silently dropped.
  - Our server command also passes `--quantization modelopt_fp4`, which conflicts with `modelopt_mixed`.
- **MTP experts.** The MTP routed experts are NVIDIA `FP8_PB_WO` (128x128 block FP8 with `weight_scale_inv`). The pinned `modelopt_mixed` config has no block-FP8 MoE path, and its MTP prefix (`mtp.layers.48`) does not match `mtp.layers.0`. The experts would be silently loaded as unscaled bf16, and MTP acceptance would collapse.

**What the setup does.** `stage_swift_model` runs in a thread while the runtime layers extract. It builds `/tmp/qwen38-swift-staged-model/`:

- Symlinks to every Swift shard and tokenizer file.
- The RadixArk `config.json` and `hf_quant_config.json`. The pinned PLE gate, the config-hash env var and `--quantization modelopt_fp4` therefore work unchanged.
- A rewritten `mtp-swift-bf16.safetensors`:
  - The 31 non-expert MTP tensors are copied byte for byte.
  - The FP8 experts are dequantized to bf16 in RadixArk's fused layout: `gate_up_proj [512,1280,2560]` and `down_proj [512,2560,640]`.
  - The result is 5,214,305,456 bytes. Swift's original bf16 MTP was 5,214,301,696 bytes.
- A new index. Its tensor-name set is identical to the RadixArk index: 296,475 names.

**How this was validated.** For experts 0, 7 and 511, the dequantized NVIDIA experts match RadixArk's bf16 MTP experts with cosine 0.9996, which is the expected FP8 error. This confirms the scale semantics and the gate/up order.

**What was removed.** The RadixArk `MODEL_MANIFEST.json` checks. In their place, `verify_swift_source` runs in both fast and full mode and takes under 1 s. It checks:

- all 196 file sizes, by stat only; the weights are not hashed;
- the sha256 of the 14 small files;
- the config identity;
- the export's source repo.

## Build

```bash
cd /home/user/Kaggriculture
python arc3/serving/make_identity.py      # only after editing serving_setup_swift.py
ARC3_MODEL=swift ARC3_DATASET_SLUG=arc3-duck-ours-swift-src ARC3_KERNEL_SLUG=arc3-duck-ours-swift \
  ARC3_QUICK_COMMIT=1 python arc3/tools/build_kaggle.py v6-swift
```

For the full 25-game run, drop `ARC3_QUICK_COMMIT`.

`ARC3_MODEL=swift` makes these changes to the bundle and kernel:

- copies `serving_setup_swift.py` to `dataset/serving_setup.py`;
- adds `swift-staging/`;
- re-derives `SOURCE_IDENTITY.json` from the copied file;
- sets `model_sources` to `["michaelpoluektov/qwen3-8-swift-nvfp4/Transformers/default/1"]`.

The default, `ARC3_MODEL=radixark`, produces exactly the old build.

## Offline test

```bash
python test_swift_setup.py /tmp/swift <work_dir> /tmp/swift/probe
```

The test builds a fake mount at `/kaggle/input/models/michaelpoluektov/qwen3-8-swift-nvfp4/transformers/default/1`. The mount holds the real small files, sparse shards of the exact sizes, and the real MTP header with real bytes for experts 0, 7 and 511. The test then runs these steps:

- identity check;
- model discovery;
- sanity check;
- staging;
- the staged-model gate;
- `server_command`.

It also checks the fp8 table and bf16 rounding bit-exactly against `ml_dtypes`.

Last run: all checks passed and staging took 57 s locally.

## What to look for in the notebook log

| Log line or file | Meaning |
|---|---|
| `SWIFT_MODEL_PATH …`, `SWIFT_MODEL_SANITY files=196 shards=179 …` | The model was found and passed the sanity check. |
| `SWIFT_MTP_STAGED … bytes=5214305456 elapsed_s=…` and `SWIFT_STAGED_MODEL …` | Staging finished. |
| `PINNED_MODEL_PATH /tmp/qwen38-swift-staged-model` | vLLM is serving the staged tree. |
| `VLLM_SETUP_COMPLETE {...}` | The server is up. |
| `VLLM_SETUP_FAILED {...}` | The setup failed. `/kaggle/working/vllm-setup-failure.json` holds the error and a 400-line server-log tail; the full log is in `/kaggle/working/vllm-openai-server.log`. |
| `vllm-metrics-final.prom` | MTP acceptance, recorded at teardown. |

Possible GPU-side failures and how each one shows up:

1. **Model not attached, or a different mount path.**
   - The setup falls back to `rglob('SWIFT_FP8_EXPORT.json')` under `/kaggle/input`.
   - If that also fails, the error is `FileNotFoundError: Expected mounted Kaggle model … Attach it as a notebook model source.`
2. **Disk.** The storage gate now asks for 36.5 GB plus 5.4 GB free on `/tmp`.
   - Failure: `The filesystem for … has too little free space: X < 41907222016`.
   - Failure at staging time: `Too little space … to stage the Swift MTP head`.
3. **Slow staging.** Reading 2.7 GB from the model mount, plus numpy dequantization, plus writing 5.2 GB takes about 60 s here. It overlaps with runtime extraction and counts against the 1800 s setup alarm.
   - Failure: `TimeoutError: The 1800s setup/preflight ceiling expired`, or a vLLM readiness timeout.
4. **vLLM load errors.** The staged names and shapes equal RadixArk's, so this is not expected. It would show up in the server log tail as, for example, `Weight files referenced in index but missing`, a shape mismatch on `mtp.layers.0.mlp.experts.*`, or `Layer … has no parameter`.
5. **Silent quality problems.** These produce no error, so you only see them in the scores and metrics.
   - The MTP head is NVIDIA's copy of the *base* Qwen MTP, the same as RadixArk's, and was not trained on Swift. The spec-decode acceptance rate may be lower than with RadixArk.
   - Compare `vllm:spec_decode_num_accepted_tokens_total` against `vllm:spec_decode_num_draft_tokens_total` in `vllm-metrics-final.prom`. If acceptance is poor, `TAAF_VLLM_MTP_TOKENS=1` or `2` is the tuning knob.
   - PLE garbage would show up as nonsense text and near-zero scores. It should not happen, because the staged config is byte-identical to RadixArk's and the pinned PLE gate therefore applies.
6. **Sampling.** Sampling stays at temperature 0.6, top_p 0.95, top_k 20. The Swift model card evaluates at temperature 1.0 with top_p 0.95 and top_k 20. That is an open tuning question, not a load risk.

## Licence (Swift Open License v1.0 + Qwen Community License 1.0)

**Swift contribution.** It is under the Swift Open License v1.0, which is Apache-2.0-derived text with extra terms:

- **§5:** commercial use is licensed only for entities with less than US$1M gross annual revenue, counting affiliates. Anyone above that needs a paid "Swift Enterprise License".
- **§12:** automatic termination on any non-compliance.
- **§4(a)–(e):** redistribution must include the licence, NOTICE and the Qwen licence.

**Base model.** It is under the Qwen Community License 1.0:

- Model-as-a-Service and AI Work Assistant businesses need a separate licence for commercial use.
- Very large products must display the model name.

This base licence already applies to the RadixArk checkpoint we use today.

**Competition rules.** Section 5 (Winner License) of `/tmp/arc3_pages.txt`:

- **5.a.1:** requires a CC-BY 4.0 licence of the Submission. It also requires an "open source model, and open source weights/parameters as defined in the OSI Open Source AI Definition checklist".
- **5.a.3:** exempts pretrained models "with an incompatible license" from the obligation to *grant* an open-source licence.
- **General rules:** external models must be "reasonably accessible" and of "minimal cost".

**Verdict.**

- **Participation is OK.** We can use the model and appear on the leaderboard. We are an individual or small team below the $1M cap, the Kaggle copy is free and public, and use for evaluation or research is allowed.
- **Prize eligibility is at risk.** A revenue-conditioned commercial restriction fails OSD #5 and #6 (no discrimination against persons or fields of endeavour). Swift weights are therefore *not* OSI or OSAID-open, and neither are the Qwen Community License weights.
  - Whether the 5.a.3 carve-out overrides the 5.a.1 "open-source model/weights" requirement is the host's call.
  - The Qwen base licence carries the same class of risk. Swift adds a stricter layer: the revenue cap, termination, and a Sponsor above $1M revenue needing an enterprise licence for commercial use.
- **Provenance issue.** The Kaggle mirror's `LICENSE` file is the *Qwen* licence text, while its README declares `swift-open-license-1.0`. Strictly, §4(a) requires shipping the Swift licence, so the mirror may itself be non-compliant.
- **Recommendation:**
  - Keep a RadixArk submission as a fallback.
  - Get a host ruling on the forum before relying on Swift for a prize-eligible final pick.
  - If we win with it, credit UkisAI/Qwen as §8 and the NOTICE require.
