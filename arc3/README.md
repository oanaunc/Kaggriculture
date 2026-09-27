# ARC-AGI-3 work

`upstream/` is an unmodified snapshot of the public "Duck" solver (Tufa Labs,
MIT-licensed, taken from Kaggle dataset `keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1`).
Our modifications go on top of this baseline so every change is diffable.
The vendored sglang build (68 MB) is not copied; it is pulled from the dataset at run time.
