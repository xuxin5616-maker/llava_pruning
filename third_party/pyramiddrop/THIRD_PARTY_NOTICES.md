# PyramidDrop source reference

Repository: https://github.com/Cooperx521/PyramidDrop

Pinned commit: `6444f304aee4b7edcb022839fb9e8cd07859704a`.

`modeling_llama_pdrop.py` is an unmodified reference snapshot and is not imported at runtime. Its canonical LF SHA-256 is recorded in `SOURCE.json`. The upstream repository's MIT license is included in `LICENSE`; copyright belongs to Xing Long (2024). The source file also retains its original LLaMA/Hugging Face Apache-2.0 notices.

The OneVision/Qwen2 adaptation lives in `vendor/llava/llava/model/language_model/pyramiddrop.py`. It retains the upstream next-layer norm/QK ranking, RoPE, full-key softmax followed by head averaging, sorted top-k, floor cumulative budgets, and compacted position IDs. It adds Qwen2 GQA, Transformers 4.46.1 per-layer DynamicCache handling, inference constraints, and experiment metadata. It does not reuse the upstream training loop or claim the upstream benchmark results.
