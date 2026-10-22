# PriorTR reference sources

Author repository: https://github.com/CodeChildCZJ/PriorTR

Pinned commit: e442a4c14af1b333067a6b8310c9111cad922033.
The original files are under `image/LLaVA/llava/vtr/{model,strategy}/`, except
`config.py` directly in `vtr/`. The root Apache-2.0 license is included.
These unmodified snapshots are reference material, not production imports.

Our modified OneVision/Qwen2 inference adaptation is
`vendor/llava/llava/model/language_model/priortr.py`. It independently recomputes
causal Q/K rows with Qwen2 GQA/RoPE instead of requesting native attention
outputs. It keeps the existing FP16/FlashAttention2 decoder, uses actual packed
spans, preserves sparse original position IDs, and compacts all existing KV
caches. It supports one boundary and percentage budgets; original zero-token
budgets receive a recorded minimum-one-token safety extension. FP16 score
projections can differ numerically from the source's full-matrix computation.

The pinned source uses epsilon=1e-10 and defaults to last-query aggregation for
LLaVA-1.6, while the paper's Appendix A.3 describes question aggregation and
epsilon=1e-6. Both aggregation modes and epsilon are explicit in our config.
This is an adaptation, not a reproduction claim for the author's benchmarks.
