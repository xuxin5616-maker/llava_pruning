# CLSE reference sources

Repository: https://github.com/zjubinchen/CLSE
Pinned commit: cfc9710abe4a1bd5480e521f3acd86ded4855ff3
Paper: https://arxiv.org/abs/2606.24165v1

`tools.py` and `clse_model.py` are unmodified algorithm reference snapshots from
`LLaVA1.5/llava/model/language_model/`. `LICENSE` is the upstream LLaVA1.5
Apache-2.0 license. `SOURCE.json` records normalized-LF SHA-256 checksums.
Reference modules are not imported by production inference.

Modified adaptation: `vendor/llava/llava/model/language_model/clse.py`.
It uses the pinned high-pass FFT and pure evolution factor, but supplies actual
OneVision Base/AnyRes geometry, channel-chunked FFT, one joint spatial budget
with protected newlines, actual-depth average-rate conversion, and the current
Qwen2 DynamicCache interfaces. It never requests native attention output or
changes the native decoder backend/dtype. The upstream default multi-stage
attention-gated schedule and hard-coded LLaVA-1.5 offsets/grid are not adopted.
