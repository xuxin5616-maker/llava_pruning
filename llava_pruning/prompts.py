"""Keep the original LLaVA MVTec prompt version mapping."""

from __future__ import annotations

from .mvtec_prompts import MVTEC_PROMPT_V0, MVTEC_PROMPT_V1_1, MVTEC_PROMPT_V1, MVTEC_PROMPT_V1_2


PROMPTS = {
    "v0": MVTEC_PROMPT_V0,
    "v1": MVTEC_PROMPT_V1_1,
    "v2": MVTEC_PROMPT_V1,
    "v3": MVTEC_PROMPT_V1_2,
}


def resolve_prompt(version: str, category: str, dataset_text: str | None) -> tuple[str, str]:
    if version not in PROMPTS:
        raise ValueError(f"Unknown prompt version: {version}")
    prompt = PROMPTS[version].get(category)
    if prompt is not None:
        return prompt, f"mvtec_{version}"
    if isinstance(dataset_text, str) and dataset_text.strip():
        return dataset_text.strip(), "dataset_text_fallback"
    raise ValueError(f"No {version} prompt for category {category!r}, and record has no text")
