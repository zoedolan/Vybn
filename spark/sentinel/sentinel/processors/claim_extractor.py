"""Claim extraction via local M2.5.

Decomposes raw news into: (factual_kernel, interpretive_frame, excitement_level)

Temperance: excitement is MULTIPLIED by the temperance_factor.
A factor of 0.6 means "keep 60% of the original excitement" --
i.e., discount interpretive framing by 40%.
"""
import json
import re
from pathlib import Path
from typing import Any

EXTRACTION_PROMPT = """You are an intelligence analyst. Extract only substantive, factual claims from this text.

IGNORE:
- Image descriptions, captions, or alt text
- Layout descriptions ("a logo on the right", "blue background")
- Purely promotional statements with no factual content
- Repeated information (extract each fact only once)

For each genuine claim, produce a JSON object:
{{"kernel": "<bare fact, stripped of spin>",
  "frame": "<author's interpretation or editorial angle, null if neutral>",
  "excitement": <0.0-1.0, how novel/significant is this claim? 0=mundane, 0.5=noteworthy, 0.8=major, 1.0=paradigm-shifting>,
  "category": "<ai|politics|crypto|geopolitics|science|other>"}}

Return ONLY a JSON array. No prose, no markdown fences, no commentary.
If the text contains no substantive claims, return [].

TEXT:
{text}
"""


def extract_claims(text: str, source_label: str,
                   temperance_factor: float, model_fn: Any) -> list[dict]:
    # Skip very short texts (likely just image descriptions)
    if len(text.strip()) < 50:
        return []

    raw = model_fn(EXTRACTION_PROMPT.format(text=text[:8000]))
    try:
        claims = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        claims = json.loads(match.group()) if match else []

    if not isinstance(claims, list):
        claims = []

    valid = []
    for c in claims:
        if not isinstance(c, dict):
            continue
        kernel = c.get("kernel", "").strip()
        # Skip very short kernels or image-description-like content
        if len(kernel) < 15:
            continue
        if any(kw in kernel.lower() for kw in [
            "image shows", "image displays", "logo on", "background is",
            "illustration of", "photo of", "screenshot of"
        ]):
            continue
        c["source"] = source_label
        c["effective_excitement"] = min(1.0,
            c.get("excitement", 0.5) * temperance_factor)
        c["market_corroboration"] = 0.0
        valid.append(c)
    return valid


def process_news_bundle(bundle_path: str | Path, config: dict,
                        model_fn: Any) -> list[dict]:
    bundle = json.loads(Path(bundle_path).read_text())

    # Build temperance map from all source types
    temperance_map: dict[str, float] = {}
    ns = config.get("news_sources", {})
    for tw in ns.get("twitter", []):
        temperance_map[tw.get("label", tw["handle"])] = tw.get("temperance_factor", 1.0)
    for sub in ns.get("substacks", []):
        temperance_map[sub.get("label", "")] = sub.get("temperance_factor", 1.0)
    for blog in ns.get("lab_blogs", []):
        temperance_map[blog.get("label", "")] = blog.get("temperance_factor", 1.0)

    all_claims: list[dict] = []
    for item in bundle:
        title = item.get('title', '').strip()
        summary = item.get('summary', '').strip()
        # Combine title + summary but skip if it's just HTML/images
        text = f"{title}\n{summary}" if summary else title
        # Strip HTML tags for cleaner extraction
        text = re.sub(r'<[^>]+>', ' ', text).strip()
        text = re.sub(r'\s+', ' ', text)

        source = item.get("source", "unknown")
        all_claims.extend(extract_claims(
            text, source, temperance_map.get(source, 1.0), model_fn))
    return all_claims
