"""Claim extraction via local M2.5.

Decomposes raw news into: (factual_kernel, interpretive_frame, excitement_level)
"""
import json
import re
from pathlib import Path
from typing import Any

EXTRACTION_PROMPT = """Extract each distinct claim as a JSON array. Per claim:
{{"kernel": "<bare fact, no editorializing>",
 "frame": "<author's interpretive spin, if any>",
 "excitement": <0.0-1.0 how hyperbolic>,
 "category": "<ai|politics|crypto|geopolitics|science|other>"}}

JSON array only. No commentary.

TEXT:
{text}
"""


def extract_claims(text: str, source_label: str,
                   temperance_factor: float, model_fn: Any) -> list[dict]:
    raw = model_fn(EXTRACTION_PROMPT.format(text=text[:8000]))
    try:
        claims = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        claims = json.loads(match.group()) if match else []
    for c in claims:
        c["source"] = source_label
        c["effective_excitement"] = min(1.0,
            c.get("excitement", 0.5) / max(temperance_factor, 0.1))
    return claims


def process_news_bundle(bundle_path: str | Path, config: dict,
                        model_fn: Any) -> list[dict]:
    bundle = json.loads(Path(bundle_path).read_text())
    temperance_map: dict[str, float] = {}
    for tw in config.get("news_sources", {}).get("twitter", []):
        temperance_map[tw.get("label", tw["handle"])] = tw.get("temperance_factor", 1.0)
    for sub in config.get("news_sources", {}).get("substacks", []):
        temperance_map[sub.get("label", "")] = sub.get("temperance_factor", 1.0)
    all_claims: list[dict] = []
    for item in bundle:
        text = f"{item.get('title', '')}\n{item.get('summary', '')}"
        source = item.get("source", "unknown")
        all_claims.extend(extract_claims(
            text, source, temperance_map.get(source, 1.0), model_fn))
    return all_claims
