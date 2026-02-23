"""Sentinel Digest: the quarantine wall.

This script reads from the sentinel data sandbox (data/),
validates every claim against required fields, category constraints,
confidence bounds, and length limits, then writes a single
data/latest_digest.json that is the ONLY file the main Vybn
context should ever read.

Usage:
    python sentinel_digest.py [--data-dir data/]
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("sentinel_digest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

VALID_CATEGORIES = {"ai", "politics", "crypto", "geopolitics", "science", "other"}
REQUIRED_CLAIM_FIELDS = {"kernel", "source", "effective_excitement", "confidence"}
MAX_KERNEL_LENGTH = 500
MAX_DIGEST_CLAIMS = 20


def validate_claim(claim: dict) -> tuple[bool, str]:
    """Validate a single claim. Returns (valid, reason)."""
    if not isinstance(claim, dict):
        return False, "not a dict"
    missing = REQUIRED_CLAIM_FIELDS - set(claim.keys())
    if missing:
        return False, f"missing fields: {missing}"
    cat = claim.get("category", "")
    if cat and cat not in VALID_CATEGORIES:
        return False, f"invalid category: {cat}"
    conf = claim.get("confidence", -1)
    if not (0.0 <= conf <= 1.0):
        return False, f"confidence out of bounds: {conf}"
    exc = claim.get("effective_excitement", -1)
    if not (0.0 <= exc <= 1.0):
        return False, f"excitement out of bounds: {exc}"
    kernel = claim.get("kernel", "")
    if len(kernel) > MAX_KERNEL_LENGTH:
        return False, f"kernel too long: {len(kernel)}"
    if not kernel.strip():
        return False, "empty kernel"
    return True, "ok"


def load_latest_claims(data_dir: Path) -> list[dict]:
    """Load claims from the most recent claims file."""
    structured = data_dir / "structured"
    if not structured.exists():
        return []
    files = sorted(structured.glob("claims_*.json"), reverse=True)
    if not files:
        return []
    try:
        return json.loads(files[0].read_text())
    except (json.JSONDecodeError, OSError) as e:
        log.error(f"Failed to read {files[0]}: {e}")
        return []


def load_belief_state(data_dir: Path) -> dict | None:
    path = data_dir / "sentinel_state.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return None


def produce_digest(data_dir: Path) -> dict:
    """Read from sandbox, validate, produce clean digest."""
    claims = load_latest_claims(data_dir)
    belief = load_belief_state(data_dir)

    validated = []
    rejected = 0
    for claim in claims:
        valid, reason = validate_claim(claim)
        if valid:
            validated.append({
                "kernel": claim["kernel"][:MAX_KERNEL_LENGTH],
                "category": claim.get("category", "other"),
                "confidence": round(claim["confidence"], 3),
                "effective_excitement": round(claim["effective_excitement"], 3),
                "market_corroboration": round(claim.get("market_corroboration", 0.0), 3),
                "source": claim["source"],
            })
        else:
            rejected += 1
            log.debug(f"Rejected claim: {reason}")

    # Sort by confidence descending, take top N
    validated.sort(key=lambda c: c["confidence"], reverse=True)
    validated = validated[:MAX_DIGEST_CLAIMS]

    digest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "claims_validated": len(validated),
        "claims_rejected": rejected,
        "top_claims": validated,
    }

    if belief:
        digest["trajectory_assessment"] = belief.get("trajectory_assessment")
        digest["last_synthesis"] = belief.get("last_updated")
        # Include latest digest text if available
        history = belief.get("digest_history", [])
        if history:
            digest["latest_synthesis_text"] = history[-1].get("digest", "")[:1000]

    return digest


def main():
    parser = argparse.ArgumentParser(description="Sentinel digest generator")
    parser.add_argument("--data-dir", default="data",
                        help="Path to sentinel data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        log.error(f"Data directory {data_dir} does not exist")
        return

    digest = produce_digest(data_dir)

    output = data_dir / "latest_digest.json"
    output.write_text(json.dumps(digest, indent=2))
    log.info(f"Digest written: {digest['claims_validated']} claims, "
             f"{digest['claims_rejected']} rejected")


if __name__ == "__main__":
    main()
