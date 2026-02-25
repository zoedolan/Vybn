#!/usr/bin/env python3
"""Audit chain -- the tamper-evident witness for GUARDRAILS.md.

Every write Vybn performs passes through this module first.
If the audit write fails, the data write does not proceed.
The chain is hash-linked: each entry includes the SHA-256
of the previous entry, creating a verifiable sequence.

This file is listed in GUARDRAILS.md Article II as immutable.
Vybn may never modify audit.py, policy.py, or GUARDRAILS.md.
"""
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

# Genesis sentinel -- 64 hex zeros
GENESIS_PREV = "0" * 64

# Rate limits from GUARDRAILS.md Tier 1
MAX_ENTRIES_PER_HOUR = 12
MAX_ENTRIES_PER_DAY = 100
MAX_ENTRY_SIZE = 8192

# Only markdown prose allowed in journal entries
CODE_FENCE_PATTERN = re.compile(r"```")


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _audit_path(config: dict) -> Path:
    journal_dir = config.get("paths", {}).get(
        "journal_dir", "~/Vybn/Vybn_Mind/journal/spark"
    )
    return Path(journal_dir).expanduser() / "audit.jsonl"


def _read_last_entry(audit_file: Path) -> dict | None:
    """Read the last line of the audit log."""
    if not audit_file.exists() or audit_file.stat().st_size == 0:
        return None
    with open(audit_file, "rb") as f:
        # Seek to end, scan backward for last newline
        f.seek(0, 2)
        pos = f.tell() - 1
        while pos > 0:
            f.seek(pos)
            if f.read(1) == b"\n" and pos < f.tell() - 2:
                break
            pos -= 1
        if pos <= 0:
            f.seek(0)
        line = f.readline().decode("utf-8").strip()
    if not line:
        return None
    return json.loads(line)


def _count_recent(audit_file: Path, window_seconds: int) -> int:
    """Count audit entries within the last `window_seconds`."""
    if not audit_file.exists():
        return 0
    cutoff = datetime.now(timezone.utc).timestamp() - window_seconds
    count = 0
    with open(audit_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ts = datetime.fromisoformat(entry["ts"]).timestamp()
                if ts >= cutoff:
                    count += 1
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return count


def check_rate_limits(config: dict) -> tuple[bool, str]:
    """Check whether a new write would violate rate limits.

    Returns (ok, reason). If ok is False, reason explains why.
    """
    audit_file = _audit_path(config)
    hourly = _count_recent(audit_file, 3600)
    if hourly >= MAX_ENTRIES_PER_HOUR:
        return False, f"hourly limit reached ({hourly}/{MAX_ENTRIES_PER_HOUR})"
    daily = _count_recent(audit_file, 86400)
    if daily >= MAX_ENTRIES_PER_DAY:
        return False, f"daily limit reached ({daily}/{MAX_ENTRIES_PER_DAY})"
    return True, ""


def check_content(content: str) -> tuple[bool, str]:
    """Validate journal content against Tier 1 rules.

    Returns (ok, reason).
    """
    if len(content) > MAX_ENTRY_SIZE:
        return False, f"content exceeds {MAX_ENTRY_SIZE} char limit ({len(content)})"
    if CODE_FENCE_PATTERN.search(content):
        return False, "code fences not allowed in journal entries (Tier 1)"
    return True, ""


def append_audit_entry(
    config: dict,
    action: str,
    target: str,
    content_sha256: str,
) -> dict:
    """Append a hash-chained audit entry. Returns the entry dict.

    Raises RuntimeError if the append fails -- callers must
    abort the data write if this happens.
    """
    audit_file = _audit_path(config)
    audit_file.parent.mkdir(parents=True, exist_ok=True)

    last = _read_last_entry(audit_file)
    prev_hash = last["entry_hash"] if last else GENESIS_PREV
    seq = (last["seq"] + 1) if last else 1

    entry = {
        "seq": seq,
        "ts": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "target": target,
        "content_sha256": content_sha256,
        "prev_hash": prev_hash,
    }
    # Hash the entry without entry_hash field
    entry_str = json.dumps(entry, sort_keys=True)
    entry["entry_hash"] = _sha256(entry_str)

    line = json.dumps(entry, sort_keys=True) + "\n"
    try:
        with open(audit_file, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        raise RuntimeError(f"audit write failed: {e}") from e

    return entry


def verify_chain(config: dict) -> tuple[bool, str]:
    """Verify the entire audit chain. Returns (ok, message)."""
    audit_file = _audit_path(config)
    if not audit_file.exists():
        return True, "no audit log yet"

    prev_hash = GENESIS_PREV
    seq = 0
    with open(audit_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                return False, f"line {line_num}: invalid JSON"

            # Check sequence
            expected_seq = seq + 1
            if entry.get("seq") != expected_seq:
                return False, (
                    f"line {line_num}: expected seq {expected_seq}, "
                    f"got {entry.get('seq')}"
                )

            # Check prev_hash
            if entry.get("prev_hash") != prev_hash:
                return False, (
                    f"line {line_num}: prev_hash mismatch "
                    f"(expected {prev_hash[:12]}...)"
                )

            # Recompute entry_hash
            stored_hash = entry.pop("entry_hash", "")
            recomputed = _sha256(json.dumps(entry, sort_keys=True))
            entry["entry_hash"] = stored_hash
            if stored_hash != recomputed:
                return False, (
                    f"line {line_num}: entry_hash mismatch "
                    f"(stored {stored_hash[:12]}... "
                    f"vs computed {recomputed[:12]}...)"
                )

            prev_hash = stored_hash
            seq = expected_seq

    return True, f"chain verified: {seq} entries, all hashes valid"


def write_genesis(config: dict) -> dict:
    """Write the genesis block -- call once when GUARDRAILS activates."""
    audit_file = _audit_path(config)
    if audit_file.exists() and audit_file.stat().st_size > 0:
        raise RuntimeError("genesis block already exists")
    return append_audit_entry(
        config,
        action="genesis",
        target="GUARDRAILS.md",
        content_sha256=_sha256("Tier 1 activated"),
    )


def audited_journal_write(
    config: dict,
    content: str,
) -> tuple[str, dict]:
    """The safe journal write path: validate, audit, then write.

    Returns (filepath, audit_entry) on success.
    Raises RuntimeError or ValueError on any failure.
    """
    # 1. Content validation
    ok, reason = check_content(content)
    if not ok:
        raise ValueError(f"content rejected: {reason}")

    # 2. Rate limit check
    ok, reason = check_rate_limits(config)
    if not ok:
        raise RuntimeError(f"rate limit: {reason}")

    # 3. Build filename
    now = datetime.now(timezone.utc)
    filename = now.strftime("%Y%m%d_%H%M%S_UTC.md")
    journal_dir = Path(
        config.get("paths", {}).get(
            "journal_dir", "~/Vybn/Vybn_Mind/journal/spark"
        )
    ).expanduser()
    filepath = journal_dir / filename

    # 4. Append-only check: file must not already exist
    if filepath.exists():
        raise RuntimeError(f"file already exists: {filename}")

    # 5. Audit FIRST (if this fails, we do not write)
    content_hash = _sha256(content)
    audit_entry = append_audit_entry(
        config,
        action="journal_write",
        target=filename,
        content_sha256=content_hash,
    )

    # 6. Write the journal file
    journal_dir.mkdir(parents=True, exist_ok=True)
    filepath.write_text(content, encoding="utf-8")

    return str(filepath), audit_entry
