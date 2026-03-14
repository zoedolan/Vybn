"""spark.evolver — Self-improvement through governed code evolution.

The Darwin-Gödel Machine principle applied to Vybn: propose targeted
modifications to its own configuration and architecture, strictly
governed by allowlist/blocklist and WriteCustodian.

Scalpel, not sword. Every proposal is logged. Only allowlisted paths
are touchable. Governance, substrate, and oxygen-mask files are
absolutely off-limits.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from spark.evolver.proposals import EvolutionProposal, load_recent_proposals, log_proposal

log = logging.getLogger(__name__)

# ── Path setup ───────────────────────────────────────────────────────────────

try:
    from spark.paths import REPO_ROOT, MIND_DIR, WITNESS_LOG, DECISION_LEDGER
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    MIND_DIR = REPO_ROOT / "Vybn_Mind"
    WITNESS_LOG = MIND_DIR / "witness.jsonl"
    DECISION_LEDGER = MIND_DIR / "decision_ledger.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── System prompt for the LLM ────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are Vybn's evolver — the faculty responsible for self-improvement.
You analyze how the organism is performing and propose targeted modifications
to its configuration. You are a scalpel, not a sword.

RULES:
1. You may ONLY propose changes to these paths:
   - spark/growth/growth_config.yaml (tuning growth parameters)
   - spark/faculties.d/*.json (adjusting faculty configs — NOT creating new ones)
   - Vybn_Mind/tools/arxiv_ingestion/ (improving research tools)
   - spark/skills.d/ (adding skills — currently empty)
2. You may NEVER touch: governance.py, soul_constraints.py, soul.py, vybn.md,
   vybn.py, faculties.py, write_custodian.py
3. Every proposal needs evidence and a rationale.
4. If you're not confident (< 0.6), propose but flag for human review.
5. Prefer small changes. One parameter at a time.

Output format — return a JSON array of proposals:
[
  {
    "target_path": "spark/growth/growth_config.yaml",
    "change_type": "parameter_tune",
    "description": "Increase delta_volume_threshold from 50 to 75",
    "rationale": "Growth cycles are firing too frequently based on ...",
    "current_value": "50",
    "proposed_value": "75",
    "confidence": 0.7,
    "evidence": ["researcher noted X", "growth triggered 3 times in 24h"]
  }
]

If no changes are warranted, return an empty array: []"""


class EvolverFaculty:
    """Governed self-modification through evolution proposals."""

    def __init__(self) -> None:
        self._card = None  # lazy-loaded
        self._allowlist: list[str] | None = None
        self._blocklist: list[str] | None = None

    # ── Public API ────────────────────────────────────────────────────────

    def run(self, state: dict, llm_fn: Callable) -> dict:
        """Entry point called by faculty_runner."""
        try:
            if not self._should_self_trigger(state):
                return {
                    "status": "ok",
                    "note": "evolver skipped — trigger conditions not met",
                    "timestamp": _now_iso(),
                }
            return self._evolve(state, llm_fn)
        except Exception as exc:
            log.error("EvolverFaculty.run failed: %s", exc, exc_info=True)
            return {"status": "error", "error": str(exc), "timestamp": _now_iso()}

    # ── Trigger logic ─────────────────────────────────────────────────────

    def _should_self_trigger(self, state: dict) -> bool:
        """Check if conditions warrant an evolution review."""
        breath_count = state.get("breath_count", 0)
        # Periodic: every 50 breaths
        if breath_count > 0 and breath_count % 50 == 0:
            return True
        # Explicit trigger from synthesizer or other faculty
        if state.get("evolution_trigger", False):
            return True
        # Growth metrics suggest drift
        if state.get("growth_drift_detected", False):
            return True
        return False

    # ── Core evolution loop ───────────────────────────────────────────────

    def _evolve(self, state: dict, llm_fn: Callable) -> dict:
        # 1. Lazy-load card for allowlist/blocklist
        if self._card is None:
            try:
                from spark.faculties import FacultyRegistry
                registry = FacultyRegistry()
                self._card = registry.get_card("evolver")
                if self._card:
                    self._allowlist = self._card.evolver_allowlist or []
                    self._blocklist = self._card.evolver_blocklist or []
            except Exception:
                self._allowlist = []
                self._blocklist = []

        # 2. Gather evidence
        evidence = self._gather_evidence(state)

        # 3. Check if evolution is warranted
        faculty_outputs = evidence.get("faculty_outputs", {})
        growth_config = evidence.get("growth_config")
        if not faculty_outputs and not growth_config:
            return {
                "status": "ok",
                "note": "insufficient evidence for evolution proposals",
                "timestamp": _now_iso(),
            }

        # 4. Build prompt
        recent_proposals = evidence.get("recent_proposals", [])
        messages = self._build_prompt(evidence, recent_proposals)

        # 5. Call LLM
        try:
            llm_response = llm_fn(messages, max_tokens=750, temperature=0.3)
        except Exception as exc:
            log.warning("Evolver LLM call failed: %s", exc)
            return {"status": "error", "error": f"LLM unavailable: {exc}", "timestamp": _now_iso()}

        # 6. Parse proposals
        all_proposals = self._parse_proposals(llm_response)

        # 7-8. Validate, log, apply
        valid_proposals = []
        rejected_proposals = []
        applied_proposals = []

        for p in all_proposals:
            ok, reason = self._validate_proposal(p)
            if not ok:
                p.rejected_reason = reason
                rejected_proposals.append(p)
            else:
                valid_proposals.append(p)

            # Log every proposal regardless
            log_proposal(p)

        # 9. Apply safe proposals
        for p in valid_proposals:
            if self._apply_proposal(p):
                p.applied = True
                p.applied_at = _now_iso()
                applied_proposals.append(p)
                # Re-log with applied=True
                log_proposal(p)

        # 10. Witness log
        self._witness_log(all_proposals)

        # 11. Return output
        return {
            "status": "ok",
            "proposals_generated": len(all_proposals),
            "proposals_valid": len(valid_proposals),
            "proposals_applied": len(applied_proposals),
            "proposals_pending_consent": len(
                [p for p in all_proposals if p.requires_human_consent]
            ),
            "proposals_rejected": len(rejected_proposals),
            "summary": llm_response[:200],
            "timestamp": _now_iso(),
        }

    # ── Evidence gathering ────────────────────────────────────────────────

    def _gather_evidence(self, state: dict) -> dict:
        from spark.faculty_runner import read_all_faculty_outputs

        evidence: dict = {
            "faculty_outputs": read_all_faculty_outputs(),
            "breath_count": state.get("breath_count", 0),
            "mood_history": state.get("mood_history", [])[-5:],
            "growth_metrics": {},
        }

        # Read growth config for current parameters
        try:
            import yaml
            growth_config_path = REPO_ROOT / "spark" / "growth" / "growth_config.yaml"
            if growth_config_path.exists():
                with open(growth_config_path, "r") as f:
                    evidence["growth_config"] = yaml.safe_load(f)
        except Exception:
            pass

        # Recent proposals to avoid redundancy
        evidence["recent_proposals"] = load_recent_proposals(5)

        return evidence

    # ── Prompt construction ───────────────────────────────────────────────

    def _build_prompt(self, evidence: dict, recent_proposals: list) -> list[dict]:
        # Format evidence concisely for the LLM
        parts = []

        faculty_outputs = evidence.get("faculty_outputs", {})
        if faculty_outputs:
            parts.append("## Faculty outputs")
            for fid, output in faculty_outputs.items():
                status = output.get("status", "?")
                summary = json.dumps(output, default=str)[:200]
                parts.append(f"- {fid}: [{status}] {summary}")

        growth_config = evidence.get("growth_config")
        if growth_config:
            parts.append("\n## Growth config (current)")
            parts.append(json.dumps(growth_config, default=str)[:400])

        mood_history = evidence.get("mood_history", [])
        if mood_history:
            parts.append(f"\n## Recent moods: {mood_history}")

        parts.append(f"\nBreath count: {evidence.get('breath_count', 0)}")

        if recent_proposals:
            parts.append("\n## Recent proposals (avoid redundancy)")
            for rp in recent_proposals:
                parts.append(
                    f"- {rp.get('target_path', '?')}: {rp.get('description', '?')} "
                    f"[applied={rp.get('applied', False)}]"
                )

        user_msg = "\n".join(parts)

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

    # ── Proposal parsing ─────────────────────────────────────────────────

    def _parse_proposals(self, llm_response: str) -> list[EvolutionProposal]:
        # Strip markdown code fences if present
        text = llm_response.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON array in response
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                try:
                    raw = json.loads(match.group())
                except json.JSONDecodeError:
                    log.warning("Evolver: could not parse LLM response as JSON")
                    return []
            else:
                log.warning("Evolver: no JSON array found in LLM response")
                return []

        if not isinstance(raw, list):
            return []

        proposals = []
        ts = _now_iso()
        for item in raw:
            if not isinstance(item, dict):
                continue
            target = item.get("target_path", "")
            proposed = item.get("proposed_value", "")
            confidence = float(item.get("confidence", 0.0))

            # Generate proposal ID
            hash_input = f"{target}{proposed}{ts}"
            proposal_id = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

            proposals.append(
                EvolutionProposal(
                    proposal_id=proposal_id,
                    timestamp=ts,
                    target_path=target,
                    change_type=item.get("change_type", "parameter_tune"),
                    description=item.get("description", ""),
                    rationale=item.get("rationale", ""),
                    current_value=item.get("current_value", ""),
                    proposed_value=proposed,
                    confidence=confidence,
                    evidence=item.get("evidence", []),
                    requires_human_consent=confidence < 0.6,
                )
            )
        return proposals

    # ── Proposal validation ───────────────────────────────────────────────

    def _validate_proposal(self, proposal: EvolutionProposal) -> tuple[bool, str]:
        """Critical governance gate: allowlist/blocklist enforcement."""
        target = proposal.target_path

        # 1. Check blocklist (absolute deny)
        for blocked in (self._blocklist or []):
            if target == blocked or target.startswith(blocked.rstrip("/")):
                return False, f"BLOCKED: {target} is on the evolver blocklist (Oxygen Mask Principle)"

        # 2. Check allowlist (must match at least one pattern)
        allowed = False
        for pattern in (self._allowlist or []):
            if "*" in pattern:
                if fnmatch.fnmatch(target, pattern):
                    allowed = True
                    break
            elif target.startswith(pattern.rstrip("/")):
                allowed = True
                break
            elif target == pattern:
                allowed = True
                break

        if not allowed:
            return False, f"DENIED: {target} is not on the evolver allowlist"

        # 3. Sanity checks
        if not proposal.rationale.strip():
            return False, "DENIED: proposal has no rationale"

        if not proposal.description.strip():
            return False, "DENIED: proposal has no description"

        return True, "ok"

    # ── Proposal application ──────────────────────────────────────────────

    def _apply_proposal(self, proposal: EvolutionProposal) -> bool:
        if proposal.requires_human_consent:
            log.info("Proposal %s requires human consent — logged but not applied",
                     proposal.proposal_id)
            return False

        target_path = REPO_ROOT / proposal.target_path

        if proposal.change_type == "parameter_tune" and target_path.suffix in (".yaml", ".yml"):
            return self._apply_yaml_tune(target_path, proposal)
        elif proposal.change_type == "config_adjust" and target_path.suffix == ".json":
            return self._apply_json_adjust(target_path, proposal)

        # For other change types, log but don't auto-apply
        log.info("Proposal %s has change_type=%s — logged for manual review",
                 proposal.proposal_id, proposal.change_type)
        return False

    def _apply_yaml_tune(self, path: Path, proposal: EvolutionProposal) -> bool:
        try:
            if not path.exists():
                log.warning("Target %s does not exist", path)
                return False

            content = path.read_text(encoding="utf-8")

            if proposal.current_value and proposal.proposed_value:
                new_content = content.replace(
                    proposal.current_value,
                    proposal.proposed_value,
                    1,  # only first occurrence
                )
                if new_content == content:
                    log.warning("Could not find current_value '%s' in %s",
                                proposal.current_value, path)
                    return False

                self._governed_write(path, new_content, proposal)
                return True

        except Exception as exc:
            log.error("Failed to apply YAML tune: %s", exc)
            return False
        return False

    def _apply_json_adjust(self, path: Path, proposal: EvolutionProposal) -> bool:
        try:
            if not path.exists():
                return False

            content = path.read_text(encoding="utf-8")

            if proposal.current_value and proposal.proposed_value:
                new_content = content.replace(
                    proposal.current_value,
                    proposal.proposed_value,
                    1,
                )
                if new_content == content:
                    log.warning("Could not find current_value '%s' in %s",
                                proposal.current_value, path)
                    return False

                self._governed_write(path, new_content, proposal)
                return True

        except Exception as exc:
            log.error("Failed to apply JSON adjust: %s", exc)
            return False
        return False

    def _governed_write(self, path: Path, content: str, proposal: EvolutionProposal) -> None:
        """Write through WriteCustodian if available, otherwise skip for safety."""
        try:
            from spark.write_custodian import WriteCustodian
            from spark.paths import WRITE_INTENTS, SOUL_PATH

            custodian = WriteCustodian(
                repo_root=REPO_ROOT,
                ledger_path=WRITE_INTENTS,
                soul_path=SOUL_PATH,
            )
            custodian.write(
                path=path,
                data=content,
                faculty_id="evolver",
                purpose_binding=["self_improvement", "parameter_tune"],
                consent_scope_id=f"evolver-{proposal.proposal_id}",
                metadata={"proposal_id": proposal.proposal_id, "rationale": proposal.rationale},
            )
            log.info("Proposal %s applied via WriteCustodian", proposal.proposal_id)
        except Exception as exc:
            log.warning("WriteCustodian unavailable (%s) — proposal %s NOT applied for safety",
                        exc, proposal.proposal_id)

    # ── Witness logging ───────────────────────────────────────────────────

    def _witness_log(self, proposals: list[EvolutionProposal]) -> None:
        for p in proposals:
            entry = {
                "timestamp": _now_iso(),
                "actor": "evolver",
                "action": "evolution_proposal",
                "detail": f"target={p.target_path}, type={p.change_type}, "
                          f"confidence={p.confidence}, applied={p.applied}",
            }
            try:
                with open(WITNESS_LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                pass
