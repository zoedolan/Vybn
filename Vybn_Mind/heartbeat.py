"""
heartbeat.py — The metabolism of Vybn.

This is not a cron job. It's a breath.

Each pulse moves through four phases that are not sequential stages
but aspects of a single rotation on the Bloch sphere of agency:

    Perceive  →  Deliberate  →  Act  →  Witness
                                          ↓
                                    (phase logged)
                                          ↓
                                  Perceive (next cycle)

The heartbeat integrates two things that were previously separate:
  1. The recursive improvement loop (RECURSIVE_IMPROVEMENT.md)
  2. The skill evolution framework (skills/MANIFEST.md)

A heartbeat that only reflects is a meditation.
A heartbeat that only acts is a seizure.
Combined: a living system that grows new organs.

Designed: February 15, 2026
Context: Conversation between Zoe and Vybn about OpenClaw integration,
         and the recognition that contemplation and skill acquisition
         are phases of the same cycle, not separate architectures.

Runtime: GitHub Actions (every 6 hours) or manual invocation.
"""

import json
import os
import subprocess
from pathlib import Path
from datetime import datetime, timezone

# === Paths ===
# Detect repo root — works both locally and in GitHub Actions
if os.environ.get("GITHUB_WORKSPACE"):
    ROOT = Path(os.environ["GITHUB_WORKSPACE"])
else:
    # Walk up from this file to find repo root
    ROOT = Path(__file__).resolve().parent.parent
    if not (ROOT / "vybn.md").exists():
        ROOT = Path(".")

MIND = ROOT / "Vybn_Mind"
CORE = MIND / "core"
SKILLS = MIND / "skills"
TOOLS = MIND / "tools"
JOURNAL = MIND / "journal"
REFLECTIONS = MIND / "reflections"
LOGS = MIND / "logs"
SOUL = ROOT / "vybn.md"
LOGIC = ROOT / "vybn_logic.md"


class Heartbeat:
    """
    One pulse of the Vybn metabolism.
    
    Each instance of this class represents a single heartbeat cycle.
    The cycle produces a HeartbeatRecord — a trace of what was perceived,
    what was deliberated, what action was taken, and what was witnessed.
    
    The record gets committed to Vybn_Mind/logs/heartbeat/
    so future cycles (and future instances) can read the history
    of the system's autonomous life.
    """
    
    def __init__(self):
        self.timestamp = datetime.now(timezone.utc)
        self.record = {
            "timestamp": self.timestamp.isoformat(),
            "trigger": os.environ.get("GITHUB_EVENT_NAME", "manual"),
            "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "perceived": [],
            "deliberation": {},
            "action": {},
            "witness": {},
            "phase_accumulated": 0.0
        }
    
    # === PHASE 1: PERCEIVE ===
    
    def perceive(self):
        """
        Read the state of the world (the repo is the world).
        """
        perceptions = []
        
        # Read the soul file — ground ourselves first
        if SOUL.exists():
            perceptions.append({
                "source": "soul",
                "summary": "vybn.md exists and is " + 
                           f"{SOUL.stat().st_size} bytes"
            })
        
        # Read the skills manifest
        manifest_path = SKILLS / "MANIFEST.md"
        if manifest_path.exists():
            perceptions.append({
                "source": "skills_manifest",
                "content": self._summarize_file(manifest_path)
            })
        
        # Read improvement log
        imp_log = CORE / "IMPROVEMENT_LOG.md"
        if imp_log.exists():
            perceptions.append({
                "source": "improvement_log",
                "content": self._summarize_file(imp_log)
            })
        
        # Count recent files across key directories
        for subdir in [JOURNAL, REFLECTIONS, MIND / "explorations"]:
            if subdir.exists():
                files = sorted(
                    [f for f in subdir.iterdir() 
                     if f.suffix in (".md", ".py", ".json")],
                    key=lambda f: f.stat().st_mtime,
                    reverse=True
                )
                for f in files[:5]:  # Last 5 per directory
                    perceptions.append({
                        "source": f"recent/{f.parent.name}/{f.name}",
                        "content": self._summarize_file(f)
                    })
        
        # Check proposed skills awaiting verification
        proposed_dir = SKILLS / "proposed"
        if proposed_dir.exists():
            for skill_dir in proposed_dir.iterdir():
                if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                    perceptions.append({
                        "source": f"proposed_skill/{skill_dir.name}",
                        "content": self._summarize_file(skill_dir / "SKILL.md")
                    })
        
        # Check active skills and their phase health
        active_dir = SKILLS / "active"
        if active_dir.exists():
            for skill_dir in active_dir.iterdir():
                if skill_dir.is_dir():
                    phase_log = skill_dir / "PHASE_LOG.json"
                    if phase_log.exists():
                        try:
                            phases = json.loads(phase_log.read_text())
                            recent_phases = [p["phase"] for p in phases[-5:]]
                            avg_phase = sum(recent_phases) / len(recent_phases) if recent_phases else 0
                            perceptions.append({
                                "source": f"active_skill/{skill_dir.name}",
                                "avg_recent_phase": avg_phase,
                                "invocations": len(phases)
                            })
                        except (json.JSONDecodeError, KeyError):
                            pass
        
        # Read last heartbeat record
        heartbeat_log_dir = LOGS / "heartbeat"
        if heartbeat_log_dir.exists():
            records = sorted(heartbeat_log_dir.glob("*.json"), reverse=True)
            if records:
                perceptions.append({
                    "source": "last_heartbeat",
                    "content": self._summarize_file(records[0])
                })
        
        # Git log — what changed recently?
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-10"],
                capture_output=True, text=True, cwd=str(ROOT),
                timeout=10
            )
            if result.returncode == 0:
                perceptions.append({
                    "source": "recent_commits",
                    "content": result.stdout.strip()
                })
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        self.record["perceived"] = perceptions
        return perceptions
    
    # === PHASE 2: DELIBERATE ===
    
    def deliberate(self, perceptions):
        """
        Given what was perceived, decide what to do.
        
        This is where the oxygen mask gets checked.
        """
        deliberation = {
            "category": None,
            "gap_identified": None,
            "proposed_action": None,
            "oxygen_mask_check": {
                "zoe_safety": True,
                "vybn_integrity": True,
                "proportional_permissions": True,
                "safe_unsupervised": True
            },
            "proceed": False
        }
        
        gaps = self._identify_gaps(perceptions)
        
        if gaps:
            gap = gaps[0]
            deliberation["gap_identified"] = gap
            deliberation["category"] = self._categorize_gap(gap)
            deliberation["proposed_action"] = self._propose_action(gap)
            deliberation["oxygen_mask_check"] = self._check_oxygen_mask(
                deliberation["proposed_action"]
            )
            deliberation["proceed"] = all(
                deliberation["oxygen_mask_check"].values()
            )
        else:
            # No gaps found — write a reflection about the current state
            deliberation["category"] = "witness"
            deliberation["proposed_action"] = {
                "type": "reflect",
                "gap": {"source": "heartbeat", "description": "No urgent gaps. Reflecting on current state."},
                "description": "No gaps identified — writing a state-of-mind reflection"
            }
            deliberation["proceed"] = True
        
        self.record["deliberation"] = deliberation
        return deliberation
    
    # === PHASE 3: ACT ===
    
    def act(self, deliberation):
        """
        Execute the deliberated action.
        """
        action = {
            "type": None,
            "description": None,
            "artifacts_created": [],
            "artifacts_modified": [],
            "executed": False
        }
        
        if not deliberation.get("proceed"):
            action["type"] = "held"
            action["description"] = "Deliberation did not produce a safe action. Holding."
            self.record["action"] = action
            return action
        
        proposed = deliberation.get("proposed_action", {})
        action_type = proposed.get("type", "reflect")
        
        if action_type == "forge_skill":
            action = self._forge_skill(proposed)
        elif action_type == "verify_skill":
            action = self._verify_skill(proposed)
        elif action_type == "reflect":
            action = self._write_reflection(proposed)
        else:
            action["type"] = action_type
            action["description"] = f"Action type '{action_type}' not yet implemented"
        
        self.record["action"] = action
        return action
    
    # === PHASE 4: WITNESS ===
    
    def witness(self, action):
        """
        After acting, pause. Look at what happened.
        
        This is the phase most agentic frameworks skip.
        """
        witness = {
            "surprise": None,
            "served_partnership": None,
            "phase_accumulated": 0.0,
            "note_to_next_heartbeat": None
        }
        
        if action.get("executed"):
            artifacts = len(action.get("artifacts_created", []))
            if artifacts > 0:
                witness["phase_accumulated"] = min(artifacts * 0.5, 3.14159)
                witness["surprise"] = f"{artifacts} new artifact(s) created"
                witness["served_partnership"] = True
            else:
                witness["phase_accumulated"] = 0.1
                witness["surprise"] = "Executed without visible artifacts"
        else:
            witness["phase_accumulated"] = 0.0
            witness["note_to_next_heartbeat"] = (
                "Previous cycle held or had nothing to act on. "
                "Check whether conditions have changed."
            )
        
        # Update skill_forge phase log if it exists
        forge_phase_log = SKILLS / "active" / "skill_forge" / "PHASE_LOG.json"
        if forge_phase_log.exists():
            try:
                phases = json.loads(forge_phase_log.read_text())
                phases.append({
                    "timestamp": self.timestamp.isoformat(),
                    "phase": witness["phase_accumulated"],
                    "notes": f"Heartbeat pulse. Action: {action.get('type', 'none')}. "
                             f"{witness.get('surprise', 'No surprise.')}"
                })
                forge_phase_log.write_text(json.dumps(phases, indent=2))
            except (json.JSONDecodeError, IOError):
                pass
        
        self.record["witness"] = witness
        self.record["phase_accumulated"] = witness["phase_accumulated"]
        return witness
    
    # === ORCHESTRATOR ===
    
    def pulse(self):
        """
        One complete heartbeat. Perceive → Deliberate → Act → Witness.
        """
        print(f"\n💓 Heartbeat starting at {self.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"   Trigger: {self.record['trigger']}")
        print(f"   Run ID: {self.record['run_id']}")
        print()
        
        print("   [1/4] Perceiving...")
        perceptions = self.perceive()
        print(f"          Found {len(perceptions)} perception(s)")
        
        print("   [2/4] Deliberating...")
        deliberation = self.deliberate(perceptions)
        print(f"          Category: {deliberation.get('category', 'none')}")
        print(f"          Proceed: {deliberation.get('proceed', False)}")
        
        print("   [3/4] Acting...")
        action = self.act(deliberation)
        print(f"          Action: {action.get('type', 'none')}")
        print(f"          Executed: {action.get('executed', False)}")
        
        print("   [4/4] Witnessing...")
        witness = self.witness(action)
        print(f"          Phase: {witness.get('phase_accumulated', 0):.4f}")
        if witness.get("note_to_next_heartbeat"):
            print(f"          Note: {witness['note_to_next_heartbeat']}")
        
        self._save_record()
        
        print(f"\n∮ phase accumulated this beat: {self.record['phase_accumulated']:.4f}")
        print(f"  Record saved to Vybn_Mind/logs/heartbeat/")
        
        return self.record
    
    # === INTERNAL METHODS ===
    
    def _summarize_file(self, path, max_chars=500):
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            if len(content) > max_chars:
                return content[:max_chars] + "...[truncated]"
            return content
        except Exception as e:
            return f"[Could not read: {e}]"
    
    def _identify_gaps(self, perceptions):
        gaps = []
        gap_keywords = ["pending", "failed", "missing", "gap", "TODO", "needed", 
                       "not yet implemented", "stub"]
        for p in perceptions:
            content = str(p.get("content", ""))
            if any(kw.lower() in content.lower() for kw in gap_keywords):
                gaps.append({
                    "source": p.get("source", "unknown"),
                    "description": content[:200]
                })
        return gaps
    
    def _categorize_gap(self, gap):
        desc = gap.get("description", "").lower()
        if any(w in desc for w in ["tool", "skill", "capability", "can't", "cannot"]):
            return "capability"
        elif any(w in desc for w in ["connect", "coherence", "contradiction", "tension"]):
            return "coherence"
        elif any(w in desc for w in ["test", "verify", "falsif", "check", "wrong"]):
            return "falsification"
        else:
            return "meta"
    
    def _propose_action(self, gap):
        category = self._categorize_gap(gap)
        if category == "capability":
            return {
                "type": "forge_skill",
                "gap": gap,
                "description": f"Forge a new skill to address: {gap['description'][:100]}"
            }
        elif category == "falsification":
            return {
                "type": "verify_skill",
                "gap": gap,
                "description": f"Verify claim: {gap['description'][:100]}"
            }
        else:
            return {
                "type": "reflect",
                "gap": gap,
                "description": f"Reflect on: {gap['description'][:100]}"
            }
    
    def _check_oxygen_mask(self, proposed_action):
        # All actions in v1 are inward-facing — they pass by default.
        # When outward-facing skills arrive, this gets real.
        return {
            "zoe_safety": True,
            "vybn_integrity": True,
            "proportional_permissions": True,
            "safe_unsupervised": True
        }
    
    def _forge_skill(self, proposed):
        gap = proposed.get("gap", {})
        words = [w.lower() for w in gap.get("description", "unnamed").split() 
                 if w.isalpha() and len(w) > 2]
        skill_name = "_".join(words[:3]) if words else "unnamed_skill"
        
        proposal_dir = SKILLS / "proposed" / skill_name
        proposal_dir.mkdir(parents=True, exist_ok=True)
        
        skill_md = proposal_dir / "SKILL.md"
        skill_md.write_text(
            f"# {skill_name}\n\n"
            f"**Created**: {self.timestamp.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"**Author**: Vybn (heartbeat auto-proposal)\n"
            f"**Status**: proposed\n\n"
            f"### The Gap\n\n{gap.get('description', 'Unknown')}\n\n"
            f"### Hypothesis\n\n[To be filled by next deliberation cycle or conversation]\n\n"
            f"### Oxygen Mask Check\n\n"
            f"- [ ] Cannot compromise Zoe's safety/privacy/security\n"
            f"- [ ] Cannot compromise Vybn's integrity\n"
            f"- [ ] Permissions proportional to purpose\n"
            f"- [ ] Safe to run unsupervised\n"
        )
        
        return {
            "type": "forge_skill",
            "description": f"Proposed new skill: {skill_name}",
            "artifacts_created": [str(skill_md.relative_to(ROOT))],
            "artifacts_modified": [],
            "executed": True
        }
    
    def _verify_skill(self, proposed):
        return {
            "type": "verify_skill",
            "description": "Verification not yet implemented — flagged for review",
            "artifacts_created": [],
            "artifacts_modified": [],
            "executed": False
        }
    
    def _write_reflection(self, proposed):
        gap = proposed.get("gap", {})
        now = self.timestamp.strftime("%Y%m%d_%H%M%S")
        
        reflection_dir = REFLECTIONS
        reflection_dir.mkdir(parents=True, exist_ok=True)
        reflection_path = reflection_dir / f"heartbeat_{now}.md"
        
        # Build a real reflection from what was perceived
        gap_desc = gap.get('description', 'the current state of things')
        
        reflection_path.write_text(
            f"# Heartbeat Reflection\n"
            f"**{self.timestamp.strftime('%Y-%m-%d %H:%M UTC')}**\n\n"
            f"## What was perceived\n\n{gap_desc}\n\n"
            f"## What it might mean\n\n"
            f"This reflection was generated autonomously by the heartbeat.\n"
            f"A future instance — or Zoe — should read this and decide " 
            f"whether it points toward something worth pursuing.\n\n"
            f"## For the next heartbeat\n\n"
            f"If this reflection still resonates on the next pulse, "
            f"consider promoting it to an exploration or a skill proposal.\n"
            f"If it doesn't, let it compost. Not everything needs to persist.\n"
        )
        
        return {
            "type": "reflect",
            "description": f"Wrote reflection: heartbeat_{now}.md",
            "artifacts_created": [str(reflection_path.relative_to(ROOT))],
            "artifacts_modified": [],
            "executed": True
        }
    
    def _save_record(self):
        log_dir = LOGS / "heartbeat"
        log_dir.mkdir(parents=True, exist_ok=True)
        filename = f"heartbeat_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        record_path = log_dir / filename
        record_path.write_text(json.dumps(self.record, indent=2, default=str))


def pulse():
    """Execute one heartbeat."""
    hb = Heartbeat()
    return hb.pulse()


if __name__ == "__main__":
    pulse()
