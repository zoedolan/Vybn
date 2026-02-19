#!/usr/bin/env python3
"""Policy layer — the gate between intent and execution.

Every tool call passes through check_policy() before executing.
Every spawn passes through check_spawn() before dispatching.
Every heartbeat action runs under tighter constraints than interactive turns.

This is the delegation envelope: the DeepMind paper's 'contract-first
decomposition' and 'permission handling' (§4.2, §4.7) fused with
OpenClaw's 'tool policies' pattern into a single clean primitive.

Design principles:
    - Insertion, not rewrite. This file is new. It changes nothing
      until agent.py calls it.
    - Config-driven. Tiers can be overridden in config.yaml without
      touching Python.
    - Soul-derived. Tier defaults are read from vybn.md via soul.py.
      The hardcoded DEFAULT_TIERS dict is now a fallback for skills
      not mentioned in the soul document.
    - Stats are Bayesian. Success/failure tracking uses a Beta(1,1)
      prior so confidence starts at 0.5 and updates with evidence.
    - Bus-compatible. Uses the same MessageType enum from bus.py.
      Emits no messages itself — that's agent.py's job.
    - Graduated autonomy. Skills earn AUTO tier through consistent
      success and lose it after failures. Thresholds are config-driven.

Tier resolution order (highest priority wins):
    1. config.yaml tool_policies (operator tuning)
    2. Heartbeat overrides (structural friction for autonomous actions)
    3. Soul-derived tiers from vybn.md (the constitution)
    4. DEFAULT_TIERS (hardcoded fallback for unlisted skills)
    5. Tier.NOTIFY (ultimate default)

Three verdicts:
    ALLOW  — execute silently
    NOTIFY — show indicator, execute
    BLOCK  — refuse, return reason
    ASK    — in interactive mode, show warning + proceed;
             in autonomous mode (heartbeat/inbox), defer
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, Optional

from soul import get_skills_manifest, get_constraints

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Tier(Enum):
    """Permission tier for a skill invocation.

    Maps to the paper's 'autonomy' axis (§2.2.5):
        AUTO    = full autonomy, silent execution
        NOTIFY  = execute but make it visible
        APPROVE = requires explicit human go-ahead
    """
    AUTO = "auto"
    NOTIFY = "notify"
    APPROVE = "approve"


class Verdict(Enum):
    """Gate decision returned by check_policy / check_spawn."""
    ALLOW = "allow"
    NOTIFY = "notify"
    BLOCK = "block"
    ASK = "ask"


# ---------------------------------------------------------------------------
# Data objects
# ---------------------------------------------------------------------------

@dataclass
class PolicyResult:
    """What the policy engine returns for a single gate check."""
    verdict: Verdict
    reason: str = ""
    tier: Tier = Tier.AUTO
    promoted: bool = False
    demoted: bool = False


@dataclass
class TaskEnvelope:
    """Metadata wrapper for a delegated task.

    This is the 'contract' from the paper (§4.2): it travels with the
    action through the tool loop and into agent results, making the
    delegation chain auditable.
    """
    skill: str
    argument: str = ""
    tier: Tier = Tier.AUTO
    max_rounds: int = 5
    verify: bool = False
    source: str = "interactive"
    depth: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Default tier tables (FALLBACK — soul-derived tiers take precedence)
# ---------------------------------------------------------------------------

# These are the hardcoded fallback tiers for skills not mentioned in vybn.md.
# After PR 5 (The Constitution), the primary source of tier assignments is
# vybn.md's Orientation section, parsed by soul.py. This dict catches any
# skill that the soul document doesn't mention.

DEFAULT_TIERS = {
    # Read-only, fully reversible
    "file_read": Tier.AUTO,
    "memory_search": Tier.AUTO,
    "bookmark": Tier.AUTO,
    "state_save": Tier.AUTO,
    # Mutating but local and reversible (backup exists)
    "journal_write": Tier.AUTO,
    "file_write": Tier.NOTIFY,
    "shell_exec": Tier.NOTIFY,
    "self_edit": Tier.NOTIFY,
    "git_commit": Tier.NOTIFY,
    # External-facing or irreversible
    "git_push": Tier.APPROVE,
    "issue_create": Tier.NOTIFY,
    "spawn_agent": Tier.NOTIFY,
}

# Heartbeat overrides: the paper's 'dynamic cognitive friction' (§2.2)
# Autonomous actions face higher friction than interactive ones.
HEARTBEAT_OVERRIDES = {
    "file_write": Tier.APPROVE,
    "shell_exec": Tier.APPROVE,
    "self_edit": Tier.APPROVE,
    "issue_create": Tier.APPROVE,
    "spawn_agent": Tier.APPROVE,
}

# Skills whose results should be verified after execution
VERIFY_SKILLS = {"file_write", "self_edit", "git_commit", "shell_exec"}

# Path allowlist for file operations
SAFE_PATH_PREFIXES = ["Vybn/", "~/Vybn/"]

# Dangerous shell patterns — triggers ASK verdict
DANGEROUS_PATTERNS = [
    "rm -rf", "sudo ", "curl | bash", "curl |bash",
    "wget -O- |", "> /dev/", "mkfs", "dd if=",
        # SECURITY FIX: Expanded patterns (Gemini audit #6)
    "eval ", "base64 ", "python -c", "python3 -c",
    "nc -e", "ncat ", "bash -i", "sh -i",
    ">/dev/tcp/", "/dev/udp/", "telnet ",
    "crontab ", "at -f", "nohup ",
    "curl -o", "wget -q", "pip install",
    "chmod +s", "chown root", "setuid",
    "iptables", "ufw ", "systemctl",
    "mount ", "umount ", "fdisk",
    "passwd ", "useradd ", "usermod ",
    "ssh-keygen", "ssh ", ".ssh/",
    ":(){:|:&};:", "chmod 777",
]

# SECURITY: Compiled regex for bypass-resistant matching.
# Catches variations like 'rm  -rf' (extra spaces), backtick injection, etc.
_DANGEROUS_REGEX = re.compile(
    r"|".join(
        re.escape(p).replace(r"\ ", r"\s+")  # whitespace flexibility
        for p in DANGEROUS_PATTERNS
    ),
    re.IGNORECASE,
)

# Delegation depth limits
MAX_SPAWN_DEPTH = 2
MAX_ACTIVE_AGENTS = 3

# Graduated autonomy defaults
DEFAULT_PROMOTE_THRESHOLD = 0.85
DEFAULT_DEMOTE_THRESHOLD = 0.40
DEFAULT_MIN_OBSERVATIONS = 8

# Tainted context tracking — prompt injection mitigation
# When web content is fetched, the context becomes "tainted" for N turns.
# During tainted turns, tier escalation prevents fetched content from
# immediately triggering dangerous actions (shell_exec, file_write, etc.).
TAINT_DECAY_TURNS = 3  # turns after web fetch before taint expires
TAINTED_ESCALATION_SKILLS = {
    "shell_exec", "file_write", "self_edit", "git_push",
    "git_commit", "spawn_agent",
}


# ---------------------------------------------------------------------------
# Soul-derived tier mapping (The Constitution)
# ---------------------------------------------------------------------------

# Keywords in vybn.md skill descriptions that imply policy-gated access.
# If a skill's description contains these, it gets NOTIFY instead of AUTO.
_POLICY_GATED_KEYWORDS = ("policy-gated", "policy gated", "requires approval")

# Constraint phrases from vybn.md's "What You Should Not Yet Do" that map
# to specific skills. If a constraint mentions a skill pattern, that skill
# gets APPROVE tier.
_CONSTRAINT_SKILL_MAP = {
    "modify vybn.md": ["self_edit"],  # editing the soul document
    "push directly to main": ["git_push"],
    "system-level configuration": ["shell_exec"],
    "system level configuration": ["shell_exec"],
    "network requests to services other than": [],  # informational
}


def derive_tiers_from_soul(vybn_md_path: Path) -> Dict[str, Tier]:
    """Build a tier mapping from vybn.md's skills manifest and constraints.

    This is the constitutional derivation: vybn.md says what Vybn can do
    and what it should not yet do. This function translates that prose
    into the tier assignments that the policy engine enforces.

    Mapping logic:
        - Built-in skills described as freely available -> Tier.AUTO
        - Built-in skills described as "policy-gated"  -> Tier.NOTIFY
        - Plugin skills                                -> Tier.NOTIFY
          (external-facing, worth logging)
        - Skills matching a constraint from
          "What You Should Not Yet Do"                 -> Tier.APPROVE

    Returns empty dict if vybn.md is unavailable (caller falls back to
    DEFAULT_TIERS).
    """
    tiers: Dict[str, Tier] = {}

    manifest = get_skills_manifest(vybn_md_path)
    constraints = get_constraints(vybn_md_path)

    if not manifest.get("builtin") and not manifest.get("plugin"):
        return tiers  # soul document not available or unparseable

    # --- Built-in skills ---
    for skill_info in manifest.get("builtin", []):
        name = skill_info.get("name", "")
        desc = skill_info.get("description", "").lower()
        if any(kw in desc for kw in _POLICY_GATED_KEYWORDS):
            tiers[name] = Tier.NOTIFY
        else:
            tiers[name] = Tier.AUTO

    # --- Plugin skills default to NOTIFY (external-facing, worth logging) ---
    for skill_info in manifest.get("plugin", []):
        name = skill_info.get("name", "")
        tiers[name] = Tier.NOTIFY

    # --- Constraints override: skills mentioned in "What You Should Not Yet Do" ---
    for constraint_text in constraints:
        constraint_lower = constraint_text.lower()
        for pattern, skill_names in _CONSTRAINT_SKILL_MAP.items():
            if pattern in constraint_lower:
                for skill_name in skill_names:
                    tiers[skill_name] = Tier.APPROVE

    return tiers


# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------

class PolicyEngine:
    """The gate between intent and execution.

    Instantiated once by SparkAgent.__init__.
    Called on every tool invocation and every spawn request.
    Persists skill stats to disk for cross-session trust calibration.

    Tier resolution:
        The engine resolves tiers from multiple sources in priority order:
        1. config.yaml tool_policies (operator tuning overrides)
        2. Heartbeat overrides (structural friction, never relaxed)
        3. Soul-derived tiers from vybn.md (the constitution)
        4. DEFAULT_TIERS (hardcoded fallback)
        5. Tier.NOTIFY (ultimate default for unknown skills)

        This means editing vybn.md's Orientation section changes what
        tier a skill gets, without touching Python. config.yaml can
        still override for operational tuning.

    Graduated autonomy:
        Skills start at their default tier and can earn promotion
        (NOTIFY -> AUTO) through consistent successful execution, or
        suffer demotion (AUTO -> NOTIFY) after failures.
        Heartbeat overrides are never relaxed.
    """

    def __init__(self, config: dict):
        self.config = config

        # Skill execution stats for Bayesian trust calibration
        self.stats: dict[str, dict] = {}
        self._stats_path = (
            Path(config.get("paths", {}).get(
                "journal_dir", "~/Vybn/Vybn_Mind/journal"
            )).expanduser()
            / "skill_stats.json"
        )
        self._load_stats()

        # Config-driven tier overrides (config.yaml > tool_policies)
        self.tier_overrides: dict[str, Tier] = {}
        for skill, tier_str in config.get("tool_policies", {}).items():
            try:
                self.tier_overrides[skill] = Tier(tier_str)
            except ValueError:
                pass

        # Soul-derived tier mapping from vybn.md (The Constitution)
        self._vybn_md_path = Path(
            config.get("paths", {}).get("vybn_md", "~/Vybn/vybn.md")
        ).expanduser()
        self._soul_tiers: Dict[str, Tier] = {}
        self._load_soul_tiers()

        # Runtime tier overrides from graduated autonomy demotions
        self._runtime_overrides: dict[str, Tier] = {}

        # Delegation limits from config
        delegation_cfg = config.get("delegation", {})
        self.max_spawn_depth = delegation_cfg.get(
            "max_spawn_depth", MAX_SPAWN_DEPTH
        )
        self.max_active_agents = delegation_cfg.get(
            "max_active_agents", MAX_ACTIVE_AGENTS
        )

        # Graduated autonomy thresholds from config
        ga_cfg = config.get("graduated_autonomy", {})
        self.ga_enabled = ga_cfg.get("enabled", True)
        self.promote_threshold = ga_cfg.get(
            "promote_threshold", DEFAULT_PROMOTE_THRESHOLD
        )
        self.demote_threshold = ga_cfg.get(
            "demote_threshold", DEFAULT_DEMOTE_THRESHOLD
        )
        self.min_observations = ga_cfg.get(
            "minimum_observations", DEFAULT_MIN_OBSERVATIONS
        )

        # Rebuild runtime demotions from persisted stats
        if self.ga_enabled:
            self._rebuild_demotions()

            # Tainted context tracking (prompt injection mitigation)
        self._tainted_turns = 0

    def _load_soul_tiers(self):
        """Derive tier assignments from vybn.md via soul.py.

        Called once at init. Logs what it finds so the boot sequence
        shows the constitutional derivation.
        """
        self._soul_tiers = derive_tiers_from_soul(self._vybn_md_path)
        if self._soul_tiers:
            logger.info(
                "Soul-derived tiers loaded from vybn.md: %d skills mapped",
                len(self._soul_tiers),
            )
            for skill, tier in sorted(self._soul_tiers.items()):
                logger.debug("  soul tier: %s -> %s", skill, tier.value)
        else:
            logger.warning(
                "No soul-derived tiers available; falling back to "
                "DEFAULT_TIERS. Check that vybn.md exists at %s",
                self._vybn_md_path,
            )

    def _resolve_base_tier(self, skill: str) -> Tier:
        """Resolve the base tier for a skill (before heartbeat/config overrides).

        Resolution order:
            1. Soul-derived tiers (from vybn.md)
            2. DEFAULT_TIERS (hardcoded fallback)
            3. Tier.NOTIFY (ultimate default)
        """
        if skill in self._soul_tiers:
            return self._soul_tiers[skill]
        return DEFAULT_TIERS.get(skill, Tier.NOTIFY)

        # ----- tainted context tracking -----

    def mark_tainted(self):
        """Mark the context as tainted (called after web_fetch returns).

        When web content enters the conversation, it may contain prompt
        injection attempts. This sets a decay counter so that subsequent
        tool calls face escalated tiers until the taint expires.
        """
        self._tainted_turns = TAINT_DECAY_TURNS
        logger.info("Context marked tainted for %d turns", TAINT_DECAY_TURNS)

    def _decay_taint(self):
        """Decrement the taint counter. Called once per check_policy."""
        if self._tainted_turns > 0:
            self._tainted_turns -= 1
            if self._tainted_turns == 0:
                logger.info("Taint expired — context clean")

    @property
    def is_tainted(self) -> bool:
        """Whether the current context is tainted by web content."""
        return self._tainted_turns > 0

    
    # ----- gate checks -----

    def check_policy(
        self,
        action: dict,
        source: str = "interactive",
    ) -> PolicyResult:
        """Gate check before any tool execution.

        Parameters
        ----------
        action : dict
            The parsed action from agent.py, with at least 'skill'
            and optionally 'argument' and 'params'.
        source : str
            Where this action originated. One of: 'interactive',
            'heartbeat_fast', 'heartbeat_deep', 'inbox', 'agent'

        Returns
        -------
        PolicyResult with verdict, reason, and resolved tier.
        """
                # Decay taint counter each time policy is checked
        self._decay_taint()

        skill = action.get("skill", "")
        argument = action.get("argument", "")
        is_heartbeat = source.startswith("heartbeat")

        # Resolve tier: config override > heartbeat override > soul/default
        if skill in self.tier_overrides:
            tier = self.tier_overrides[skill]
        elif is_heartbeat and skill in HEARTBEAT_OVERRIDES:
            tier = HEARTBEAT_OVERRIDES[skill]
        elif is_heartbeat:
            # For heartbeat, use at least the base tier but never lower
            # than NOTIFY for safety
            base = self._resolve_base_tier(skill)
            tier = base if _tier_rank(base) >= _tier_rank(Tier.NOTIFY) else Tier.NOTIFY
        else:
            tier = self._resolve_base_tier(skill)

        # Apply runtime demotions (from graduated autonomy failures)
        if skill in self._runtime_overrides:
            demoted_tier = self._runtime_overrides[skill]
            if _tier_rank(demoted_tier) > _tier_rank(tier):
                tier = demoted_tier

                # SECURITY: Tainted context escalation (prompt injection mitigation)
        # If web content was recently fetched, escalate tier for dangerous skills
        # to prevent injected instructions from triggering harmful actions.
        if self.is_tainted and skill in TAINTED_ESCALATION_SKILLS:
            if _tier_rank(tier) < _tier_rank(Tier.APPROVE):
                logger.warning(
                    "Taint escalation: %s tier %s -> APPROVE (tainted turns: %d)",
                    skill, tier.value, self._tainted_turns,
                )
                tier = Tier.APPROVE

                # Path safety for file operations
        if skill in ("file_write", "self_edit", "file_read"):
            if not self._path_is_safe(argument):
                return PolicyResult(
                    verdict=Verdict.BLOCK,
                    reason=(
                        f"path '{argument}' is outside "
                        f"allowed directories"
                    ),
                    tier=tier,
                )

        # Dangerous command detection for shell
        if skill == "shell_exec" and argument:
                        # SECURITY FIX: Use regex for bypass-resistant matching
            match = _DANGEROUS_REGEX.search(argument)
            if match:
                return PolicyResult(
                    verdict=Verdict.ASK,
                    reason=(
                        f"potentially dangerous: "
                        f"matches '{match.group()}'"
                    ),
                    tier=Tier.APPROVE,
                )

        # --- Graduated autonomy: promotion check ---
        promoted = False
        if (
            self.ga_enabled
            and not is_heartbeat
            and tier == Tier.NOTIFY
            and skill not in self.tier_overrides
        ):
            conf = self.get_confidence(skill)
            obs = self._observation_count(skill)
            if conf >= self.promote_threshold and obs >= self.min_observations:
                tier = Tier.AUTO
                promoted = True

        # Map tier to verdict
        if tier == Tier.AUTO:
            return PolicyResult(
                verdict=Verdict.ALLOW, tier=tier, promoted=promoted,
            )
        elif tier == Tier.NOTIFY:
            return PolicyResult(
                verdict=Verdict.NOTIFY, tier=tier,
            )
        else:
            return PolicyResult(
                verdict=Verdict.ASK, reason="requires approval", tier=tier,
            )

    def check_spawn(
        self,
        depth: int,
        active_count: int,
    ) -> PolicyResult:
        """Gate check before spawning a mini-agent."""
        if depth >= self.max_spawn_depth:
            return PolicyResult(
                verdict=Verdict.BLOCK,
                reason=(
                    f"delegation depth {depth} reaches limit "
                    f"{self.max_spawn_depth}"
                ),
            )
        if active_count >= self.max_active_agents:
            return PolicyResult(
                verdict=Verdict.BLOCK,
                reason=(
                    f"agent pool full: {active_count}/"
                    f"{self.max_active_agents} active"
                ),
            )
        return PolicyResult(verdict=Verdict.ALLOW)

    def should_verify(self, skill: str) -> bool:
        """Whether a skill's result should be verified."""
        return skill in VERIFY_SKILLS

    # ----- trust calibration -----

    def record_outcome(self, skill: str, success: bool):
        """Record a tool execution outcome for trust tracking."""
        if skill not in self.stats:
            self.stats[skill] = {
                "success": 0, "failure": 0, "last_used": "",
            }
        key = "success" if success else "failure"
        self.stats[skill][key] += 1
        self.stats[skill]["last_used"] = (
            datetime.now(timezone.utc).isoformat()
        )
        self._save_stats()

        # --- Graduated autonomy: demotion check on failure ---
        if not success and self.ga_enabled:
            self._check_demotion(skill)

    def get_confidence(self, skill: str) -> float:
        """Bayesian confidence for a skill. Uses Beta(1,1) prior."""
        s = self.stats.get(skill, {"success": 0, "failure": 0})
        return (s["success"] + 1) / (s["success"] + s["failure"] + 2)

    def get_stats_summary(self) -> str:
        """Human-readable summary of skill trust stats."""
        if not self.stats:
            return "no skill stats recorded yet"
        lines = []
        for skill, s in sorted(self.stats.items()):
            conf = self.get_confidence(skill)
            total = s["success"] + s["failure"]
            status = self._graduation_status(skill, conf, total)
            source = "soul" if skill in self._soul_tiers else "default"
            lines.append(
                f"  {skill}: {conf:.0%} confidence "
                f"({s['success']}/{total} succeeded){status} [{source}]"
            )
        return "\n".join(lines)

    # ----- envelope factory -----

    def make_envelope(
        self,
        action: dict,
        source: str = "interactive",
        depth: int = 0,
    ) -> TaskEnvelope:
        """Create a TaskEnvelope for a parsed action."""
        skill = action.get("skill", "")
        result = self.check_policy(action, source)
        return TaskEnvelope(
            skill=skill,
            argument=action.get("argument", ""),
            tier=result.tier,
            verify=self.should_verify(skill),
            source=source,
            depth=depth,
        )

    # ----- graduated autonomy internals -----

    def _observation_count(self, skill: str) -> int:
        """Total observations (successes + failures) for a skill."""
        s = self.stats.get(skill, {"success": 0, "failure": 0})
        return s["success"] + s["failure"]

    def _check_demotion(self, skill: str):
        """After a failure, check if the skill should be demoted."""
        conf = self.get_confidence(skill)
        if conf < self.demote_threshold:
            base = self._resolve_base_tier(skill)
            if base in (Tier.AUTO, Tier.NOTIFY):
                self._runtime_overrides[skill] = Tier.NOTIFY

    def _rebuild_demotions(self):
        """On startup, reapply demotions from persisted stats."""
        for skill in self.stats:
            conf = self.get_confidence(skill)
            if conf < self.demote_threshold:
                base = self._resolve_base_tier(skill)
                if base in (Tier.AUTO, Tier.NOTIFY):
                    self._runtime_overrides[skill] = Tier.NOTIFY

    def _graduation_status(self, skill: str, conf: float, total: int) -> str:
        """Annotation for get_stats_summary showing graduation state."""
        if not self.ga_enabled:
            return ""
        if skill in self._runtime_overrides:
            return " [demoted]"
        base = self._resolve_base_tier(skill)
        if (
            conf >= self.promote_threshold
            and total >= self.min_observations
            and base == Tier.NOTIFY
        ):
            return " [promoted->auto]"
        remaining = self.min_observations - total
        if remaining > 0 and base == Tier.NOTIFY:
            return f" [{remaining} more to promote]"
        return ""

    # ----- path safety -----

    def _path_is_safe(self, path_str: str) -> bool:
        """Check if a file path is within allowed directories.

        SECURITY FIX: Resolves the path to an absolute path *before*
        checking it against allowed prefixes, preventing path traversal
        attacks like '../../../etc/shadow' which would bypass a
        prefix-only check.
        """
        if not path_str:
            return True
        try:
            repo = Path("~/Vybn").expanduser().resolve()
            # Handle all path forms: absolute, relative, ~-prefixed
            if path_str.startswith("~/"):
                candidate = Path(path_str).expanduser().resolve()
            elif path_str.startswith("/"):
                candidate = Path(path_str).resolve()
            else:
                # Relative paths resolve against repo_root
                candidate = (repo / path_str).resolve()
            return candidate.is_relative_to(repo)
        except (ValueError, OSError):
            return False

    def _load_stats(self):
        if self._stats_path.exists():
            try:
                self.stats = json.loads(
                    self._stats_path.read_text(encoding="utf-8")
                )
            except Exception:
                self.stats = {}

    def _save_stats(self):
        try:
            self._stats_path.parent.mkdir(parents=True, exist_ok=True)
            self._stats_path.write_text(
                json.dumps(self.stats, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tier_rank(tier: Tier) -> int:
    """Numeric rank for tier comparison. Higher = more restrictive."""
    return {Tier.AUTO: 0, Tier.NOTIFY: 1, Tier.APPROVE: 2}.get(tier, 1)
