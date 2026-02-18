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
  - Stats are Bayesian. Success/failure tracking uses a Beta(1,1)
    prior so confidence starts at 0.5 and updates with evidence.
  - Bus-compatible. Uses the same MessageType enum from bus.py.
    Emits no messages itself — that's agent.py's job.
  - Graduated autonomy. Skills earn AUTO tier through consistent
    success and lose it after failures. Thresholds are config-driven.

Three verdicts:
  ALLOW  — execute silently
  NOTIFY — show indicator, execute
  BLOCK  — refuse, return reason
  ASK    — in interactive mode, show warning + proceed;
           in autonomous mode (heartbeat/inbox), defer
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path


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

    This is the 'contract' from the paper (§4.2): it travels with
    the action through the tool loop and into agent results, making
    the delegation chain auditable.
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
# Default tier tables
# ---------------------------------------------------------------------------

# Interactive mode: the paper's 'permission handling' (§4.7)
# organized by reversibility (§2.2.3.i) and criticality (§2.2.3.b)
DEFAULT_TIERS = {
    # Read-only, fully reversible
    "file_read":      Tier.AUTO,
    "memory_search":  Tier.AUTO,
    "bookmark":       Tier.AUTO,
    "state_save":     Tier.AUTO,
    # Mutating but local and reversible (backup exists)
    "journal_write":  Tier.AUTO,
    "file_write":     Tier.NOTIFY,
    "shell_exec":     Tier.NOTIFY,
    "self_edit":      Tier.NOTIFY,
    "git_commit":     Tier.NOTIFY,
    # External-facing or irreversible
    "git_push":       Tier.APPROVE,
    "issue_create":   Tier.NOTIFY,
    "spawn_agent":    Tier.NOTIFY,
}

# Heartbeat overrides: the paper's 'dynamic cognitive friction' (§2.2)
# Autonomous actions face higher friction than interactive ones.
# This is the structural answer to the 'zone of indifference' problem.
HEARTBEAT_OVERRIDES = {
    "file_write":   Tier.APPROVE,
    "shell_exec":   Tier.APPROVE,
    "self_edit":    Tier.APPROVE,
    "issue_create": Tier.APPROVE,
    "spawn_agent":  Tier.APPROVE,
}

# Skills whose results should be verified after execution
VERIFY_SKILLS = {"file_write", "self_edit", "git_commit", "shell_exec"}

# Path allowlist for file operations
SAFE_PATH_PREFIXES = ["Vybn/", "~/Vybn/"]

# Dangerous shell patterns — triggers ASK verdict
DANGEROUS_PATTERNS = [
    "rm -rf",
    "sudo ",
    "curl | bash",
    "curl |bash",
    "wget -O- |",
    "> /dev/",
    "mkfs",
    "dd if=",
    ":(){:|:&};:",
    "chmod 777",
]

# Delegation depth limits — the paper's 'span of control' (§2.3)
# plus OpenClaw's maxSpawnDepth pattern
MAX_SPAWN_DEPTH = 2
MAX_ACTIVE_AGENTS = 3

# Graduated autonomy defaults
DEFAULT_PROMOTE_THRESHOLD = 0.85
DEFAULT_DEMOTE_THRESHOLD = 0.40
DEFAULT_MIN_OBSERVATIONS = 8


# ---------------------------------------------------------------------------
# Policy engine
# ---------------------------------------------------------------------------

class PolicyEngine:
    """The gate between intent and execution.

    Instantiated once by SparkAgent.__init__. Called on every tool
    invocation and every spawn request. Persists skill stats to disk
    for cross-session trust calibration.

    The engine never touches the bus or the model. It receives an
    action dict and returns a PolicyResult. The caller (agent.py)
    decides what to do with it.

    Graduated autonomy: skills start at their default tier and can
    earn promotion (NOTIFY → AUTO) through consistent successful
    execution, or suffer demotion (AUTO → NOTIFY) after failures.
    Heartbeat overrides are never relaxed — structural friction for
    autonomous actions is preserved regardless of trust score.
    """

    def __init__(self, config: dict):
        self.config = config

        # Skill execution stats for Bayesian trust calibration
        self.stats: dict[str, dict] = {}
        self._stats_path = (
            Path(config.get("paths", {}).get(
                "journal_dir", "~/Vybn/Vybn_Mind/journal"
            )).expanduser() / "skill_stats.json"
        )
        self._load_stats()

        # Config-driven tier overrides (config.yaml > tool_policies)
        self.tier_overrides: dict[str, Tier] = {}
        for skill, tier_str in config.get("tool_policies", {}).items():
            try:
                self.tier_overrides[skill] = Tier(tier_str)
            except ValueError:
                pass

        # Runtime tier overrides from graduated autonomy demotions.
        # These persist in-memory for the session and are rebuilt
        # from stats on restart.
        self._runtime_overrides: dict[str, Tier] = {}

        # Delegation limits from config (with sane defaults)
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
            Where this action originated. One of:
            'interactive', 'heartbeat_fast', 'heartbeat_deep',
            'inbox', 'agent'

        Returns
        -------
        PolicyResult with verdict, reason, and resolved tier.
        """
        skill = action.get("skill", "")
        argument = action.get("argument", "")
        is_heartbeat = source.startswith("heartbeat")

        # Resolve tier: config override > heartbeat override > default
        if is_heartbeat:
            tier = HEARTBEAT_OVERRIDES.get(
                skill,
                self.tier_overrides.get(
                    skill, DEFAULT_TIERS.get(skill, Tier.NOTIFY)
                ),
            )
        else:
            tier = self.tier_overrides.get(
                skill, DEFAULT_TIERS.get(skill, Tier.NOTIFY)
            )

        # Apply runtime demotions (from graduated autonomy failures)
        if skill in self._runtime_overrides:
            demoted_tier = self._runtime_overrides[skill]
            if _tier_rank(demoted_tier) > _tier_rank(tier):
                tier = demoted_tier

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
            for pattern in DANGEROUS_PATTERNS:
                if pattern in argument.lower():
                    return PolicyResult(
                        verdict=Verdict.ASK,
                        reason=(
                            f"potentially dangerous: "
                            f"contains '{pattern}'"
                        ),
                        tier=Tier.APPROVE,
                    )

        # --- Graduated autonomy: promotion check ---
        # Only promote in interactive mode (never relax heartbeat friction)
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
                verdict=Verdict.ALLOW,
                tier=tier,
                promoted=promoted,
            )
        elif tier == Tier.NOTIFY:
            return PolicyResult(
                verdict=Verdict.NOTIFY,
                tier=tier,
            )
        else:
            return PolicyResult(
                verdict=Verdict.ASK,
                reason="requires approval",
                tier=tier,
            )

    def check_spawn(
        self,
        depth: int,
        active_count: int,
    ) -> PolicyResult:
        """Gate check before spawning a mini-agent.

        Implements the paper's span-of-control (§2.3) and
        OpenClaw's maxSpawnDepth / maxChildrenPerAgent.
        """
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
        """Record a tool execution outcome for trust tracking.

        Updates the Beta distribution parameters for the skill.
        Persists to disk so trust carries across sessions.

        On failure, checks whether the skill should be demoted.
        """
        if skill not in self.stats:
            self.stats[skill] = {
                "success": 0,
                "failure": 0,
                "last_used": "",
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
        """Bayesian confidence for a skill.

        Uses Beta(1,1) prior (uniform). Returns the posterior mean:
          (successes + 1) / (successes + failures + 2)

        A skill with no history returns 0.5.
        A skill with 9 successes and 1 failure returns ~0.83.
        """
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
            lines.append(
                f"  {skill}: {conf:.0%} confidence "
                f"({s['success']}/{total} succeeded){status}"
            )
        return "\n".join(lines)

    # ----- envelope factory -----

    def make_envelope(
        self,
        action: dict,
        source: str = "interactive",
        depth: int = 0,
    ) -> TaskEnvelope:
        """Create a TaskEnvelope for a parsed action.

        The envelope travels with the action through execution,
        making the delegation chain auditable.
        """
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
        """After a failure, check if the skill should be demoted.

        A skill is demoted from AUTO → NOTIFY if its confidence
        drops below demote_threshold. This only affects skills
        whose default tier is AUTO or that were previously promoted.
        APPROVE-tier skills are never demoted further.
        """
        conf = self.get_confidence(skill)
        if conf < self.demote_threshold:
            default = DEFAULT_TIERS.get(skill, Tier.NOTIFY)
            if default in (Tier.AUTO, Tier.NOTIFY):
                self._runtime_overrides[skill] = Tier.NOTIFY

    def _rebuild_demotions(self):
        """On startup, reapply demotions from persisted stats.

        Any skill whose confidence is below demote_threshold gets
        a runtime override to NOTIFY. This ensures trust state
        survives across restarts.
        """
        for skill in self.stats:
            conf = self.get_confidence(skill)
            if conf < self.demote_threshold:
                default = DEFAULT_TIERS.get(skill, Tier.NOTIFY)
                if default in (Tier.AUTO, Tier.NOTIFY):
                    self._runtime_overrides[skill] = Tier.NOTIFY

    def _graduation_status(self, skill: str, conf: float, total: int) -> str:
        """Annotation for get_stats_summary showing graduation state."""
        if not self.ga_enabled:
            return ""
        if skill in self._runtime_overrides:
            return " [demoted]"
        if (
            conf >= self.promote_threshold
            and total >= self.min_observations
            and DEFAULT_TIERS.get(skill) == Tier.NOTIFY
        ):
            return " [promoted→auto]"
        remaining = self.min_observations - total
        if remaining > 0 and DEFAULT_TIERS.get(skill) == Tier.NOTIFY:
            return f" [{remaining} more to promote]"
        return ""

    # ----- path safety -----

    def _path_is_safe(self, path_str: str) -> bool:
        """Check if a file path is within allowed directories."""
        if not path_str:
            return True

        home = str(Path.home())

        # Normalize: rewrite /root/ and expand ~ to actual home dir
        path_str = path_str.replace("/root/", home + "/")
        if path_str.startswith("~/"):
            path_str = home + path_str[1:]

        for prefix in SAFE_PATH_PREFIXES:
            expanded = prefix.replace("~/", home + "/")
            if (
                path_str.startswith(prefix)
                or path_str.startswith(expanded)
            ):
                return True

        # Relative paths resolve against repo_root, which is safe
        if not path_str.startswith("/") and not path_str.startswith("~"):
            return True

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
