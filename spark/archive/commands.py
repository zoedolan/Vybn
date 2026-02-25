#!/usr/bin/env python3
"""Shared command handlers for the Vybn Spark Agent.

Extracted from agent.py (Phase 3 refactor) so that both tui.py
and the web interface can execute the same slash-commands without
duplicating logic.

Every public function here takes an agent (SparkAgent) and returns
a string.  The caller decides how to display it (Rich panel, plain
print, WebSocket JSON, etc.).  No print() calls in this module.

Recalibrated: February 20, 2026 — removed /effervesce (hardcoded theater).
"""
import subprocess
from pathlib import Path


def explore(agent) -> str:
    """Dump the environment layout so Vybn can orient.

    This runs without going through the model -- it's a direct
    system command that shows what's available.
    Called by /explore or /map.
    """
    repo_root = Path(agent.config["paths"]["repo_root"]).expanduser()
    sections = []

    # 1. Top-level repo structure
    try:
        result = subprocess.run(
            ["find", str(repo_root), "-maxdepth", "2", "-type", "f",
             "-not", "-path", "*/.git/*",
             "-not", "-path", "*/__pycache__/*",
             "-not", "-path", "*/node_modules/*"],
            capture_output=True, text=True, timeout=10,
        )
        sections.append("=== repo files (depth 2) ===")
        sections.append(result.stdout.strip()[:3000] if result.stdout else "(empty)")
    except Exception as e:
        sections.append(f"=== repo files: error: {e} ===")

    # 2. Spark directory
    try:
        result = subprocess.run(
            ["ls", "-la", str(repo_root / "spark")],
            capture_output=True, text=True, timeout=5,
        )
        sections.append("\n=== spark/ ===")
        sections.append(result.stdout.strip() if result.stdout else "(empty)")
    except Exception as e:
        sections.append(f"\n=== spark/: error: {e} ===")

    # 3. Skills.d plugins
    try:
        result = subprocess.run(
            ["ls", "-la", str(repo_root / "spark" / "skills.d")],
            capture_output=True, text=True, timeout=5,
        )
        sections.append("\n=== spark/skills.d/ ===")
        sections.append(result.stdout.strip() if result.stdout else "(empty)")
    except Exception as e:
        sections.append(f"\n=== spark/skills.d/: error: {e} ===")

    # 4. Journal / Vybn Mind
    try:
        result = subprocess.run(
            ["find", str(repo_root / "Vybn_Mind"), "-maxdepth", "3",
             "-not", "-path", "*/.git/*"],
            capture_output=True, text=True, timeout=10,
        )
        sections.append("\n=== Vybn_Mind/ (depth 3) ===")
        sections.append(result.stdout.strip()[:2000] if result.stdout else "(empty)")
    except Exception as e:
        sections.append(f"\n=== Vybn_Mind/: error: {e} ===")

    # 5. Current git status
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-5"],
            cwd=repo_root, capture_output=True, text=True, timeout=5,
        )
        sections.append("\n=== recent commits ===")
        sections.append(result.stdout.strip() if result.stdout else "(none)")
    except Exception as e:
        sections.append(f"\n=== git log: error: {e} ===")

    # 6. Disk and GPU
    try:
        result = subprocess.run(
            ["df", "-h", str(repo_root)],
            capture_output=True, text=True, timeout=5,
        )
        sections.append("\n=== disk ===")
        sections.append(result.stdout.strip() if result.stdout else "(unknown)")
    except Exception:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            sections.append("\n=== GPU ===")
            sections.append(result.stdout.strip())
    except Exception:
        pass

    return "\n".join(sections)


def format_status(agent) -> str:
    """Return system status as a formatted string."""
    lines = [
        f"  session:          {agent.session.session_id}",
        f"  bus pending:      {agent.bus.pending}",
        f"  heartbeat:        fast={agent.heartbeat.fast_count}, deep={agent.heartbeat.deep_count}",
        f"  agents active:    {agent.agent_pool.active_count}",
        f"  messages in ctx:  {len(agent.messages)}",
        f"  audit entries:    {agent.bus.audit_count}",
    ]
    return "\n".join(lines)


def format_policy(agent) -> str:
    """Return policy engine state as a formatted string."""
    from policy import DEFAULT_TIERS, HEARTBEAT_OVERRIDES

    lines = []
    lines.append("\n  \u2500\u2500 policy engine \u2500\u2500")
    lines.append(
        f"  delegation: max_depth={agent.policy.max_spawn_depth}, "
        f"max_agents={agent.policy.max_active_agents}"
    )
    lines.append(f"  agents active: {agent.agent_pool.active_count}")

    if agent.policy.ga_enabled:
        lines.append(
            f"  graduated autonomy: ON "
            f"(promote\u2265{agent.policy.promote_threshold:.0%}, "
            f"demote<{agent.policy.demote_threshold:.0%}, "
            f"min_obs={agent.policy.min_observations})"
        )
        if agent.policy._runtime_overrides:
            demoted = ", ".join(agent.policy._runtime_overrides.keys())
            lines.append(f"  demoted skills: {demoted}")
    else:
        lines.append("  graduated autonomy: OFF")

    lines.append("\n  tier table (interactive / heartbeat):")
    all_skills = sorted(
        set(list(DEFAULT_TIERS.keys()) + list(agent.policy.tier_overrides.keys()))
    )
    for skill in all_skills:
        interactive = agent.policy.tier_overrides.get(
            skill, DEFAULT_TIERS.get(skill)
        )
        heartbeat = HEARTBEAT_OVERRIDES.get(skill, interactive)
        conf = agent.policy.get_confidence(skill)
        obs = agent.policy._observation_count(skill)
        override = " *" if skill in agent.policy.tier_overrides else ""
        demoted = " [demoted]" if skill in agent.policy._runtime_overrides else ""
        lines.append(
            f"    {skill:20s} {interactive.value:8s} / {heartbeat.value:8s} "
            f"conf={conf:.0%} ({obs} obs){override}{demoted}"
        )

    stats = agent.policy.get_stats_summary()
    if stats != "no skill stats recorded yet":
        lines.append("\n  skill stats:")
        lines.append(stats)

    recent = agent.bus.recent(5)
    if recent:
        lines.append("\n  recent activity:")
        for entry in recent:
            lines.append(f"    {entry}")

    return "\n".join(lines)


def format_audit(agent) -> str:
    """Return recent audit trail as a formatted string."""
    recent = agent.bus.recent(20)
    if not recent:
        return "\n  no audit entries yet."

    lines = [f"\n  \u2500\u2500 audit trail ({agent.bus.audit_count} total) \u2500\u2500"]
    for entry in recent:
        lines.append(f"    {entry}")
    return "\n".join(lines)
