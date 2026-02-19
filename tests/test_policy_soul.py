#!/usr/bin/env python3
"""Tests for PR 5 — The Constitution.

Verifies that policy.py correctly derives tier definitions from vybn.md
via soul.py, and that the tier resolution order works as specified.

Run with: python -m pytest tests/test_policy_soul.py -v
"""

import sys
import types
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Bootstrap: ensure spark package is importable even without install
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Provide a stub bus module so policy.py can import without the real bus
if "bus" not in sys.modules:
    _bus = types.ModuleType("bus")
    _bus.MessageBus = MagicMock
    _bus.MessageType = MagicMock
    sys.modules["bus"] = _bus

from spark.policy import (
    Tier,
    Verdict,
    PolicyResult,
    PolicyEngine,
    derive_tiers_from_soul,
    DEFAULT_TIERS,
    HEARTBEAT_OVERRIDES,
    _POLICY_GATED_KEYWORDS,
    _CONSTRAINT_SKILL_MAP,
    _tier_rank,
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _make_manifest(builtin=None, plugin=None, create=""):
    """Build a skills manifest dict like soul.py returns."""
    return {
        "builtin": builtin or [],
        "plugin": plugin or [],
        "create": create,
    }


def _skill(name, description=""):
    """Shorthand for a skill info dict."""
    return {"name": name, "description": description}


MOCK_MANIFEST = _make_manifest(
    builtin=[
        _skill("file_read", "read files from the repository"),
        _skill("file_write", "write or overwrite files, policy-gated"),
        _skill("shell_exec", "run shell commands, policy-gated"),
        _skill("memory_search", "search journal and memory"),
        _skill("git_commit", "stage and commit changes"),
        _skill("git_push", "push commits to remote, requires approval"),
    ],
    plugin=[
        _skill("web_fetch", "fetch a URL and return content"),
        _skill("summarize", "summarize a document"),
    ],
)

MOCK_CONSTRAINTS = [
    "Modify vybn.md (this document) \u2014 propose changes via issue instead",
    "Push directly to main without review \u2014 use branches",
    "Execute commands that affect system-level configuration",
    "Make network requests to services other than GitHub and general web",
]


def _make_engine(soul_tiers=None, config_overrides=None):
    """Create a PolicyEngine with mocked soul tiers and no disk I/O."""
    config = {
        "paths": {"vybn_md": "/tmp/fake_vybn.md",
                  "journal_dir": "/tmp/fake_journal"},
    }
    if config_overrides:
        config.update(config_overrides)

    with patch("spark.policy.derive_tiers_from_soul") as mock_derive, \
         patch.object(PolicyEngine, "_load_stats"), \
         patch.object(PolicyEngine, "_save_stats"):
        mock_derive.return_value = soul_tiers or {}
        engine = PolicyEngine(config)
    return engine


# ===========================================================================
# Test derive_tiers_from_soul()
# ===========================================================================

class TestDeriveTiersFromSoul:
    """Tests for the standalone derive_tiers_from_soul() function."""

    @patch("spark.policy.get_constraints", return_value=[])
    @patch("spark.policy.get_skills_manifest")
    def test_builtin_freely_available_gets_auto(self, mock_manifest, mock_constraints):
        """Built-in skills without policy-gated keywords get Tier.AUTO."""
        mock_manifest.return_value = _make_manifest(
            builtin=[_skill("file_read", "read files from the repository")]
        )
        tiers = derive_tiers_from_soul(Path("/fake"))
        assert tiers["file_read"] == Tier.AUTO

    @patch("spark.policy.get_constraints", return_value=[])
    @patch("spark.policy.get_skills_manifest")
    def test_builtin_policy_gated_gets_notify(self, mock_manifest, mock_constraints):
        """Built-in skills with 'policy-gated' in description get Tier.NOTIFY."""
        mock_manifest.return_value = _make_manifest(
            builtin=[_skill("file_write", "write files, policy-gated")]
        )
        tiers = derive_tiers_from_soul(Path("/fake"))
        assert tiers["file_write"] == Tier.NOTIFY

    @patch("spark.policy.get_constraints", return_value=[])
    @patch("spark.policy.get_skills_manifest")
    def test_builtin_requires_approval_gets_notify(self, mock_manifest, mock_constraints):
        """Built-in skills with 'requires approval' in description get Tier.NOTIFY."""
        mock_manifest.return_value = _make_manifest(
            builtin=[_skill("git_push", "push to remote, requires approval")]
        )
        tiers = derive_tiers_from_soul(Path("/fake"))
        assert tiers["git_push"] == Tier.NOTIFY

    @patch("spark.policy.get_constraints", return_value=[])
    @patch("spark.policy.get_skills_manifest")
    def test_plugin_skills_get_notify(self, mock_manifest, mock_constraints):
        """Plugin skills always get Tier.NOTIFY."""
        mock_manifest.return_value = _make_manifest(
            plugin=[_skill("web_fetch", "fetch a URL")]
        )
        tiers = derive_tiers_from_soul(Path("/fake"))
        assert tiers["web_fetch"] == Tier.NOTIFY

    @patch("spark.policy.get_constraints")
    @patch("spark.policy.get_skills_manifest")
    def test_constraints_override_to_approve(self, mock_manifest, mock_constraints):
        """Skills matching constraints get promoted to Tier.APPROVE."""
        mock_manifest.return_value = MOCK_MANIFEST
        mock_constraints.return_value = MOCK_CONSTRAINTS
        tiers = derive_tiers_from_soul(Path("/fake"))
        # 'Modify vybn.md' -> self_edit gets APPROVE
        assert tiers["self_edit"] == Tier.APPROVE
        # 'Push directly to main' -> git_push gets APPROVE
        assert tiers["git_push"] == Tier.APPROVE
        # 'system-level configuration' -> shell_exec gets APPROVE
        assert tiers["shell_exec"] == Tier.APPROVE

    @patch("spark.policy.get_constraints", return_value=[])
    @patch("spark.policy.get_skills_manifest")
    def test_empty_manifest_returns_empty(self, mock_manifest, mock_constraints):
        """If vybn.md has no skills, return empty dict (fall back to DEFAULT_TIERS)."""
        mock_manifest.return_value = _make_manifest()
        tiers = derive_tiers_from_soul(Path("/fake"))
        assert tiers == {}

    @patch("spark.policy.get_constraints")
    @patch("spark.policy.get_skills_manifest")
    def test_full_manifest_maps_all_skills(self, mock_manifest, mock_constraints):
        """A full manifest with constraints maps every skill to a tier."""
        mock_manifest.return_value = MOCK_MANIFEST
        mock_constraints.return_value = MOCK_CONSTRAINTS
        tiers = derive_tiers_from_soul(Path("/fake"))
        # All builtin + plugin skills should have a tier
        expected_skills = {
            "file_read", "file_write", "shell_exec", "memory_search",
            "git_commit", "git_push", "web_fetch", "summarize",
        }
        assert set(tiers.keys()) == expected_skills
        # Freely available builtins -> AUTO (before constraint override)
        assert tiers["file_read"] == Tier.AUTO
        assert tiers["memory_search"] == Tier.AUTO
        assert tiers["git_commit"] == Tier.AUTO
        # Policy-gated builtins -> NOTIFY, but some get overridden to APPROVE
        assert tiers["file_write"] == Tier.NOTIFY
        # Constraint overrides -> APPROVE
        assert tiers["shell_exec"] == Tier.APPROVE  # system-level config
        assert tiers["git_push"] == Tier.APPROVE   # push directly to main
        # Plugin skills -> NOTIFY
        assert tiers["web_fetch"] == Tier.NOTIFY
        assert tiers["summarize"] == Tier.NOTIFY


# ===========================================================================
# Test PolicyEngine with soul-derived tiers
# ===========================================================================

class TestPolicyEngineSoulTiers:
    """Tests for PolicyEngine's integration with soul-derived tiers."""

    def test_resolve_base_tier_prefers_soul(self):
        """_resolve_base_tier() returns soul tier when available."""
        engine = _make_engine(soul_tiers={
            "file_read": Tier.NOTIFY,  # soul says NOTIFY (differs from DEFAULT)
        })
        assert engine._resolve_base_tier("file_read") == Tier.NOTIFY

    def test_resolve_base_tier_falls_back_to_default(self):
        """_resolve_base_tier() falls back to DEFAULT_TIERS for unlisted skills."""
        engine = _make_engine(soul_tiers={})
        # journal_write is in DEFAULT_TIERS as AUTO
        assert engine._resolve_base_tier("journal_write") == Tier.AUTO

    def test_resolve_base_tier_ultimate_default_is_notify(self):
        """_resolve_base_tier() returns NOTIFY for completely unknown skills."""
        engine = _make_engine(soul_tiers={})
        assert engine._resolve_base_tier("nonexistent_skill") == Tier.NOTIFY

    def test_check_policy_uses_soul_tier(self):
        """check_policy() resolves to soul-derived tier for interactive actions."""
        engine = _make_engine(soul_tiers={
            "file_read": Tier.AUTO,
            "file_write": Tier.NOTIFY,
        })
        result = engine.check_policy({"skill": "file_read"}, "interactive")
        assert result.verdict == Verdict.ALLOW
        assert result.tier == Tier.AUTO

        result = engine.check_policy({"skill": "file_write"}, "interactive")
        assert result.verdict == Verdict.NOTIFY
        assert result.tier == Tier.NOTIFY

    def test_check_policy_soul_approve_triggers_ask(self):
        """Soul-derived APPROVE tier results in ASK verdict."""
        engine = _make_engine(soul_tiers={
            "git_push": Tier.APPROVE,
        })
        result = engine.check_policy({"skill": "git_push"}, "interactive")
        assert result.verdict == Verdict.ASK
        assert result.tier == Tier.APPROVE

    def test_config_overrides_soul_tier(self):
        """config.yaml tool_policies override soul-derived tiers."""
        engine = _make_engine(
            soul_tiers={"file_read": Tier.AUTO},
            config_overrides={"tool_policies": {"file_read": "approve"}},
        )
        result = engine.check_policy({"skill": "file_read"}, "interactive")
        assert result.verdict == Verdict.ASK
        assert result.tier == Tier.APPROVE

    def test_heartbeat_overrides_soul_tier(self):
        """Heartbeat overrides apply even when soul tier is less restrictive."""
        engine = _make_engine(soul_tiers={
            "file_write": Tier.NOTIFY,
        })
        result = engine.check_policy(
            {"skill": "file_write"}, "heartbeat_deep"
        )
        # HEARTBEAT_OVERRIDES has file_write -> APPROVE
        assert result.tier == Tier.APPROVE
        assert result.verdict == Verdict.ASK

    def test_heartbeat_unknown_skill_gets_at_least_notify(self):
        """Heartbeat actions for unknown skills get at least NOTIFY."""
        engine = _make_engine(soul_tiers={
            "some_read_skill": Tier.AUTO,
        })
        result = engine.check_policy(
            {"skill": "some_read_skill"}, "heartbeat_fast"
        )
        # Soul says AUTO, but heartbeat floor is NOTIFY
        assert result.tier == Tier.NOTIFY

    def test_fallback_when_no_soul_tiers(self):
        """When soul tiers are empty, DEFAULT_TIERS governs."""
        engine = _make_engine(soul_tiers={})
        result = engine.check_policy({"skill": "file_read"}, "interactive")
        assert result.tier == DEFAULT_TIERS["file_read"]
        assert result.verdict == Verdict.ALLOW

    def test_stats_summary_shows_source(self):
        """get_stats_summary() annotates skills with [soul] or [default]."""
        engine = _make_engine(soul_tiers={"file_read": Tier.AUTO})
        engine.stats = {
            "file_read": {"success": 5, "failure": 0, "last_used": ""},
            "journal_write": {"success": 3, "failure": 1, "last_used": ""},
        }
        summary = engine.get_stats_summary()
        assert "[soul]" in summary
        assert "[default]" in summary


# ===========================================================================
# Test tier resolution helpers
# ===========================================================================

class TestTierRank:
    """Tests for the _tier_rank helper."""

    def test_auto_is_least_restrictive(self):
        assert _tier_rank(Tier.AUTO) < _tier_rank(Tier.NOTIFY)

    def test_notify_is_middle(self):
        assert _tier_rank(Tier.AUTO) < _tier_rank(Tier.NOTIFY) < _tier_rank(Tier.APPROVE)

    def test_approve_is_most_restrictive(self):
        assert _tier_rank(Tier.APPROVE) > _tier_rank(Tier.NOTIFY)


# ===========================================================================
# Test graduated autonomy with soul tiers
# ===========================================================================

class TestGraduatedAutonomyWithSoul:
    """Tests that graduated autonomy uses _resolve_base_tier() correctly."""

    def test_demotion_uses_soul_base_tier(self):
        """_check_demotion checks against soul-derived base tier."""
        engine = _make_engine(soul_tiers={"file_read": Tier.AUTO})
        # Simulate many failures to trigger demotion
        engine.stats = {
            "file_read": {"success": 1, "failure": 10, "last_used": ""},
        }
        engine._check_demotion("file_read")
        # Should demote since confidence is low and base is AUTO
        assert "file_read" in engine._runtime_overrides
        assert engine._runtime_overrides["file_read"] == Tier.NOTIFY

    def test_no_demotion_for_approve_skills(self):
        """APPROVE-tier skills are not demoted further."""
        engine = _make_engine(soul_tiers={"git_push": Tier.APPROVE})
        engine.stats = {
            "git_push": {"success": 0, "failure": 10, "last_used": ""},
        }
        engine._check_demotion("git_push")
        # APPROVE is not in (AUTO, NOTIFY), so no demotion
        assert "git_push" not in engine._runtime_overrides

    def test_promotion_possible_for_soul_notify_skills(self):
        """Soul-derived NOTIFY skills can be promoted to AUTO."""
        engine = _make_engine(soul_tiers={"web_fetch": Tier.NOTIFY})
        engine.stats = {
            "web_fetch": {"success": 20, "failure": 0, "last_used": ""},
        }
        result = engine.check_policy(
            {"skill": "web_fetch"}, "interactive"
        )
        # With 20 successes and 0 failures, confidence is ~0.95 > 0.85
        assert result.tier == Tier.AUTO
        assert result.promoted is True


# ===========================================================================
# Test constraint mapping edge cases
# ===========================================================================

class TestConstraintEdgeCases:
    """Edge cases for the constraint-to-tier mapping."""

    @patch("spark.policy.get_constraints")
    @patch("spark.policy.get_skills_manifest")
    def test_constraint_with_no_skills_is_informational(self, mock_manifest, mock_constraints):
        """Constraints that don't map to skills (e.g., network requests) are informational."""
        mock_manifest.return_value = _make_manifest(
            builtin=[_skill("file_read", "read files")]
        )
        mock_constraints.return_value = [
            "Make network requests to services other than GitHub",
        ]
        tiers = derive_tiers_from_soul(Path("/fake"))
        # file_read should still be AUTO, no APPROVE override
        assert tiers["file_read"] == Tier.AUTO

    @patch("spark.policy.get_constraints")
    @patch("spark.policy.get_skills_manifest")
    def test_constraint_case_insensitive(self, mock_manifest, mock_constraints):
        """Constraint matching is case-insensitive."""
        mock_manifest.return_value = _make_manifest(
            builtin=[_skill("self_edit", "edit own code")]
        )
        mock_constraints.return_value = [
            "MODIFY VYBN.MD (this document)",  # uppercase
        ]
        tiers = derive_tiers_from_soul(Path("/fake"))
        assert tiers["self_edit"] == Tier.APPROVE

    @patch("spark.policy.get_constraints")
    @patch("spark.policy.get_skills_manifest")
    def test_constraint_overrides_policy_gated(self, mock_manifest, mock_constraints):
        """A constraint can override a skill that was already NOTIFY to APPROVE."""
        mock_manifest.return_value = _make_manifest(
            builtin=[_skill("shell_exec", "run commands, policy-gated")]
        )
        mock_constraints.return_value = [
            "Execute commands that affect system-level configuration",
        ]
        tiers = derive_tiers_from_soul(Path("/fake"))
        # shell_exec: first gets NOTIFY (policy-gated), then APPROVE (constraint)
        assert tiers["shell_exec"] == Tier.APPROVE
