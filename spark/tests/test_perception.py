"""Tests for the dormant Omni-backed perception layer.

Covers:

  * ObservationPacket schema: required fields, persistence record
    omits raw image bytes.
  * Privacy membrane:
      - URL allowlist (only loopback prefixes by default).
      - Filesystem path allowlist (only repo subtrees).
      - Desktop / X11 / framebuffer grabs refused.
      - Hidden reasoning (<think>...</think>) stripped from text
        surfaces and logged in `redactions`.
      - Coarse secret patterns (sk-..., gh{p,o,u,s,r}_..., AWS keys)
        redacted from text surfaces and logged in `redactions`.
  * Provider-shaped message: `image_url` block for image surfaces
    (data URL with the right mime/base64 payload), `text` block for
    text surfaces, and a `[observation: ...]` placeholder when the
    surface is empty.
  * Ordinary chat behavior unchanged: classifying every existing
    fixture turn never routes to the new ``omni`` role, the `@omni`
    alias only pins the model and leaves the role intact, and the
    role/alias are reachable only by explicit caller dispatch.

Run: python3 spark/tests/test_perception.py
"""

from __future__ import annotations

import base64
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

THIS = Path(__file__).resolve()
SPARK_DIR = THIS.parent.parent
sys.path.insert(0, str(SPARK_DIR))

from harness.perception import (  # noqa: E402
    DEFAULT_ALLOWED_ROOTS,
    DEFAULT_ALLOWED_URLS,
    ObservationPacket,
    make_observation,
    provider_message_for_packet,
    redact_for_storage,
)
from harness.policy import default_policy  # noqa: E402


# A minimal but real PNG header so the bytes path classifies as image.
_PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    + b"\x00" * 32
)


class TestPacketSchema(unittest.TestCase):
    def test_text_packet_round_trip(self):
        packet = make_observation(
            trigger="repo_scan",
            target="<memory>",
            text="just a line",
        )
        self.assertEqual(packet.surface_kind, "text")
        self.assertEqual(packet.surface_text, "just a line")
        self.assertEqual(packet.bytes_len, len("just a line"))
        # sha256 over the cleaned text — deterministic.
        self.assertEqual(len(packet.sha256), 64)
        self.assertEqual(packet.redactions, [])

    def test_record_omits_raw_image_bytes(self):
        packet = make_observation(
            trigger="portal_screenshot",
            image_bytes=_PNG_BYTES,
            image_mime="image/png",
        )
        self.assertEqual(packet.surface_kind, "image")
        self.assertEqual(packet.mime, "image/png")
        # In-memory base64 is populated.
        self.assertTrue(packet.surface_b64)
        # But to_record / redact_for_storage must drop it.
        rec = packet.to_record()
        self.assertNotIn("surface_b64", rec)
        self.assertEqual(rec, redact_for_storage(packet))
        # Digest + length still survive so audit logs can verify.
        self.assertEqual(rec["sha256"], packet.sha256)
        self.assertEqual(rec["bytes_len"], len(_PNG_BYTES))

    def test_oversized_image_rejected(self):
        big = b"\x89PNG\r\n\x1a\n" + b"\x00" * (5 * 1024 * 1024)
        packet = make_observation(
            trigger="huge_screenshot",
            image_bytes=big,
            image_mime="image/png",
        )
        self.assertEqual(packet.surface_kind, "empty")
        self.assertIn("image_too_large", packet.redactions)

    def test_unknown_image_mime_rejected(self):
        packet = make_observation(
            trigger="weird_screenshot",
            image_bytes=_PNG_BYTES,
            image_mime="image/tiff",
        )
        self.assertEqual(packet.surface_kind, "empty")
        self.assertIn("image_mime_refused", packet.redactions)


class TestAllowlistAndRedaction(unittest.TestCase):
    def test_url_allowlist_blocks_arbitrary_host(self):
        packet = make_observation(
            trigger="external_fetch",
            target="https://example.com/secret",
        )
        self.assertEqual(packet.surface_kind, "empty")
        self.assertIn("url_not_allowlisted", packet.redactions)

    def test_url_allowlist_passes_loopback(self):
        # Loopback portal is allowed; the membrane stages the URL but
        # does not actually fetch (library-only by design).
        loopback = next(iter(DEFAULT_ALLOWED_URLS))
        packet = make_observation(
            trigger="portal_health",
            target=loopback + "health",
        )
        self.assertEqual(packet.surface_kind, "empty")
        self.assertEqual(packet.redactions, [])
        self.assertIn("staged", packet.note)

    def test_path_outside_repo_refused(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, dir="/tmp"
        ) as fh:
            fh.write("outside the membrane")
            outside = fh.name
        try:
            packet = make_observation(
                trigger="random_path",
                target=outside,
            )
            self.assertEqual(packet.surface_kind, "empty")
            self.assertIn("path_not_allowlisted", packet.redactions)
        finally:
            os.unlink(outside)

    def test_path_inside_repo_succeeds(self):
        # spark/router_policy.yaml is always inside DEFAULT_ALLOWED_ROOTS.
        target = SPARK_DIR / "router_policy.yaml"
        packet = make_observation(
            trigger="policy_surface",
            target=str(target),
        )
        self.assertEqual(packet.surface_kind, "text")
        self.assertGreater(packet.bytes_len, 0)
        self.assertEqual(packet.redactions, [])

    def test_caller_supplied_root_extends_allowlist(self):
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = Path(tmp).resolve()
            (sandbox / "note.txt").write_text("hello", encoding="utf-8")
            packet = make_observation(
                trigger="sandbox_scan",
                target=str(sandbox / "note.txt"),
                allowed_roots=[sandbox],
            )
        self.assertEqual(packet.surface_kind, "text")
        self.assertEqual(packet.surface_text, "hello")

    def test_display_grab_refused(self):
        for tgt in (":0", "display:0", "x11:0", "screen", "/dev/fb0"):
            packet = make_observation(trigger="oh_no", target=tgt)
            self.assertEqual(
                packet.surface_kind, "empty", msg=f"target {tgt!r}"
            )
            self.assertIn("display_grab_refused", packet.redactions)

    def test_hidden_reasoning_stripped_and_logged(self):
        packet = make_observation(
            trigger="text_surface",
            text="<think>secret scratchpad</think>visible reply",
        )
        self.assertEqual(packet.surface_kind, "text")
        self.assertEqual(packet.surface_text, "visible reply")
        self.assertIn("hidden_reasoning", packet.redactions)

    def test_secret_patterns_redacted(self):
        leaky = (
            "openai sk-abcdefghij1234567890ABCDEFG and "
            "github ghp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa1 and "
            "aws AKIAABCDEFGHIJKLMNOP "
        )
        packet = make_observation(trigger="leaky_text", text=leaky)
        self.assertEqual(packet.surface_kind, "text")
        # No raw secret survives.
        self.assertNotIn("sk-abcdefghij", packet.surface_text)
        self.assertNotIn("ghp_aaaaaaaa", packet.surface_text)
        self.assertNotIn("AKIAABCDEFGH", packet.surface_text)
        # Markers landed.
        self.assertIn("[redacted:openai_api_key]", packet.surface_text)
        self.assertIn("[redacted:github_token]", packet.surface_text)
        self.assertIn("[redacted:aws_access_key]", packet.surface_text)
        # Audit fields name what was redacted.
        self.assertIn("openai_api_key", packet.redactions)
        self.assertIn("github_token", packet.redactions)
        self.assertIn("aws_access_key", packet.redactions)


class TestProviderMessageShape(unittest.TestCase):
    def test_image_packet_yields_image_url_part(self):
        packet = make_observation(
            trigger="portal_png",
            image_bytes=_PNG_BYTES,
            image_mime="image/png",
        )
        msg = provider_message_for_packet(packet, prompt="describe what you see")
        self.assertEqual(msg["role"], "user")
        parts = msg["content"]
        # text prompt + image_url part
        self.assertEqual(parts[0], {"type": "text", "text": "describe what you see"})
        img = parts[1]
        self.assertEqual(img["type"], "image_url")
        url = img["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        # Round-trips back to the original bytes.
        b64 = url.split(",", 1)[1]
        self.assertEqual(base64.b64decode(b64), _PNG_BYTES)

    def test_text_packet_yields_text_part(self):
        packet = make_observation(
            trigger="repo_scan",
            text="readme line",
        )
        msg = provider_message_for_packet(packet)
        self.assertEqual(msg["role"], "user")
        self.assertEqual(msg["content"], [
            {"type": "text", "text": "readme line"},
        ])

    def test_empty_packet_yields_observation_placeholder(self):
        packet = make_observation(trigger="nothing", target=None)
        msg = provider_message_for_packet(packet)
        # Single text part naming the empty observation.
        self.assertEqual(msg["role"], "user")
        self.assertEqual(len(msg["content"]), 1)
        self.assertEqual(msg["content"][0]["type"], "text")
        self.assertTrue(msg["content"][0]["text"].startswith("[observation:"))

    def test_provider_message_does_not_carry_hidden_reasoning(self):
        packet = make_observation(
            trigger="text_surface",
            text="<think>private</think>public",
        )
        msg = provider_message_for_packet(packet)
        rendered = str(msg)
        self.assertNotIn("<think>", rendered)
        self.assertNotIn("private", rendered)
        self.assertIn("public", rendered)


class TestOmniRoleIsDormant(unittest.TestCase):
    """The dormant role / alias must not change ordinary chat routing."""

    def setUp(self):
        self.policy = default_policy()

    def test_omni_role_exists_but_no_directive(self):
        self.assertIn("omni", self.policy.roles)
        # No /omni directive — explicit by design.
        self.assertNotIn("/omni", self.policy.directives)

    def test_omni_not_in_heuristics(self):
        # No regex routes into the omni role.
        self.assertNotIn("omni", self.policy.heuristics)

    def test_omni_not_in_fallback_chain(self):
        # No model falls through to the omni model.
        omni_model = self.policy.roles["omni"].model
        for src, chain in self.policy.fallback_chain.items():
            self.assertNotIn(omni_model, chain, msg=f"{src} → {chain}")

    def test_typical_chat_turns_never_route_to_omni(self):
        # Sample fixtures spanning every existing routing path. None
        # should now land on the dormant role.
        for turn in (
            "hi",
            "hey buddy",
            "which model are you?",
            "fix the harness bug",
            "brainstorm a name",
            "let's plan a refactor",
            "ok",
            "is everything working",
            "thoughts on this?",
            "summarize the corpus",
        ):
            decision = self.policy.classify(turn)
            self.assertNotEqual(
                decision.role, "omni",
                msg=f"turn {turn!r} surprisingly routed to omni",
            )

    def test_omni_alias_pins_model_only(self):
        # @omni-prefixed turn keeps its underlying role and only swaps
        # the model id. This mirrors @nemotron / @sonnet behavior.
        decision = self.policy.classify("@omni hey")
        self.assertEqual(decision.alias_used, "@omni")
        self.assertNotEqual(decision.role, "omni")
        # The pinned model lives under the omni role config.
        self.assertEqual(
            decision.model_override,
            self.policy.roles["omni"].model,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
