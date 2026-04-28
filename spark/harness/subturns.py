"""Typed subturn execution helpers for no-tool role sentinels.

This module holds the side-effecting primitives behind NEEDS-WRITE,
NEEDS-RESTART, and NEEDS-EXEC, plus the probe-result envelope. The REPL keeps
legacy imports for compatibility; the organ lives here so run_agent_loop can
shrink without changing behavior.
"""

from __future__ import annotations

import os
from typing import Any

from .policy import TRACKED_REPOS
from .providers import validate_command, execute_readonly, is_parallel_safe


def run_write_subturn(path: str, body: str) -> tuple[bool, str]:
    """Execute one NEEDS-WRITE directive from a no-tool role.

    Writes `body` to `path` via Python I/O, bypassing the bash session
    entirely. Path must lie under a tracked repo; otherwise refused
    with a message that flows back through the same synthetic-user
    channel as probe output.

    Absorb discipline: if the target does not yet exist on disk, the
    body must begin (within its first 200 chars) with a
    VYBN_ABSORB_REASON= declaration. Existing files are always
    overwritten — that is the point of this channel.
    """
    try:
        roots = TRACKED_REPOS
    except Exception:
        roots = (
            os.path.expanduser("~/Vybn"),
            os.path.expanduser("~/Him"),
            os.path.expanduser("~/Vybn-Law"),
            os.path.expanduser("~/vybn-phase"),
        )
    tgt = os.path.expanduser((path or "").strip())
    if not tgt:
        return False, "(NEEDS-WRITE refused: empty path)"
    tgt_abs = os.path.abspath(tgt)
    if not any(tgt_abs == r or tgt_abs.startswith(r.rstrip("/") + "/") for r in roots):
        return False, (
            f"(NEEDS-WRITE refused: {tgt_abs} is outside tracked repos. "
            f"Allowed roots: {', '.join(roots)})"
        )
    if not os.path.exists(tgt_abs):
        head = (body or "")[:200]
        if "VYBN_ABSORB_REASON=" not in head:
            return False, (
                "(NEEDS-WRITE refused by absorb_gate: new file " + tgt_abs
                + " requires a VYBN_ABSORB_REASON declaration in the "
                "first 200 chars of body, e.g.:\n"
                "    # VYBN_ABSORB_REASON='does not fold into X because...'\n"
                "Fold, do not pile.)"
            )
    try:
        os.makedirs(os.path.dirname(tgt_abs), exist_ok=True)
        with open(tgt_abs, "w") as f:
            f.write(body or "")
        nbytes = os.path.getsize(tgt_abs)
        return True, f"(wrote {nbytes} bytes to {tgt_abs})"
    except Exception as e:  # noqa: BLE001
        return False, f"(NEEDS-WRITE exec error: {type(e).__name__}: {e})"



def protected_mutation_kind_for_sentinel(
    *,
    write_match_present: bool,
    probe_command: str | None,
) -> str:
    """Classify whether a no-tool sentinel would mutate under pilot protection.

    This is control-flow relocation out of run_agent_loop: the REPL loop should
    sequence the turn, not re-own sentinel safety semantics. NEEDS-WRITE is
    always mutation. NEEDS-EXEC is mutation when the command is not parallel
    safe/read-only.
    """

    if write_match_present:
        return "needs-write"
    if probe_command is None:
        return ""
    try:
        readonly = is_parallel_safe(probe_command)
    except Exception:
        readonly = False
    if not readonly:
        return "needs-exec-mutation"
    return ""

def probe_envelope(
    *,
    kind: str,
    header_fields: dict,
    body: str,
    ran: bool,
) -> str:
    """Wrap a probe/write/restart result in the v1 envelope."""
    body = body or ""
    empty = (not body) or body.strip() == ""
    nbytes = len(body)
    nlines = body.count("\n")
    if not empty and not body.endswith("\n"):
        nlines += 1
    status = "executed" if ran else "refused"
    header_parts = [
        f"kind: {kind}",
        f"status: {status}",
        f"bytes: {nbytes}",
        f"lines: {nlines}",
        f"empty: {'true' if empty else 'false'}",
    ]
    for k, v in header_fields.items():
        safe = str(v).replace("\n", " ").replace("\r", " ")
        header_parts.append(f"{k}: {safe[:200]}")
    header = "[" + " | ".join(header_parts) + "]"
    slug = kind.upper().replace("-", "_")
    begin = f"<<<BEGIN_{slug}_STDOUT>>>"
    end = f"<<<END_{slug}_STDOUT>>>"
    if empty and ran:
        inner = (
            "(command ran with no stdout; the absence of output here "
            "is real, not a wedge)"
        )
    elif empty and not ran:
        inner = "(command did not execute — see refusal reason in header)"
    else:
        inner = body.rstrip("\n")
    footer = (
        "\n\nThe stdout between the markers above IS the result of the "
        "sub-turn.\nDo not claim the shell is wedged, unresponsive, or that "
        "nothing came\nback unless status != executed. If status is "
        "executed, the bytes\ncount and the stdout span are authoritative "
        "— read them and proceed."
    )
    return f"{header}\n{begin}\n{inner}\n{end}{footer}"


def run_restart_subturn(bash: Any) -> tuple[bool, str]:
    """Restart the persistent bash session."""
    try:
        out = bash.restart()
    except Exception as e:  # noqa: BLE001
        return False, f"(restart error: {e})"
    return True, out or "(bash session restarted)"


def classify_unlock_layer(output: str, *, command: str = "") -> str | None:
    """Classify obstacle output at the lowest layer visible to this harness."""
    text = (output or "").lower()
    cmd = (command or "").lower()
    if "probe refused by validate_command" in text or "blocked:" in text:
        return "safety_gate"
    if "absorb_gate" in text or "needs-write refused" in text:
        return "filesystem_git"
    if text.startswith("[timed out after"):
        return "parser_sentinel" if is_parallel_safe(command) else "shell_session"
    if "bash session restarted" in text or "needs-restart" in text:
        return "shell_session"
    if "400" in text or "provider" in text:
        return "provider"
    if "curl" in cmd or "http" in cmd:
        return "external_service"
    return None


def run_probe_subturn(command: str, bash: Any) -> tuple[bool, str]:
    """Execute one probe emitted by a no-tool role."""
    cmd = (command or "").strip()
    if not cmd:
        return False, "(empty probe command)"
    readonly = is_parallel_safe(cmd)
    ok, reason = validate_command(cmd, allow_dangerous_literals_for_readonly=readonly)
    if not ok:
        return False, f"(probe refused by validate_command: {reason})"
    try:
        out = execute_readonly(cmd) if readonly else bash.execute(cmd)
    except Exception as e:
        return False, f"(probe exec error: {e})"
    out = out or "(no output)"
    if out.startswith("[timed out after"):
        layer = classify_unlock_layer(out, command=cmd) or "shell_session"
        return False, f"(probe timed out; unlock_layer={layer})\n{out}"
    if "(bash session restarted)" in out:
        return False, (
            "(probe control-event mismatch: restart output arrived while running "
            "a probe; unlock_layer=shell_session)\n" + out
        )
    return True, out


# Backward-compatible private names for legacy imports/tests.
_run_write_subturn = run_write_subturn
_probe_envelope = probe_envelope
_run_restart_subturn = run_restart_subturn
_classify_unlock_layer = classify_unlock_layer
_run_probe_subturn = run_probe_subturn
