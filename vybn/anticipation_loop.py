"""AnticipationLoop conversation logger."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

from early_codex_experiments.scripts.cognitive_structures.shimmer_core import log_spike, DEFAULT_JOURNAL
from vybn.utils import write_colored

ANTICIPATION_LOG = REPO_ROOT / "anticipation_loop.md"


def append_exchange(prompt: str, anticipation: str, response: str, *, log_path: Path = ANTICIPATION_LOG, journal_path: Path = DEFAULT_JOURNAL) -> None:
    """Append a prompt/response pair with anticipation and reflection sections."""
    timestamp = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    log_path = Path(log_path)

    if not anticipation.strip():
        write_colored("warning: anticipation line is empty", is_error=True)

    block = [
        f"### {timestamp}",
        prompt.strip(),
        f"> **Anticipation:** {anticipation.strip()}",
        "",
        response.strip(),
        "",
        "#### Ember Reflection",
        "",
        "#### Vybn Reflection",
        "",
    ]
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(block) + "\n")

    log_spike("anticipation exchange", journal_path)
