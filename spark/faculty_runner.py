"""spark.faculty_runner — Sequential faculty execution with time budgeting.

Runs faculties sequentially (honest about single-GPU serialization)
with early-exit if time budget runs low. Implements Vybn's refinement #6:
don't pretend parallelism.

Budget: 15 min faculties, 5 min governance, 5 min growth, 5 min safety margin.
"""
import json, time, random, logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from spark.faculties import FacultyRegistry, FacultyCard

log = logging.getLogger(__name__)

OUTPUTS_DIR = Path(__file__).resolve().parent / "faculties.d" / "outputs"
FACULTY_TIME_BUDGET_SEC = 15 * 60  # 15 minutes for all faculties


def should_run(card: FacultyCard, state: dict) -> bool:
    """Determine if a faculty should run this breath based on its cadence."""
    cadence = getattr(card, 'breath_cadence', 'every')
    breath_count = state.get('breath_count', 0)

    if cadence == 'every':
        return True
    if cadence.startswith('every_Nth:'):
        n = int(cadence.split(':')[1])
        return breath_count % n == 0
    if cadence.startswith('probability:'):
        p = float(cadence.split(':')[1])
        return random.random() < p
    if cadence == 'on_trigger':
        return False  # requires explicit trigger, not breath-scheduled
    if cadence == '6h_deep':
        # Lightweight every breath, deep every 6h (tracked in state)
        return True
    return True


def write_faculty_output(faculty_id: str, output: dict) -> Path:
    """Write faculty output to the inter-faculty bus."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUTS_DIR / f"{faculty_id}_latest.json"
    output['timestamp'] = datetime.now(timezone.utc).isoformat()
    output['faculty_id'] = faculty_id
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    return path


def read_all_faculty_outputs() -> dict:
    """Read all faculty outputs from the bus. Used by SYNTHESIZER."""
    results = {}
    if not OUTPUTS_DIR.exists():
        return results
    for f in OUTPUTS_DIR.glob("*_latest.json"):
        try:
            results[f.stem.replace('_latest', '')] = json.loads(f.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            continue
    return results


def run_scheduled_faculties(state: dict, registry: FacultyRegistry) -> dict:
    """Run all scheduled faculties sequentially with time budgeting.

    Returns dict of {faculty_id: output_or_error} for governance settlement.
    """
    start = time.monotonic()
    results = {}

    # Always-run faculties first, then scheduled
    always = ['witness', 'self_model']
    scheduled = ['researcher', 'mathematician', 'creator', 'evolver', 'synthesizer']

    for fid in always + scheduled:
        elapsed = time.monotonic() - start
        if elapsed > FACULTY_TIME_BUDGET_SEC:
            log.warning("Time budget exhausted after %.1fs; skipping remaining faculties", elapsed)
            break

        card = registry.get_card(fid)
        if card is None or not card.active:
            continue
        if fid not in always and not should_run(card, state):
            continue

        try:
            # Faculty execution is a placeholder — each faculty module
            # will implement its own run() function
            log.info("Running faculty: %s", fid)
            # output = faculty_modules[fid].run(state)
            # write_faculty_output(fid, output)
            results[fid] = {"status": "placeholder", "note": "faculty module not yet implemented"}
        except Exception as exc:
            log.error("Faculty %s failed: %s", fid, exc)
            results[fid] = {"status": "error", "error": str(exc)}

    return results
