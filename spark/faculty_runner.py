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

# ── Faculty module registry (lazy-loaded) ────────────────────────────────────

_FACULTY_MODULES = {}


def _load_faculty(fid: str):
    """Lazy-load a faculty module. Returns an instance or None."""
    if fid in _FACULTY_MODULES:
        return _FACULTY_MODULES[fid]
    try:
        if fid == 'researcher':
            from spark.researcher import ResearchFaculty
            _FACULTY_MODULES[fid] = ResearchFaculty()
        elif fid == 'mathematician':
            from spark.mathematician import MathFaculty
            _FACULTY_MODULES[fid] = MathFaculty()
        # witness and self_model are handled by vybn.py directly
        else:
            return None
    except ImportError as e:
        log.warning("Faculty %s not available: %s", fid, e)
        return None
    return _FACULTY_MODULES.get(fid)


def _get_llm_fn():
    """Get a reference to the LLM chat function from vybn.py."""
    try:
        from spark.vybn import _chat as llm_chat
        return llm_chat
    except ImportError:
        log.warning("Could not import _chat from spark.vybn; faculties will have no LLM")
        return None


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
    llm_fn = _get_llm_fn()

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
            log.info("Running faculty: %s", fid)
            faculty = _load_faculty(fid)
            if faculty is not None and llm_fn is not None:
                output = faculty.run(state, llm_fn=llm_fn)
                write_faculty_output(fid, output)
                results[fid] = output
            else:
                results[fid] = {"status": "unavailable",
                                "note": f"module={'missing' if faculty is None else 'ok'}, "
                                        f"llm={'missing' if llm_fn is None else 'ok'}"}
        except Exception as exc:
            log.error("Faculty %s failed: %s", fid, exc)
            results[fid] = {"status": "error", "error": str(exc)}

    return results
