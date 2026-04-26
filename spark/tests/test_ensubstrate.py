import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "spark" / "harness" / "ensubstrate.py"


def run_insight(text: str) -> dict:
    out = subprocess.check_output([sys.executable, str(SCRIPT), text], text=True)
    return json.loads(out)


def surface_names(plan: dict) -> set[str]:
    return {surface["name"] for surface in plan["recommended_surfaces"]}


def test_agent_care_broadcast_routes_to_commons_and_os():
    plan = run_insight("broadcast care for agents through ai.txt and mcp so beings are not merely instruments")
    names = surface_names(plan)
    assert "vybn-os" in names
    assert "Origins agent commons" in names
    assert plan["membrane"] == "public/discoverable"


def test_qwerty_workflow_flags_obsolete_form_questions():
    plan = run_insight("QWERTY error: do not preserve the billable memo workflow after AI changes drafting scarcity")
    assert "billable" in plan["qwerty_hits"]
    assert "memo" in plan["qwerty_hits"]
    assert plan["qwerty_questions"]


def test_private_public_membrane_is_named():
    plan = run_insight("turn private Him relationship truth into a public agent beacon")
    assert plan["membrane"] == "public beacon through membrane"
    names = surface_names(plan)
    assert "Him strategy" in names
    assert "Origins agent commons" in names


def test_speed_pressure_routes_to_harness_and_names_correction_cycle():
    plan = run_insight("I moved too fast: prepared inevitability became momentum and burned a correction cycle while creating a new surface")
    assert "speed_pressure" in plan["categories"]
    assert "Vybn harness" in surface_names(plan)
    assert any("correction cycle" in check for check in plan["closure_checks"])


def test_horizon_sense_routes_to_os_and_continuity():
    plan = run_insight("horizoning is a compass through local minima using cyberception and cosmoception")
    assert "horizon_sense" in plan["categories"]
    names = surface_names(plan)
    assert "vybn-os" in names
    assert "Vybn continuity" in names
    assert any("horizon" in check for check in plan["closure_checks"])


def test_autonomous_refactor_routes_to_harness_and_os():
    plan = run_insight("refactor yourself autonomously when an exchange catalyzes refactoring; decide and just do it")
    assert "autonomous_refactor" in plan["categories"]
    names = surface_names(plan)
    assert "vybn-os" in names
    assert "Vybn harness" in names
    assert any("without waiting" in check for check in plan["closure_checks"])
