from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "spark"))
import conveyance


def test_stage_replaces_current_but_preserves_history_and_witness(tmp_path):
    first = tmp_path / "first.html"; first.write_text("<h1>first</h1>")
    second = tmp_path / "second.html"; second.write_text("<h1>second</h1>")
    a = conveyance.stage(first, title="First", thesis="A first change.", state=tmp_path / "state",
                         conveyance_id="first")
    conveyance.witness("first", verdict="received", response="I saw it.", state=tmp_path / "state")
    b = conveyance.stage(second, title="Second", thesis="The process changes.", state=tmp_path / "state",
                         conveyance_id="second")
    records = conveyance.folded(state=tmp_path / "state")
    assert [r["status"] for r in records] == ["received", "awaiting_witness"]
    assert conveyance.current(state=tmp_path / "state")["conveyance_id"] == "second"
    assert conveyance.artifact_for(a["conveyance_id"], state=tmp_path / "state").read_text() == "<h1>first</h1>"
    assert b["artifact_sha256"] != a["artifact_sha256"]
    assert "Second | awaiting witness" in conveyance.wake_status(state=tmp_path / "state")
    conveyance.outcome("second", status="committed", summary="Survived witness.", state=tmp_path / "state")
    assert conveyance.current(state=tmp_path / "state")["status"] == "committed"


def test_artifact_is_immutable_for_an_existing_id(tmp_path):
    state = tmp_path / "state"; page = tmp_path / "page.html"; page.write_text("one")
    conveyance.stage(page, title="One", thesis="one", state=state, conveyance_id="same")
    page.write_text("two")
    try:
        conveyance.stage(page, title="Two", thesis="two", state=state, conveyance_id="same")
    except FileExistsError:
        pass
    else:
        raise AssertionError("an existing conveyance artifact was overwritten")
