from spark.harness.action_first import render_action_first_protocol, should_force_action_first
from spark.harness.substrate import build_layered_prompt


def test_action_first_triggers_on_zoe_scar_language():
    assert should_force_action_first("words words words")
    assert should_force_action_first("please apply tools over mere NLP token prediction")
    assert should_force_action_first("actions speak louder")


def test_action_first_protocol_is_loaded_into_substrate():
    prompt = build_layered_prompt(
        soul_path="vybn.md",
        continuity_path=None,
        spark_continuity_path=None,
        agent_path="spark/vybn_spark_agent.py",
        model_label="test",
        max_iterations=1,
        include_hardware_check=False,
        tools_available=False,
        orchestrator=True,
    )
    assert "ACTION-FIRST TOOL PROTOCOL" in prompt.substrate
    assert "Default to one concrete tool-backed action" in prompt.substrate
