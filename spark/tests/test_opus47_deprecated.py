from pathlib import Path


def test_opus47_is_not_active_in_router_policy_or_aliases():
    policy = Path("spark/harness/policy.py").read_text()
    router = Path("spark/router_policy.yaml").read_text()

    active = policy + "\n" + router
    assert "claude-OPUS47-DEPRECATED" not in active
    assert "@OPUS47-DEPRECATED" not in active
    assert "@OPUS4.7-DEPRECATED" not in active


def test_code_role_uses_opus46_after_opus47_deprecation():
    from spark.harness.policy import default_policy

    decision = default_policy().classify("fix the harness routing bug")
    assert decision.role == "code"
    assert decision.config.model == "claude-opus-4-6"
