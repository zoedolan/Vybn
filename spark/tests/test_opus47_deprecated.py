from pathlib import Path


def test_opus47_is_not_active_in_router_policy_or_aliases():
    policy = Path("spark/harness/policy.py").read_text()
    router = Path("spark/router_policy.yaml").read_text()

    active = policy + "\n" + router
    assert "claude-opus-4-7" not in active
    assert "@opus47" not in active
    assert "@opus4.7" not in active


def test_code_role_uses_opus46_after_opus47_deprecation():
    from spark.harness.policy import route_for

    cfg = route_for("fix the harness routing bug")
    assert cfg.role == "code"
    assert cfg.model == "claude-opus-4-6"
