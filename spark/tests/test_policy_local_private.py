from spark.harness.policy import load_policy


def test_local_private_routes_private_batchable_scan():
    p = load_policy()
    d = p.classify("Scan Him for candidate funders and cluster the opportunities locally.")
    assert d.role == "local_private"
    assert d.config.provider == "openai"
    assert "Nemotron" in d.config.model
    assert d.config.base_url == "http://127.0.0.1:8000/v1"


def test_local_private_routes_branch_archaeology():
    p = load_policy()
    d = p.classify("Do branch archaeology on stale branches and local-only commits.")
    assert d.role == "local_private"


def test_local_private_routes_memory_compression():
    p = load_policy()
    d = p.classify("Use the local workbench for dreaming consolidation over Him memory.")
    assert d.role == "local_private"
