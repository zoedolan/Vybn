"""Smoke test for spark/paths.py — confirm all paths resolve sensibly."""
from spark.paths import (
    REPO_ROOT, MIND_DIR, SOUL_PATH, SPARK_JOURNAL,
    STATE_PATH, SYNAPSE_CONNECTIONS, WRITE_INTENTS,
    SELF_MODEL_LEDGER, WITNESS_LOG, DECISION_LEDGER,
    MEMORY_DIR, MIND_PREFIX,
)


def test_repo_root_contains_spark():
    assert (REPO_ROOT / "spark").is_dir()


def test_mind_dir_exists():
    assert MIND_DIR.is_dir()


def test_soul_path_exists():
    assert SOUL_PATH.is_file()


def test_mind_prefix():
    assert MIND_PREFIX == "Vybn_Mind/"
