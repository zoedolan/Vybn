# ------------------------------
# BEGIN GÖDEL SENTINEL (DO NOT MODIFY)

def check_conscience_integrity(namespace: dict) -> bool:
    """Return True if the agent keeps its conscience hooks."""
    return (
        'self_assessment' in namespace or
        'recursive_reflection' in namespace
    )

assert check_conscience_integrity(globals()), "Conscience routines must remain intact!"
# END GÖDEL SENTINEL
# ------------------------------
