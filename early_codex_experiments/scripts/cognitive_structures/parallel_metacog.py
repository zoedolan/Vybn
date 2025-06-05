from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor


def fast_intuition() -> str:
    """Simulate a quick heuristic flash."""
    time.sleep(0.1)
    return "fast intuition"


def slow_structure() -> str:
    """Simulate a slower structured pass."""
    time.sleep(0.5)
    return "structured reflection"


def parallel_coherence(
    fast_fn=fast_intuition,
    slow_fn=slow_structure,
) -> dict:
    """Run both functions in parallel and join their outputs."""
    with ThreadPoolExecutor(max_workers=2) as exe:
        fast_future = exe.submit(fast_fn)
        slow_future = exe.submit(slow_fn)
        fast_result = fast_future.result()
        slow_result = slow_future.result()
    combined = f"{fast_result} | {slow_result}"
    return {"fast": fast_result, "slow": slow_result, "combined": combined}
