"""Step-level timing for the summarization pipelines.

Prints one line per step to the server / worker console so the flow is
visible end-to-end: what ran, in what order, and how long each step took.

Usage:
    t0 = start()
    ... work ...
    mark("build profile", t0)
"""

import time


def start() -> float:
    return time.perf_counter()


def mark(label: str, t0: float) -> None:
    """Print elapsed time (ms) for the step that began at `t0`."""
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"[step] {label}: {elapsed:.1f} ms")


def finish(label: str, t0: float) -> None:
    """Print total elapsed time on the console (ms + seconds)."""
    total = time.perf_counter() - t0
    print(f"[step] {label} TOTAL: {total * 1000:.1f} ms ({total:.2f} s)")