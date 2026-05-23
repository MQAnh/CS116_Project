from contextlib import contextmanager
from datetime import datetime
from time import perf_counter


def log_step(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


@contextmanager
def log_time(message):
    log_step(f"START {message}")
    start = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - start
        log_step(f"DONE  {message} ({elapsed:.1f}s)")
