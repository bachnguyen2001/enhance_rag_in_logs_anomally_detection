"""Small, visible progress log for sequential LLM experiments."""

import time


class Progress:
    def __init__(self, method, total, every=10):
        self.method = method.upper()
        self.total = total
        self.every = every
        self.started = time.perf_counter()
        self.cache_hits = 0
        self.api_calls = 0
        self.rate_limit_retries = 0
        self.transient_retries = 0

    def before(self, index):
        if index == 1:
            print(f"[{self.method}] query 1/{self.total}: sending first request...", flush=True)

    def after(self, index, row):
        self.cache_hits += int(row.get("cache_hit", False))
        self.api_calls += int(row.get("new_api_calls", 0))
        self.rate_limit_retries += int(row.get("rate_limit_retries", 0))
        self.transient_retries += int(row.get("transient_retries", 0))
        if index % self.every == 0 or index == self.total:
            elapsed = time.perf_counter() - self.started
            contexts = len(row.get("contexts", []))
            print(
                f"[{self.method}] {index}/{self.total} | last={row['prediction']} "
                f"| contexts={contexts} | api_calls={self.api_calls} "
                f"| cache_hits={self.cache_hits} | rate_retries={self.rate_limit_retries} "
                f"| network_retries={self.transient_retries} | elapsed={elapsed:.1f}s",
                flush=True,
            )
