"""
Automated Benchmarking Harness for Cloud TPU LLM Serving.
"""

import time
import statistics
import concurrent.futures
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from tabulate import tabulate

from .client import TPUInferenceClient


@dataclass
class BenchmarkMetrics:
    """Consolidated performance metrics for a test run."""
    concurrency: int
    total_requests: int
    successful_requests: int
    total_time_seconds: float
    requests_per_second: float
    estimated_tokens_per_second: float
    latency_mean_sec: float
    latency_p50_sec: float
    latency_p90_sec: float
    latency_p95_sec: float
    latency_p99_sec: float


class TPUBenchmarkRunner:
    """Evaluates throughput, latency percentiles, and concurrency scaling on TPU endpoints."""

    def __init__(self, client: TPUInferenceClient):
        self.client = client

    def run_benchmark(
        self,
        test_prompts: List[str],
        concurrency_levels: List[int] = [1, 2, 4, 8],
        max_output_tokens: int = 128,
        temperature: float = 0.0,
    ) -> List[BenchmarkMetrics]:
        """Execute benchmarking matrix across variable concurrency levels."""
        all_metrics = []

        for c in concurrency_levels:
            # Repeat prompts to ensure enough requests per concurrency tier
            prompts_to_run = (test_prompts * ((c * 4) // len(test_prompts) + 1))[: c * 4]
            total_reqs = len(prompts_to_run)
            latencies = []

            def _worker(prompt: str):
                t0 = time.perf_counter()
                res = self.client.generate(
                    prompt=prompt,
                    max_tokens=max_output_tokens,
                    temperature=temperature,
                )
                lat = time.perf_counter() - t0
                return lat, len(res.generated_text.split()) * 1.3  # rough token estimate

            t_start = time.perf_counter()
            total_tokens = 0

            with concurrent.futures.ThreadPoolExecutor(max_workers=c) as executor:
                futures = [executor.submit(_worker, p) for p in prompts_to_run]
                for f in concurrent.futures.as_completed(futures):
                    try:
                        lat, tokens = f.result()
                        latencies.append(lat)
                        total_tokens += tokens
                    except Exception as e:
                        print(f"Request failed: {e}")

            total_elapsed = time.perf_counter() - t_start
            success = len(latencies)

            if not latencies:
                continue

            latencies.sort()
            p50 = statistics.median(latencies)
            p90 = latencies[int(len(latencies) * 0.90)]
            p95 = latencies[int(len(latencies) * 0.95)]
            p99 = latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)]
            mean = statistics.mean(latencies)

            metrics = BenchmarkMetrics(
                concurrency=c,
                total_requests=total_reqs,
                successful_requests=success,
                total_time_seconds=round(total_elapsed, 3),
                requests_per_second=round(success / total_elapsed, 2),
                estimated_tokens_per_second=round(total_tokens / total_elapsed, 1),
                latency_mean_sec=round(mean, 3),
                latency_p50_sec=round(p50, 3),
                latency_p90_sec=round(p90, 3),
                latency_p95_sec=round(p95, 3),
                latency_p99_sec=round(p99, 3),
            )
            all_metrics.append(metrics)

        return all_metrics

    @staticmethod
    def format_table(results: List[BenchmarkMetrics]) -> str:
        """Render markdown/ASCII table from benchmark results."""
        rows = [
            [
                m.concurrency,
                m.successful_requests,
                f"{m.requests_per_second} req/s",
                f"{m.estimated_tokens_per_second} tok/s",
                f"{m.latency_p50_sec * 1000:.0f} ms",
                f"{m.latency_p95_sec * 1000:.0f} ms",
                f"{m.latency_p99_sec * 1000:.0f} ms",
            ]
            for m in results
        ]
        headers = ["Concurrency", "Requests", "Req/sec", "Est. Throughput", "p50 Latency", "p95 Latency", "p99 Latency"]
        return tabulate(rows, headers=headers, tablefmt="github")
