#!/usr/bin/env python3
"""
CLI Script: Benchmark Cloud TPU Endpoint throughput, latency percentiles, and concurrency.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from tpu_serving.client import TPUInferenceClient
from tpu_serving.benchmark import TPUBenchmarkRunner


def main():
    parser = argparse.ArgumentParser(description="Benchmark Vertex AI TPU Endpoint.")
    parser.add_argument("--endpoint", required=True, help="Endpoint ID or full resource name")
    parser.add_argument("--concurrency", default="1,2,4,8", help="Comma-separated concurrency levels")
    parser.add_argument("--max-tokens", default=128, type=int, help="Output tokens per request")
    parser.add_argument("--temperature", default=0.0, type=float, help="Sampling temperature")

    args = parser.parse_args()

    client = TPUInferenceClient(endpoint_resource_name=args.endpoint)
    runner = TPUBenchmarkRunner(client)

    test_prompts = [
        "What are the main differences between TPUs and GPUs?",
        "Explain how PagedAttention improves memory utilization in vLLM.",
        "Provide an architectural overview of Google's Optical Circuit Switches (OCS) in TPU supercomputers.",
        "Write a Python script demonstrating asynchronous request dispatch.",
    ]
    concurrency_levels = [int(c.strip()) for c in args.concurrency.split(",")]

    print(f"[*] Running performance benchmarks on endpoint {args.endpoint}...")
    print(f"    Concurrency levels: {concurrency_levels}")
    print(f"    Max tokens per request: {args.max_tokens}")
    print("---------------------------------------------------------------")

    results = runner.run_benchmark(
        test_prompts=test_prompts,
        concurrency_levels=concurrency_levels,
        max_output_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    print("\nBenchmark Results:")
    print(runner.format_table(results))


if __name__ == "__main__":
    main()
