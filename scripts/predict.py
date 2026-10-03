#!/usr/bin/env python3
"""
CLI Script: Test inference against a deployed Cloud TPU Vertex AI Endpoint.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from tpu_serving.client import TPUInferenceClient


def main():
    parser = argparse.ArgumentParser(description="Query Vertex AI TPU Endpoint.")
    parser.add_argument("--endpoint", required=True, help="Endpoint ID or full resource name")
    parser.add_argument("--prompt", default="Explain the benefits of Google Cloud TPU v6e Trillium.", help="Prompt text")
    parser.add_argument("--max-tokens", default=256, type=int, help="Maximum generated tokens")
    parser.add_argument("--temperature", default=0.7, type=float, help="Sampling temperature")
    parser.add_argument("--top-p", default=0.95, type=float, help="Top-p sampling")

    args = parser.parse_args()

    client = TPUInferenceClient(endpoint_resource_name=args.endpoint)
    print(f"[*] Sending prompt to TPU Endpoint: {args.endpoint}...")
    print(f"    Prompt: {args.prompt}\n")

    result = client.generate(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print("-------------------- Response --------------------")
    print(result.generated_text)
    print("--------------------------------------------------")
    print(f"Generation Latency: {result.latency_seconds * 1000:.1f} ms")


if __name__ == "__main__":
    main()
