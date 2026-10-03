#!/usr/bin/env python3
"""
CLI Script: Deploy an LLM onto Google Cloud TPU with vLLM on Vertex AI.
"""

import os
import argparse
import sys

# Ensure local package is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from tpu_serving.config import TPUConfig, ModelServingConfig, VertexDeploymentConfig, TPUGeneration
from tpu_serving.deployer import VertexTPUDeployer


def main():
    parser = argparse.ArgumentParser(description="Deploy open-weights LLMs on Google Cloud TPU v6e / v4.")
    parser.add_argument("--project-id", default=os.environ.get("GOOGLE_CLOUD_PROJECT"), required=not os.environ.get("GOOGLE_CLOUD_PROJECT"), help="GCP Project ID")
    parser.add_argument("--region", default=os.environ.get("TPU_DEPLOYMENT_REGION", "europe-west4"), help="GCP TPU Region")
    parser.add_argument("--bucket", default=os.environ.get("BUCKET_URI"), required=not os.environ.get("BUCKET_URI"), help="GCS Staging Bucket (gs://...)")
    parser.add_argument("--model", default="google/gemma-3-27b-it", help="Model Garden or HuggingFace ID")
    parser.add_argument("--machine-type", default="ct6e-standard-4t", help="TPU machine type (e.g. ct6e-standard-1t, ct6e-standard-4t)")
    parser.add_argument("--topology", default="2x2", help="Physical TPU topology (1x1, 2x2)")
    parser.add_argument("--tpu-count", default=4, type=int, help="TPU chip count (1, 2, 4)")
    parser.add_argument("--max-len", default=8192, type=int, help="Maximum context length")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"), help="Hugging Face read token")
    parser.add_argument("--name", default=None, help="Custom deployment display name")

    args = parser.parse_args()

    print("================================================================")
    print(" Google Cloud Vertex AI — TPU v6e / v4 LLM Serving Deployment   ")
    print("================================================================")
    print(f"Model:        {args.model}")
    print(f"Machine Type: {args.machine_type} ({args.tpu_count} chips, topology {args.topology})")
    print(f"Region:       {args.region}")
    print(f"Project ID:   {args.project_id}")
    print(f"GCS Bucket:   {args.bucket}")
    print("----------------------------------------------------------------")

    tpu_cfg = TPUConfig(
        machine_type=args.machine_type,
        tpu_count=args.tpu_count,
        tpu_topology=args.topology,
    )
    model_cfg = ModelServingConfig(
        model_id=args.model,
        max_model_len=args.max_len,
    )
    deploy_cfg = VertexDeploymentConfig(
        project_id=args.project_id,
        region=args.region,
        staging_bucket=args.bucket,
    )

    deployer = VertexTPUDeployer(tpu_cfg, model_cfg, deploy_cfg, hf_token=args.hf_token)
    print("[*] Initiating model upload & TPU endpoint provisioning...")
    model, endpoint = deployer.deploy(display_name=args.name)

    print("\n[+] Deployment Succeeded!")
    print(f"    Model Resource:    {model.resource_name}")
    print(f"    Endpoint Resource: {endpoint.resource_name}")
    print(f"    Endpoint ID:       {endpoint.name}")
    print("\nNext step: Test your endpoint with:")
    print(f"    python scripts/predict.py --endpoint {endpoint.name} --prompt 'What is Cloud TPU?'")


if __name__ == "__main__":
    main()
