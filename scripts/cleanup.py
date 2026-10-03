#!/usr/bin/env python3
"""
CLI Script: Safe teardown and cleanup of Vertex AI TPU models and endpoints.
"""

import os
import argparse
from google.cloud import aiplatform


def main():
    parser = argparse.ArgumentParser(description="Clean up Vertex AI TPU models and endpoints.")
    parser.add_argument("--project-id", default=os.environ.get("GOOGLE_CLOUD_PROJECT"), required=not os.environ.get("GOOGLE_CLOUD_PROJECT"))
    parser.add_argument("--region", default=os.environ.get("TPU_DEPLOYMENT_REGION", "europe-west4"))
    parser.add_argument("--endpoint", required=True, help="Endpoint ID or full resource name to undeploy and delete")
    parser.add_argument("--delete-model", action="store_true", help="Also delete the uploaded model registry artifact")

    args = parser.parse_args()

    aiplatform.init(project=args.project_id, location=args.region)

    endpoint_name = (
        f"projects/{args.project_id}/locations/{args.region}/endpoints/{args.endpoint}"
        if not args.endpoint.startswith("projects/")
        else args.endpoint
    )

    print(f"[*] Fetching endpoint {endpoint_name}...")
    endpoint = aiplatform.Endpoint(endpoint_name)

    deployed_models = endpoint.list_models()
    print(f"    Found {len(deployed_models)} deployed model(s) on endpoint.")

    print(f"[*] Undeploying models and deleting endpoint...")
    endpoint.delete(force=True)
    print("    Endpoint successfully deleted.")

    if args.delete_model:
        for dm in deployed_models:
            model_id = dm.model
            print(f"[*] Deleting model resource: {model_id}...")
            try:
                m = aiplatform.Model(model_id)
                m.delete()
                print(f"    Model {model_id} deleted.")
            except Exception as e:
                print(f"    Could not delete model {model_id}: {e}")

    print("\n[+] Resource teardown complete. No continuous TPU charges will incur.")


if __name__ == "__main__":
    main()
