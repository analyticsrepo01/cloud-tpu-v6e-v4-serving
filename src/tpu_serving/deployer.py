"""
Vertex AI Cloud TPU Model Deployment Orchestrator.
"""

import os
import time
from typing import Tuple, Optional, Dict, Any
from google.cloud import aiplatform

from .config import TPUConfig, ModelServingConfig, VertexDeploymentConfig


class VertexTPUDeployer:
    """Production deployer for hosting LLMs on Cloud TPUs with vLLM via Vertex AI."""

    def __init__(
        self,
        tpu_config: TPUConfig,
        model_config: ModelServingConfig,
        deployment_config: VertexDeploymentConfig,
        hf_token: Optional[str] = None,
    ):
        self.tpu_config = tpu_config
        self.model_config = model_config
        self.deployment_config = deployment_config
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

        # Initialize Vertex AI SDK
        aiplatform.init(
            project=self.deployment_config.project_id,
            location=self.deployment_config.region,
            staging_bucket=self.deployment_config.staging_bucket,
        )

    def build_container_args(self) -> list[str]:
        """Construct the optimized command-line flags for vLLM on TPU."""
        tensor_parallel = self.model_config.get_effective_tensor_parallel(self.tpu_config.tpu_count)

        args = [
            "python",
            "-m",
            "vllm.entrypoints.api_server",
            "--host=0.0.0.0",
            "--port=7080",
            f"--model={self.model_config.model_id}",
            f"--tensor_parallel_size={tensor_parallel}",
            f"--max_model_len={self.model_config.max_model_len}",
            "--limit_mm_per_prompt.image=0",
        ]

        if self.model_config.enable_chunked_prefill:
            args.append("--enable-chunked-prefill")

        if self.model_config.enable_prefix_caching:
            args.append("--enable-prefix-caching")

        return args

    def build_environment_variables(self) -> Dict[str, str]:
        """Construct serving container environment variables."""
        base_id = self.model_config.base_model_id or self.model_config.model_id
        env_vars = {
            "MODEL_ID": base_id,
            "DEPLOY_SOURCE": "tpu_serving_framework",
            "VLLM_USE_V1": "1" if self.model_config.use_vllm_v1 else "0",
        }
        if self.hf_token:
            env_vars["HF_TOKEN"] = self.hf_token
        return env_vars

    def deploy(
        self,
        display_name: Optional[str] = None,
        existing_endpoint_id: Optional[str] = None,
    ) -> Tuple[aiplatform.Model, aiplatform.Endpoint]:
        """Upload model and deploy to Vertex AI TPU Endpoint."""
        run_timestamp = int(time.time())
        name = display_name or f"tpu-{self.model_config.publisher_model_id}-{run_timestamp}"

        # 1. Resolve or Create Endpoint
        if existing_endpoint_id:
            endpoint_name = (
                f"projects/{self.deployment_config.project_id}/"
                f"locations/{self.deployment_config.region}/"
                f"endpoints/{existing_endpoint_id}"
            )
            endpoint = aiplatform.Endpoint(endpoint_name)
        else:
            endpoint = aiplatform.Endpoint.create(
                display_name=f"{name}-endpoint",
                location=self.deployment_config.region,
                dedicated_endpoint_enabled=self.deployment_config.use_dedicated_endpoint,
            )

        # 2. Upload Model with vLLM TPU Container
        num_hosts = int(self.tpu_config.tpu_topology.split("x")[0])
        container_args = self.build_container_args()
        env_vars = self.build_environment_variables()

        model = aiplatform.Model.upload(
            display_name=name,
            serving_container_image_uri=self.model_config.vllm_container_uri,
            serving_container_args=container_args,
            serving_container_ports=[7080],
            serving_container_predict_route="/generate",
            serving_container_health_route="/ping",
            serving_container_environment_variables=env_vars,
            serving_container_shared_memory_size_mb=self.tpu_config.shared_memory_mb,
            serving_container_deployment_timeout=self.deployment_config.container_timeout_seconds,
            model_garden_source_model_name=(
                f"publishers/{self.model_config.publisher}/models/{self.model_config.publisher_model_id}"
            ),
            location=self.deployment_config.region,
        )

        # 3. Deploy Model to Hardware
        model.deploy(
            endpoint=endpoint,
            machine_type=self.tpu_config.machine_type,
            tpu_topology=self.tpu_config.tpu_topology if num_hosts > 1 else None,
            deploy_request_timeout=self.deployment_config.deploy_timeout_seconds,
            service_account=self.deployment_config.service_account,
            min_replica_count=self.deployment_config.min_replica_count,
            max_replica_count=self.deployment_config.max_replica_count,
            system_labels={
                "workload": "llm-serving",
                "accelerator": self.tpu_config.generation.value,
                "framework": "vllm",
            },
        )

        return model, endpoint
