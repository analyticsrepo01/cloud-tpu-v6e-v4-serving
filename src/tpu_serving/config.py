"""
Configuration models for Google Cloud TPU LLM Serving with vLLM.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator


class TPUGeneration(str, Enum):
    V6E_TRILLIUM = "tpu_v6e"
    V4 = "tpu_v4"
    V5E = "tpu_v5e"
    V5P = "tpu_v5p"


class TPUConfig(BaseModel):
    """Hardware and topology configuration for Cloud TPU."""

    generation: TPUGeneration = Field(
        default=TPUGeneration.V6E_TRILLIUM,
        description="Google Cloud TPU hardware generation."
    )
    machine_type: str = Field(
        default="ct6e-standard-4t",
        description="GCP machine type (e.g., ct6e-standard-1t, ct6e-standard-4t, v4-8)."
    )
    tpu_count: int = Field(
        default=4,
        description="Number of TPU accelerator cores/chips attached to the instance."
    )
    tpu_topology: str = Field(
        default="2x2",
        description="Physical mesh interconnect topology (e.g., '1x1' for 1-chip, '2x2' for 4-chip)."
    )
    shared_memory_mb: int = Field(
        default=16384,
        description="Shared memory allocation (/dev/shm) in megabytes for IPC and vLLM KV-cache management."
    )

    @field_validator("tpu_count", mode="before")
    @classmethod
    def validate_tpu_count(cls, v: int) -> int:
        if v not in [1, 2, 4, 8, 16, 32, 64]:
            raise ValueError(f"Unsupported TPU count: {v}. Must be a power of 2.")
        return v


class ModelServingConfig(BaseModel):
    """LLM serving and vLLM runtime configuration."""

    model_id: str = Field(
        default="google/gemma-3-27b-it",
        description="Hugging Face or Model Garden model identifier."
    )
    base_model_id: Optional[str] = Field(
        default=None,
        description="Underlying base model ID if deploying a fine-tuned adapter or alias."
    )
    publisher: str = Field(
        default="google",
        description="Model publisher name."
    )
    publisher_model_id: str = Field(
        default="gemma-3-27b-it",
        description="Publisher catalog identifier."
    )
    max_model_len: int = Field(
        default=8192,
        description="Maximum sequence length (prompt + output tokens) supported by the vLLM engine."
    )
    tensor_parallel_size: Optional[int] = Field(
        default=None,
        description="Number of TPU tensor parallel ranks. Defaults to TPU count."
    )
    enable_chunked_prefill: bool = Field(
        default=False,
        description="Enable vLLM chunked prefill to interleave prefill and decode phases."
    )
    enable_prefix_caching: bool = Field(
        default=True,
        description="Enable automatic KV-cache prefix caching for multi-turn chats and shared system prompts."
    )
    vllm_container_uri: str = Field(
        default="us-docker.pkg.dev/vertex-ai/vertex-vision-model-garden-dockers/pytorch-vllm-serve:20250819_0917_tpu_experimental_RC01",
        description="Artifact Registry URI for the vLLM TPU serving container."
    )
    use_vllm_v1: bool = Field(
        default=True,
        description="Enable vLLM V1 architecture engine for optimized TPU XLA kernel dispatch."
    )

    def get_effective_tensor_parallel(self, tpu_count: int) -> int:
        return self.tensor_parallel_size if self.tensor_parallel_size is not None else tpu_count


class VertexDeploymentConfig(BaseModel):
    """Google Cloud Vertex AI infrastructure deployment settings."""

    project_id: str = Field(..., description="Google Cloud Project ID.")
    region: str = Field(default="europe-west4", description="GCP Region where TPU quota is allocated.")
    staging_bucket: str = Field(..., description="Google Cloud Storage bucket for temporary deployment artifacts.")
    service_account: Optional[str] = Field(
        default=None,
        description="IAM service account for running Vertex AI endpoint tasks."
    )
    min_replica_count: int = Field(default=1, ge=1)
    max_replica_count: int = Field(default=1, ge=1)
    use_dedicated_endpoint: bool = Field(
        default=True,
        description="Deploy to a dedicated endpoint for predictable latency and enterprise SLAs."
    )
    deploy_timeout_seconds: int = Field(default=1800)
    container_timeout_seconds: int = Field(default=4500)
