"""
Unit tests for TPU configuration validation and container argument generation.
"""

import pytest
from tpu_serving.config import TPUConfig, ModelServingConfig, VertexDeploymentConfig, TPUGeneration
from tpu_serving.deployer import VertexTPUDeployer


def test_tpu_config_defaults():
    cfg = TPUConfig()
    assert cfg.generation == TPUGeneration.V6E_TRILLIUM
    assert cfg.machine_type == "ct6e-standard-4t"
    assert cfg.tpu_count == 4
    assert cfg.tpu_topology == "2x2"
    assert cfg.shared_memory_mb == 16384


def test_tpu_config_invalid_count():
    with pytest.raises(ValueError):
        TPUConfig(tpu_count=3)


def test_model_serving_config_tensor_parallel():
    cfg = ModelServingConfig(model_id="google/gemma-3-27b-it")
    # By default, uses TPU count
    assert cfg.get_effective_tensor_parallel(4) == 4

    # Overridden tensor parallel size
    cfg_override = ModelServingConfig(model_id="google/gemma-3-27b-it", tensor_parallel_size=2)
    assert cfg_override.get_effective_tensor_parallel(4) == 2


def test_deployer_container_args():
    tpu_cfg = TPUConfig(machine_type="ct6e-standard-4t", tpu_count=4, tpu_topology="2x2")
    model_cfg = ModelServingConfig(
        model_id="google/gemma-3-27b-it",
        max_model_len=8192,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )
    deploy_cfg = VertexDeploymentConfig(
        project_id="test-project",
        region="europe-west4",
        staging_bucket="gs://test-bucket",
    )

    # Instantiate deployer (aiplatform init will execute with test params)
    deployer = VertexTPUDeployer(tpu_cfg, model_cfg, deploy_cfg, hf_token="test_hf_token")

    args = deployer.build_container_args()
    assert "--model=google/gemma-3-27b-it" in args
    assert "--tensor_parallel_size=4" in args
    assert "--max_model_len=8192" in args
    assert "--enable-prefix-caching" in args
    assert "--enable-chunked-prefill" in args

    env = deployer.build_environment_variables()
    assert env["MODEL_ID"] == "google/gemma-3-27b-it"
    assert env["VLLM_USE_V1"] == "1"
    assert env["HF_TOKEN"] == "test_hf_token"
