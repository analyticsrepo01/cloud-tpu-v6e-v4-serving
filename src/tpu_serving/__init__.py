"""
Vertex TPU vLLM Serving
=======================
Enterprise-grade LLM serving infrastructure on Google Cloud TPU v6e (Trillium) and TPU v4.
"""

from .config import TPUConfig, ModelServingConfig, TPUGeneration
from .deployer import VertexTPUDeployer
from .client import TPUInferenceClient
from .benchmark import TPUBenchmarkRunner

__version__ = "1.0.0"
__all__ = [
    "TPUConfig",
    "ModelServingConfig",
    "TPUGeneration",
    "VertexTPUDeployer",
    "TPUInferenceClient",
    "TPUBenchmarkRunner",
]
