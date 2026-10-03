"""
Client for querying models hosted on Vertex AI TPU endpoints.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from google.cloud import aiplatform


@dataclass
class GenerationResult:
    """Standardized output from TPU inference."""
    prompt: str
    generated_text: str
    latency_seconds: float
    raw_prediction: Any


class TPUInferenceClient:
    """Synchronous and batch inference client for Vertex AI TPU vLLM endpoints."""

    def __init__(
        self,
        endpoint_resource_name: str,
        use_dedicated_endpoint: bool = True,
    ):
        """
        Initialize inference client.
        
        Args:
            endpoint_resource_name: Full resource path or ID of the Vertex AI endpoint.
            use_dedicated_endpoint: Whether the endpoint is provisioned as dedicated.
        """
        self.endpoint = aiplatform.Endpoint(endpoint_resource_name)
        self.use_dedicated_endpoint = use_dedicated_endpoint

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        raw_response: bool = True,
    ) -> GenerationResult:
        """Execute text generation on the TPU endpoint."""
        import time

        instances = [
            {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "raw_response": raw_response,
            }
        ]

        t0 = time.perf_counter()
        response = self.endpoint.predict(
            instances=instances,
            use_dedicated_endpoint=self.use_dedicated_endpoint,
        )
        latency = time.perf_counter() - t0

        pred = response.predictions[0] if response.predictions else ""
        generated = pred if isinstance(pred, str) else str(pred)

        return GenerationResult(
            prompt=prompt,
            generated_text=generated,
            latency_seconds=latency,
            raw_prediction=pred,
        )

    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        raw_response: bool = True,
    ) -> List[GenerationResult]:
        """Execute high-concurrency batch inference on the TPU endpoint."""
        import time

        instances = [
            {
                "prompt": p,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "raw_response": raw_response,
            }
            for p in prompts
        ]

        t0 = time.perf_counter()
        response = self.endpoint.predict(
            instances=instances,
            use_dedicated_endpoint=self.use_dedicated_endpoint,
        )
        total_latency = time.perf_counter() - t0
        per_item_latency = total_latency / max(1, len(prompts))

        results = []
        for i, pred in enumerate(response.predictions):
            generated = pred if isinstance(pred, str) else str(pred)
            results.append(
                GenerationResult(
                    prompt=prompts[i],
                    generated_text=generated,
                    latency_seconds=per_item_latency,
                    raw_prediction=pred,
                )
            )
        return results
