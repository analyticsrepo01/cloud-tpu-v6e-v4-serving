"""
Unified Command Line Interface for Cloud TPU Serving.
"""

import sys
import click
from rich.console import Console
from rich.panel import Panel

from .config import TPUConfig, ModelServingConfig, VertexDeploymentConfig, TPUGeneration
from .deployer import VertexTPUDeployer
from .client import TPUInferenceClient
from .benchmark import TPUBenchmarkRunner

console = Console()


@click.group()
def main():
    """Vertex TPU vLLM Serving CLI — Production LLM deployment on Cloud TPUs."""
    pass


@main.command()
@click.option("--project-id", envvar="GOOGLE_CLOUD_PROJECT", required=True, help="Google Cloud Project ID.")
@click.option("--region", default="europe-west4", help="GCP Region where TPU quota exists.")
@click.option("--bucket", required=True, help="GCS staging bucket (e.g., gs://my-bucket-tpu).")
@click.option("--model", default="google/gemma-3-27b-it", help="Model ID (Hugging Face or Model Garden).")
@click.option("--machine-type", default="ct6e-standard-4t", help="TPU machine type (e.g. ct6e-standard-1t, ct6e-standard-4t).")
@click.option("--topology", default="2x2", help="TPU mesh topology (1x1, 2x2).")
@click.option("--tpu-count", default=4, type=int, help="TPU chip count.")
@click.option("--max-len", default=8192, type=int, help="Maximum model context length.")
@click.option("--hf-token", envvar="HF_TOKEN", default=None, help="Hugging Face read token for gated weights.")
def deploy(project_id, region, bucket, model, machine_type, topology, tpu_count, max_len, hf_token):
    """Deploy an LLM on Cloud TPU v6e / v4 with vLLM."""
    console.print(Panel(f"[bold cyan]Deploying {model} to Cloud TPU ({machine_type})[/bold cyan]\nRegion: {region} | Bucket: {bucket}"))

    tpu_cfg = TPUConfig(
        machine_type=machine_type,
        tpu_count=tpu_count,
        tpu_topology=topology,
    )
    model_cfg = ModelServingConfig(
        model_id=model,
        max_model_len=max_len,
    )
    deploy_cfg = VertexDeploymentConfig(
        project_id=project_id,
        region=region,
        staging_bucket=bucket,
    )

    deployer = VertexTPUDeployer(tpu_cfg, model_cfg, deploy_cfg, hf_token=hf_token)
    with console.status("[bold green]Uploading model & provisioning TPU hardware on Vertex AI (takes 15-30m)..."):
        model_res, endpoint_res = deployer.deploy()

    console.print(f"[bold green]✓ Successfully Deployed![/bold green]")
    console.print(f"Endpoint Resource: [yellow]{endpoint_res.resource_name}[/yellow]")
    console.print(f"Endpoint ID: [yellow]{endpoint_res.name}[/yellow]")


@main.command()
@click.option("--endpoint", required=True, help="Vertex AI Endpoint resource name or ID.")
@click.option("--prompt", required=True, help="Input prompt text.")
@click.option("--max-tokens", default=256, type=int, help="Maximum tokens to generate.")
@click.option("--temperature", default=0.7, type=float, help="Sampling temperature.")
def predict(endpoint, prompt, max_tokens, temperature):
    """Query a deployed TPU endpoint for inference."""
    client = TPUInferenceClient(endpoint_resource_name=endpoint)
    console.print(f"[bold]Prompt:[/bold] {prompt}\n")
    
    with console.status("[bold blue]Querying TPU Endpoint..."):
        res = client.generate(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        
    console.print(Panel(res.generated_text, title="[bold green]TPU Generated Response[/bold green]"))
    console.print(f"Latency: [cyan]{res.latency_seconds * 1000:.1f} ms[/cyan]")


@main.command()
@click.option("--endpoint", required=True, help="Vertex AI Endpoint resource name or ID.")
@click.option("--concurrency", default="1,2,4,8", help="Comma-separated list of concurrency levels.")
def benchmark(endpoint, concurrency):
    """Run automated throughput and latency benchmarks against the TPU endpoint."""
    client = TPUInferenceClient(endpoint_resource_name=endpoint)
    runner = TPUBenchmarkRunner(client)

    test_prompts = [
        "Explain quantum computing in simple terms.",
        "Write a Python function to compute Fibonacci numbers efficiently.",
        "Summarize the key architectural advantages of Google Cloud TPU v6e Trillium.",
        "Draft a professional email proposing a cloud infrastructure migration to TPUs.",
    ]
    concurrency_levels = [int(x.strip()) for x in concurrency.split(",")]

    console.print(Panel("[bold cyan]Starting Cloud TPU Benchmark[/bold cyan]"))
    results = runner.run_benchmark(test_prompts, concurrency_levels=concurrency_levels)
    table_md = runner.format_table(results)
    console.print(table_md)


if __name__ == "__main__":
    main()
