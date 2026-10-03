# Technical Architecture: Cloud TPU v6e (Trillium) & vLLM Serving

## 1. Cloud TPU v6e (Trillium) Hardware Overview

Google Cloud TPU v6e (Trillium) represents Google's 6th-generation custom Application-Specific Integrated Circuit (ASIC) engineered specifically for large-scale transformer inference and training.

```
+-------------------------------------------------------------------------+
|                  Google Cloud TPU v6e (Trillium) ASIC                    |
|                                                                         |
|  +--------------------+  +--------------------+  +-------------------+  |
|  | Matrix Multiply    |  | Vector Processing  |  | Scalar Processing |  |
|  | Unit (MXU)         |  | Unit (VPU)         |  | Unit (SPU)        |  |
|  | (BF16 / FP8)       |  | (Elementwise Ops)  |  | (Control Flow)    |  |
|  +--------------------+  +--------------------+  +-------------------+  |
|            |                        |                       |           |
|  +-------------------------------------------------------------------+  |
|  | High Bandwidth Memory (HBM) Subsystem - Ultra-Low Latency Bus     |  |
|  +-------------------------------------------------------------------+  |
|                                    |                                    |
|  +-------------------------------------------------------------------+  |
|  | Inter-Chip Interconnect (ICI) - Optical Circuit Switching (OCS)   |  |
|  +-------------------------------------------------------------------+  |
+-------------------------------------------------------------------------+
```

### Key Hardware Characteristics:
* **Compute Capabilities:** Features 3rd-generation Matrix Multiply Units (MXUs) natively supporting `BF16` and `FP8` precision math with up to **4.7x compute performance per dollar** over TPU v5e.
* **HBM Bandwidth:** High Bandwidth Memory allows serving models like Gemma 3 27B and Llama 3.1 70B with extreme memory bandwidth, preventing memory-bound decoding stalls.
* **Inter-Chip Interconnect (ICI):** High-speed interconnect enables transparent multi-chip tensor parallelism across `1x1`, `2x2`, and larger multi-dimensional topologies without PCIe or network latency bottlenecks.

---

## 2. vLLM Serving Engine on TPU XLA

vLLM utilizes **PagedAttention** to eliminate memory fragmentation in the Key-Value (KV) cache. On Cloud TPUs, vLLM compiles transformer attention heads and feed-forward layers directly into optimized XLA (Accelerated Linear Algebra) computation graphs.

```
Incoming Request
      │
      ▼
┌──────────────────────────────┐
│  vLLM API Server (Port 7080) │
│  - FastAPI / Uvicorn         │
│  - Dynamic Batching          │
└──────────────┬───────────────┘
               │
               ▼
┌────────────────────────────────────────────────────────┐
│  vLLM TPU Worker Runtime (V1 Engine)                   │
│                                                        │
│  ┌────────────────────────┐  ┌──────────────────────┐  │
│  | PagedAttention TPU XLA |  | Shared Memory (16GB) |  │
│  | KV-Cache Blocks        |  | Inter-Process IPC    |  │
│  └────────────────────────┘  └──────────────────────┘  │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  | Tensor Parallelism (TP=1, 2, 4 ranks)            |  │
│  | Synchronized via TPU ICI Mesh                    |  │
│  └──────────────────────────────────────────────────┘  │
└──────────────┬──────────────────────────────┬──────────┘
               │                              │
               ▼                              ▼
      ┌─────────────────┐            ┌─────────────────┐
      │ TPU Chip Rank 0 │  ◄──ICI──► │ TPU Chip Rank 1 │
      └─────────────────┘            └─────────────────┘
```

### Critical Serving Optimizations

1. **PagedAttention on TPU Memory:**
   - Standard attention allocates contiguous memory chunks, causing up to 60–80% internal fragmentation.
   - PagedAttention divides the KV cache into fixed-size virtual pages mapped dynamically onto TPU HBM, allowing higher batch sizes and concurrent requests.

2. **Prefix Caching (`--enable-prefix-caching`):**
   - Automatically reuses computed KV blocks across user queries that share common system prompts, multi-turn chat history, or document context in RAG workloads.
   - Drastically slashes Time-To-First-Token (TTFT).

3. **Chunked Prefill (`--enable-chunked-prefill`):**
   - Chunks massive input prompts and interleaves them with decode operations, preventing long prefill requests from starving active generation streams.

4. **16GB Shared Memory (`/dev/shm`):**
   - Configured via `serving_container_shared_memory_size_mb = 16384` to prevent Linux out-of-memory (OOM) bus errors during inter-process weight exchange and buffer synchronization.

---

## 3. Vertex AI Cloud Infrastructure Architecture

```
                                  Client Request
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │ Google Cloud Load Balancer (Vertex AI)    │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │ Dedicated Vertex AI Endpoint              │
                  │ - Predict Route: /generate                │
                  │ - Health Route:  /ping                    │
                  │ - SLA: 99.9% uptime                       │
                  └─────────────────────┬─────────────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
                         ▼                             ▼
              ┌─────────────────────┐       ┌─────────────────────┐
              │ Replica 1 (Active)  │       │ Replica 2 (Scale)   │
              │ Machine:            │       │ Machine:            │
              │ ct6e-standard-4t    │       │ ct6e-standard-4t    │
              │ 4x TPU v6e Trillium │       │ 4x TPU v6e Trillium │
              │ Topology: 2x2       │       │ Topology: 2x2       │
              └─────────────────────┘       └─────────────────────┘
```

### Deployment Topology Matrix

| Machine Type | TPU Generation | Chip Count | Mesh Topology | Tensor Parallelism | Target Models |
|---|---|---|---|---|---|
| `ct6e-standard-1t` | TPU v6e (Trillium) | 1 | `1x1` | 1 | Gemma 2 9B, Llama 3.1 8B, Qwen 2.5 7B |
| `ct6e-standard-4t` | TPU v6e (Trillium) | 4 | `2x2` | 4 | Gemma 3 27B, SEA-LION 27B, Qwen 3 32B |
| `ct6e-standard-8t` | TPU v6e (Trillium) | 8 | `2x4` | 8 | Llama 3.1 70B, Qwen 2.5 72B |
| `v4-8` | TPU v4 | 4 | `2x2x1` | 4 | Gemma 2 27B, Mixtral 8x7B |
