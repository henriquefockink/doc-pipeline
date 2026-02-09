# Benchmarks — Doc Pipeline

## Environment

- **GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM)
- **Platform**: WSL2 on Windows, Linux 6.6.87.2
- **Docker**: 29.2.0, Compose v5.0.2
- **Inference Server**: PyTorch 2.7.0 + CUDA 12.8
- **VLM**: Qwen2.5-VL-3B-Instruct (HuggingFace transformers, static batching)
- **Batch Size**: 4 (default)
- **Batch Timeout**: 100ms
- **Inference Timeout**: 120s
- **Workers**: 5x DocID (stateless, no GPU) + 1x OCR (stateless, no GPU)
- **VRAM Usage (idle)**: ~10.4 GB (classifier + EasyOCR + VLM)
- **Test File**: `cnh_noventa_graus.jpg` (CNH rotated 90 degrees)
- **Pipeline**: orientation correction (EasyOCR) + classification (EfficientNet) + extraction (VLM)
- **Date**: 2026-02-09

## Baseline — Custom Inference Server (transformers + static batching)

### Test 1: Single Request (baseline latency)

| Metric | Value |
|--------|-------|
| Requests | 10 |
| Concurrency | 1 |
| RPS | 1 |
| OK / Fail | 10 / 0 |
| Latency mean | 5,225 ms |
| P50 | 5,138 ms |
| P95 | 5,440 ms |
| P99 | 5,445 ms |
| Throughput | 0.19 rps |

### Test 2: Low Concurrency

| Metric | Value |
|--------|-------|
| Requests | 30 |
| Concurrency | 5 |
| RPS | 2 |
| OK / Fail | 30 / 0 |
| Latency mean | 13,503 ms |
| P50 | 13,878 ms |
| P95 | 14,213 ms |
| P99 | 14,221 ms |
| Throughput | 0.36 rps |

### Test 3: Medium Concurrency

| Metric | Value |
|--------|-------|
| Requests | 50 |
| Concurrency | 10 |
| RPS | 5 |
| OK / Fail | 50 / 0 |
| Latency mean | 21,738 ms |
| P50 | 20,076 ms |
| P95 | 27,491 ms |
| P99 | 29,174 ms |
| Throughput | 0.42 rps |

### Test 4: High Concurrency

| Metric | Value |
|--------|-------|
| Requests | 50 |
| Concurrency | 20 |
| RPS | 10 |
| OK / Fail | 50 / 0 |
| Latency mean | 38,869 ms |
| P50 | 44,063 ms |
| P95 | 48,755 ms |
| P99 | 50,383 ms |
| Throughput | 0.42 rps |

### Test 5: Stress Test

| Metric | Value |
|--------|-------|
| Requests | 60 |
| Concurrency | 30 |
| RPS | 15 |
| OK / Fail | 60 / 0 |
| Latency mean | 51,035 ms |
| P50 | 59,108 ms |
| P95 | 67,834 ms |
| P99 | 69,776 ms |
| Throughput | 0.46 rps |

### Resource Usage During Tests

| Metric | Idle | Under Load (30 concurrent) |
|--------|------|---------------------------|
| VRAM | 10.4 GB | 10.8 GB (stable) |
| GPU Utilization | 1-3% | 28-89% (bursty) |
| Redis Memory | 1.5 MB | 2.0 MB |
| Redis Clients | 26 | 32 |
| Inference Queue Peak | 0 | 17 |

### Key Observations

- **Baseline latency**: ~5.1s per request (single, no contention)
- **Throughput ceiling**: ~0.42-0.46 rps regardless of concurrency — VLM with static batch=4 is the bottleneck
- **Latency scales linearly with queue depth**: each additional batch in queue adds ~8s
- **VRAM is stable**: no memory leak under sustained load, stays at ~10.4-10.8 GB
- **Zero errors** with 120s timeout (previously 72% failure rate at 30s timeout with 20 concurrent)
- **GPU utilization is bursty**: spikes to 89% during VLM forward pass, drops between batches

### Bottleneck Analysis

The VLM processes requests in static batches of 4. Each batch takes ~7-8s for the full forward pass.
With N concurrent requests, the queue depth grows and latency = `(queue_position / batch_size) * batch_time`.

Example with 20 concurrent:
- Requests 1-4: ~5s (first batch, immediate)
- Requests 5-8: ~14s (wait for batch 1 + batch 2)
- Requests 9-12: ~22s (wait for batches 1-3)
- Requests 13-16: ~32s (wait for batches 1-4)
- Requests 17-20: ~41s (wait for batches 1-5)

**Continuous batching (vLLM)** would reduce this by interleaving token generation across requests
instead of processing them sequentially in fixed groups.

## vLLM Comparison

_TBD — run same tests after vLLM migration_

## Summary Table

| Scenario | Concurrency | Latency P50 | Latency P95 | Throughput | Errors |
|----------|-------------|-------------|-------------|------------|--------|
| Single | 1 | 5.1s | 5.4s | 0.19 rps | 0% |
| Low | 5 | 13.9s | 14.2s | 0.36 rps | 0% |
| Medium | 10 | 20.1s | 27.5s | 0.42 rps | 0% |
| High | 20 | 44.1s | 48.8s | 0.42 rps | 0% |
| Stress | 30 | 59.1s | 67.8s | 0.46 rps | 0% |
