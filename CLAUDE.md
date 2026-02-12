# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

doc-pipeline is a document processing pipeline for Brazilian identity documents (RG, CNH, CIN) with two main functions:

1. **Classification + Extraction** (Worker DocID) — Classifies document type via EfficientNet (12 classes), extracts structured data via VLM (Qwen2.5-VL) or OCR+regex (EasyOCR)
2. **Generic OCR** (Worker OCR) — Extracts text from any PDF or image using EasyOCR with Portuguese support

## Commands

```bash
# Install dependencies (requires editable doc-classifier from sibling dir)
pip install -r requirements.txt

# Install dev dependencies (pre-commit, ruff, pytest)
pip install -e ".[dev]"
pre-commit install

# Run CLI
python cli.py documento.jpg -m models/classifier.pth                    # full pipeline
python cli.py documento.jpg -m models/classifier.pth --no-extraction    # classify only
python cli.py documento.jpg -m models/classifier.pth --backend easy-ocr # low VRAM

# Run API server locally
python api.py

# Run with ngrok (persists after SSH disconnect)
./start-server.sh                              # random ngrok URL
./start-server.sh start meu-dominio.ngrok.io   # custom domain
./start-server.sh status
./stop-server.sh

# Linting (ruff not installed system-wide; pip install ruff first)
ruff check . && ruff format --check .
pre-commit run --all-files

# Tests
pytest                      # run all tests
pytest tests/test_foo.py -k "test_name"  # single test

# Docker
docker compose up -d                           # start all services
docker compose build && docker compose up -d   # rebuild after code changes
```

## Architecture

```
                                    ┌─────────────────────────────────────────────────┐
                                    │                 GPU Node                        │
                                    │                                                 │
Client ─► api.py ─► Redis queue ──► │  inference_server.py (port 9020, GPU)           │
           (9000)    (6379)         │    ├── EfficientNet classifier (~200MB)          │
           no GPU                   │    ├── EasyOCR engine (~2-4GB)                   │
              │                     │    ├── OrientationCorrector (shares EasyOCR)     │
              │                     │    ├── PDFConverter (CPU)                         │
              │                     │    └── VLLMEmbeddedClient (in-process vLLM)      │
              │                     │         Qwen2.5-VL — GPU ~10-16GB                │
              │                     │         PIL images passed directly (zero-copy)    │
              │                     │         continuous batching + PagedAttention      │
              │                     └─────────────────────────────────────────────────┘
              │
              ├──► worker_docid.py × N (stateless, no GPU)
              │      └── Delegates all inference to inference_server via Redis
              │
              └──► worker_ocr.py (stateless, no GPU)
                     └── Delegates all inference to inference_server via Redis
```

**Entry points**: `cli.py`, `api.py`, `worker_docid.py`, `worker_ocr.py`, `inference_server.py`

### VLM Backend: vLLM Embedded

The key architectural decision is running vLLM **in-process** inside the inference server (not as a separate HTTP container). The `VLLMEmbeddedClient` (`doc_pipeline/extractors/vllm_embedded.py`) uses vLLM's offline `LLM` class to pass PIL images directly via `multi_modal_data`, bypassing all base64 encoding and HTTP serialization.

### Request Flow

1. **Client** sends image/PDF to `api.py` via REST
2. **API** saves file to shared volume (`/tmp/doc-pipeline/`), enqueues job to `queue:doc:documents`
3. **Worker DocID** picks up the job (BRPOP), sends inference request to `queue:doc:inference`
4. **Inference Server** collects up to `INFERENCE_BATCH_SIZE` requests (default 16):
   - **Parallel preprocessing**: concurrent image load + orientation + classification via `asyncio.gather`
   - Passes PIL images directly to in-process vLLM (no serialization)
   - **Pipeline overlap**: collects batch N+1 while VLM processes batch N
   - Publishes result to `inference:result:{id}`
5. **Worker** polls the reply key, builds final response
6. **API** returns result to client (sync) or POSTs to webhook (async)

### Key Design Decisions

- **vLLM in-process**: eliminates HTTP+base64 overhead. PIL images go directly to GPU.
- **Pipeline overlap**: event loop collects next batch while VLM generates for current batch.
- **Workers are stateless and GPU-free**: all GPU work centralized in inference server.
- **File passing via shared volume**: images referenced by path through Redis, not serialized.
- **Redis as message bus**: all inter-service communication via queues + reply keys. No direct HTTP between workers and inference server.

## API Endpoints

| Endpoint | Method | Description | Worker |
|----------|--------|-------------|--------|
| `/classify` | POST | Classify document type (image or PDF) | docid |
| `/extract` | POST | Extract structured data (image or PDF) | docid |
| `/process` | POST | Classify + extract full pipeline (image or PDF) | docid |
| `/ocr` | POST | Generic OCR for PDF/image | ocr |
| `/warmup` | POST | Pre-scale workers (requires `WARMUP_API_KEY`) | - |
| `/jobs/{id}/status` | GET | Poll job status (sync mode) | - |
| `/health` | GET | Health check | - |
| `/metrics` | GET | Aggregated Prometheus metrics | - |

**Delivery modes**: `sync` (default, polls Redis up to 5min), `webhook` (returns 202, POSTs result)

## Port Allocation

**IMPORTANT**: doc-pipeline uses the **9000-9099** port range. The **8000-8099** range belongs to ASR platform. Never use ports in the 8xxx range to avoid conflicts with ASR services.

| Service | Port | Description |
|---------|------|-------------|
| API | 9000 | REST API (FastAPI) |
| DocID workers 1-5 | 9010, 9012, 9014, 9016, 9018 | Worker health/metrics |
| OCR worker | 9011 | OCR worker health/metrics |
| Inference server | 9020 | Centralized GPU inference |

### Port ranges on this server (H200)

| Range | Service | Owner |
|-------|---------|-------|
| 6379 | Redis | shared |
| 8000-8099 | ASR platform (API, workers, TTS) | asr-platform |
| 8080 | cAdvisor | monitoring |
| 9000-9099 | Doc pipeline (API, workers, inference) | doc-pipeline |
| 9100 | Node exporter | monitoring |
| 9400 | DCGM exporter | monitoring |

## Configuration

All settings prefixed with `DOC_PIPELINE_` (see `doc_pipeline/config.py`). Defaults shown are from config.py; docker-compose.yml may override them.

| Setting | Default | Description |
|---------|---------|-------------|
| `CLASSIFIER_MODEL_TYPE` | efficientnet_b0 | b0, b2, or b4 |
| `EXTRACTOR_BACKEND` | hybrid | hybrid, qwen-vl, or easy-ocr |
| `EXTRACTOR_MODEL_QWEN` | Qwen/Qwen2.5-VL-7B-Instruct | VLM model (docker-compose overrides to 3B) |
| `CLASSIFIER_DEVICE` / `EXTRACTOR_DEVICE` | cuda:0 | cuda:N or cpu |
| `API_KEY` | - | Optional auth (requires `X-API-Key` header) |
| `INFERENCE_BATCH_SIZE` | 16 | Max requests per VLM batch |
| `INFERENCE_BATCH_TIMEOUT_MS` | 100 | Max wait to fill a batch (ms) |
| `WORKER_CONCURRENT_JOBS` | 1 | Parallel jobs per worker (docker-compose overrides to 4) |
| `INFERENCE_TIMEOUT_SECONDS` | 120 | Timeout for inference requests (seconds) |
| `VLLM_EMBEDDED` | true | Run vLLM in-process (production mode) |
| `VLLM_MODEL` | Qwen/Qwen2.5-VL-7B-Instruct | Model name for vLLM (docker-compose overrides to 3B) |
| `VLLM_MAX_TOKENS` | 1024 | Max tokens for VLM generation |
| `VLLM_GPU_MEMORY_UTILIZATION` | 0.40 | Fraction of GPU memory for vLLM KV cache (0.1-0.95) |
| `VLLM_MAX_MODEL_LEN` | 4096 | Maximum context length for vLLM |

## Docker Deployment

### Containers

| Container | Dockerfile | GPU | Description |
|-----------|-----------|-----|-------------|
| **redis** | redis:7-alpine | No | Message broker + result store (AOF persistence) |
| **api** | Dockerfile.api | No | FastAPI REST API, rate limiting, job enqueuing |
| **inference-server** | Dockerfile.inference-server | Yes (~14-20GB) | EfficientNet + EasyOCR + vLLM in-process (Qwen2.5-VL) |
| **worker-docid-1..5** | Dockerfile.worker | No | Stateless job consumers, no ML models |
| **worker-ocr** | Dockerfile.worker | No | Stateless OCR job consumer |
| **promtail** | grafana/promtail:latest | No | Log collector for centralized logging |

### Volumes

| Volume | Mounted at | Purpose |
|--------|-----------|---------|
| `temp-images` | `/tmp/doc-pipeline` | Shared file passing between API, workers, inference server |
| `model-cache` | `/root/.cache/huggingface` | HuggingFace model cache |
| `easyocr-cache` | `/root/.EasyOCR` | EasyOCR model cache |
| `redis-data` | `/data` | Redis AOF persistence |

## Document Types

12 classes across 3 document types: `rg_frente`, `rg_verso`, `rg_aberto`, `rg_digital`, `cnh_frente`, `cnh_verso`, `cnh_aberta`, `cnh_digital`, `cin_frente`, `cin_verso`, `cin_aberta`, `cin_digital`

Extraction yields `RGData`, `CNHData`, or `CINData` based on document type (defined in `schemas.py`).

### Extraction Backends

| Backend | How it works | Speed | Accuracy |
|---------|-------------|-------|----------|
| `hybrid` (default) | EasyOCR for digits + VLM for structure, CPF cross-validation | ~15s | Best (CPF) |
| `vlm` | Qwen2.5-VL only | ~5s | Good |
| `ocr` | EasyOCR + regex only | ~2s | Lower |

## Key Patterns

- **Lazy loading**: Models loaded on first access via `@property` (DocumentPipeline, workers)
- **Abstract base + factory**: `BaseExtractor` ABC; pipeline creates correct extractor from backend enum
- **Singletons**: `get_settings()`, `get_metrics()`, `get_queue_service()` return cached instances
- **JobContext dataclass**: Lightweight `@dataclass` (not Pydantic) for serializable job state through Redis
- **Shared OCR engine**: Single EasyOCR instance shared across extractors to avoid duplicate model loads
- **External dependency**: Requires `doc-classifier` package installed as editable (`pip install -e ../doc-classifier`)

## Monitoring (Grafana + Prometheus)

Workers push metrics to Redis; API aggregates them at `/metrics` for Prometheus scraping.

### Grafana Folder Structure

```
Doc Pipeline (uid: bfbjyfdf0uhhcf)
├── Workers (uid: doc-pipeline-workers-nested)         → worker dashboards
└── Workers Alerts (uid: doc-pipeline-workers-alerts-nested) → worker alerts
```

### Managing Dashboards and Alerts

```bash
# Create/update via API scripts (uses .env credentials: GRAFANA_URL, GRAFANA_TOKEN)
./monitoring/scripts/create-dashboards.sh
./monitoring/scripts/create-alerts.sh
```

Scripts are the source of truth for all alerts and dashboards. Alerts use deterministic UIDs (`dp-*` prefix) and delete-before-create for idempotency.

## Error Tracking (GlitchTip / Sentry)

All entry points integrate with GlitchTip via `sentry-sdk`. Set `DOC_PIPELINE_SENTRY_DSN` to enable. Each service uses a unique `server_name` in `sentry_sdk.init()` so errors identify the source container.

## Documentation

- `docs/API_FLOW.md` — Detailed request/response flow diagrams for all endpoints
- `docs/DEPENDENCIA_DOC_CLASSIFIER.md` — External dependency notes (doc-classifier package)
- `docs/postmortems/` — Incident postmortems

## GPU / VRAM Requirements

Production (vLLM embedded, `VLLM_EMBEDDED=true`):
- **Inference server**: ~14-20GB VRAM (EfficientNet ~200MB + EasyOCR ~2-4GB + vLLM ~10-16GB)
- **Workers**: 0 GB VRAM (stateless, CPU-only)
- Production: H200 (144GB) with `gpu-memory-utilization=0.40`, shared with ASR services

NVIDIA MPS is enabled system-wide (`systemctl status nvidia-mps`) for GPU time-slicing between services.

## Code Quality

- **Ruff**: Linter + formatter (line length 100, Python 3.12 target)
- Rules: E, F, I (isort), B (bugbear), UP (pyupgrade), SIM (simplify)
- B008 ignored (FastAPI `Depends()` in defaults)
- Pre-commit hooks run on `git commit`; manual: `pre-commit run --all-files`

## Git Commits

**IMPORTANTE**: Do NOT include `Co-Authored-By: Claude` or any mention of Claude in commit messages.

## Documentation Lookup (Context7)

Use Context7 MCP tools for up-to-date library docs:

1. `mcp__context7__resolve-library-id(libraryName="pydantic", query="...")`
2. `mcp__context7__query-docs(libraryId="/pydantic/pydantic", query="...")`

Key IDs: `/pydantic/pydantic`, `/fastapi/fastapi`, `/pytorch/pytorch`, `/huggingface/transformers`, `/python-pillow/pillow`
