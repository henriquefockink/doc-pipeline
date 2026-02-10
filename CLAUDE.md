# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

doc-pipeline is a document processing pipeline for Brazilian identity documents (RG, CNH, CIN) with two main functions:

1. **Classification + Extraction** (Worker DocID) — Classifies document type via EfficientNet (8 classes), extracts structured data via VLM (Qwen2.5-VL) or OCR+regex (EasyOCR)
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
```

## Architecture

```
                                    ┌─────────────────────────────────────────────────┐
                                    │                 GPU Node                        │
                                    │                                                 │
Client ─► api.py ─► Redis queue ──► │  inference_server.py (port 9020, GPU ~3-4GB)    │
           (9000)    (6379)         │    ├── EfficientNet classifier (~200MB)          │
           no GPU                   │    ├── EasyOCR engine (~2-4GB)                   │
              │                     │    ├── OrientationCorrector (shares EasyOCR)     │
              │                     │    ├── PDFConverter (CPU)                         │
              │                     │    └── VLLMClient (HTTP) ──► vLLM (port 8000)    │
              │                     │                               GPU ~16GB          │
              │                     │                               Qwen2.5-VL-7B      │
              │                     │                               continuous batching │
              │                     └─────────────────────────────────────────────────┘
              │
              ├──► worker_docid.py × N (stateless, no GPU)
              │      └── Delegates all inference to inference_server via Redis
              │
              └──► worker_ocr.py (stateless, no GPU)
                     └── Delegates all inference to inference_server via Redis
```

### Request Flow (full /process pipeline)

1. **Client** sends image/PDF to `api.py` via REST (`POST /process`)
2. **API** saves file to shared volume (`/tmp/doc-pipeline/`), creates job in Redis queue `queue:doc:documents`
3. **Worker DocID** picks up the job (BRPOP), sends inference request to `queue:doc:inference` via Redis LPUSH
4. **Inference Server** collects requests from `queue:doc:inference`:
   - Loads image from shared volume
   - Corrects orientation (EasyOCR textbox detection, 90/270 degree rotation)
   - Classifies document type (EfficientNet, ~200ms)
   - Sends image + prompt to **vLLM** via HTTP (`/v1/chat/completions`, OpenAI-compatible API)
   - **vLLM** processes with continuous batching + PagedAttention (tokens interleaved across requests)
   - Parses VLM JSON response, validates CPF (hybrid mode: cross-validates with EasyOCR)
   - Publishes result to Redis key `inference:result:{id}`
5. **Worker** polls the reply key, builds final response, publishes to job result key
6. **API** returns result to client (sync mode) or POSTs to webhook (async mode)

### Key Design Decisions

- **Workers are stateless and GPU-free**: all GPU work is centralized in the inference server + vLLM. Workers can scale horizontally on CPU-only nodes.
- **File passing via shared volume**: images are written to `/tmp/doc-pipeline/` (Docker named volume `temp-images`) and referenced by path through Redis, avoiding large payloads in the queue.
- **Redis as message bus**: all inter-service communication goes through Redis (queues + reply keys). No direct HTTP between workers and inference server.
- **VLM backend is swappable**: `DOC_PIPELINE_VLLM_ENABLED=true` uses vLLM (production); `false` falls back to local HuggingFace transformers (dev/testing).

**Entry points**: `cli.py`, `api.py`, `worker_docid.py`, `worker_ocr.py`, `inference_server.py`

### Documentation

The `docs/` folder contains detailed architecture and API documentation that must be kept in sync with code changes:
- `docs/API_FLOW.md` — Detailed request/response flow diagrams for all endpoints
- `docs/DEPENDENCIA_DOC_CLASSIFIER.md` — External dependency notes (doc-classifier package)

### Core Modules

```
doc_pipeline/
├── pipeline.py              # DocumentPipeline orchestrator (lazy-loads models via @property)
├── config.py                # Pydantic Settings, DOC_PIPELINE_ env prefix, get_settings() singleton
├── schemas.py               # DocumentType enum, RGData/CNHData/CINData, result models
├── classifier/adapter.py    # Wraps external doc-classifier package (EfficientNet)
├── extractors/
│   ├── base.py              # BaseExtractor ABC — extend with extract_rg/extract_cnh/extract_cin
│   ├── qwen_vl.py           # QwenVLExtractor (HuggingFace transformers, fallback when vLLM disabled)
│   ├── vllm_client.py       # VLLMClient — async HTTP client for vLLM OpenAI API (production)
│   ├── easyocr.py           # EasyOCRExtractor (~2GB VRAM, OCR + regex)
│   └── hybrid.py            # HybridExtractor — EasyOCR + VLM with CPF validation fallback
├── preprocessing/
│   └── orientation.py       # OrientationCorrector — EasyOCR textbox direction (90°/270°)
├── ocr/
│   ├── engine.py            # OCREngine — EasyOCR wrapper with warmup
│   └── converter.py         # PDFConverter — PDF to image via PyMuPDF (200 DPI)
├── shared/
│   ├── job_context.py       # JobContext dataclass — serializable job state through Redis
│   ├── queue.py             # QueueService — Redis LPUSH/BRPOP operations
│   ├── delivery.py          # DeliveryService — sync (Redis polling) or webhook delivery
│   ├── inference_client.py  # InferenceClient — sends inference requests to server via Redis
│   └── constants.py         # Queue names, TTLs, key generators
├── observability/
│   ├── metrics.py           # Prometheus metrics with middleware
│   └── worker_metrics.py    # Workers push metrics to Redis, API aggregates at /metrics
├── prompts/                 # VLM prompt templates for RG, CNH, CIN extraction
├── utils/cpf.py             # CPF validation, normalization, RG/CPF swap detection
└── auth.py                  # Optional API key auth (env keys or database)
```

### Extraction Backends

Workers support three backends (set via `DOC_PIPELINE_EXTRACTOR_BACKEND` or per-job):

| Backend | How it works | Speed | Accuracy |
|---------|-------------|-------|----------|
| `hybrid` (default) | EasyOCR for digits + VLM for structure, CPF cross-validation | ~15s | Best (CPF) |
| `vlm` | Qwen2.5-VL only | ~5s | Good |
| `ocr` | EasyOCR + regex only | ~2s | Lower |

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
| Inference server | 9020 | Centralized VLM batching |
| vLLM | 9030 | VLM server (OpenAI-compatible API) |

### Port ranges on this server (H200)

| Range | Service | Owner |
|-------|---------|-------|
| 6379 | Redis | shared |
| 8000-8099 | ASR platform (API, workers, TTS) | asr-platform |
| 8080 | cAdvisor | monitoring |
| 9000-9099 | Doc pipeline (API, workers, inference, vLLM) | doc-pipeline |
| 9100 | Node exporter | monitoring |
| 9400 | DCGM exporter | monitoring |

## Docker Deployment

```bash
# Core services (Redis + API + Inference Server + Workers) — VLM via HuggingFace
docker compose up -d

# Rebuild after code changes
docker compose build && docker compose up -d
```

### Container Architecture

| Container | Image | GPU | Description |
|-----------|-------|-----|-------------|
| **redis** | redis:7-alpine | No | Message broker + result store (AOF persistence) |
| **api** | Dockerfile.api | No | FastAPI REST API, rate limiting, job enqueuing |
| **vllm** | vllm/vllm-openai:latest | Yes (~16-20GB) | vLLM server (Qwen2.5-VL-7B), continuous batching, PagedAttention |
| **inference-server** | Dockerfile.inference-server | Yes (~3-4GB) | EfficientNet + EasyOCR + VLM proxy, batched request processing |
| **worker-docid-1..5** | Dockerfile.worker (python:3.11-slim) | No | Stateless job consumers, no ML models |
| **worker-ocr** | Dockerfile.worker (python:3.11-slim) | No | Stateless OCR job consumer, no ML models |

### Docker Volumes

| Volume | Mounted at | Purpose |
|--------|-----------|---------|
| `temp-images` | `/tmp/doc-pipeline` | Shared file passing between API, workers, inference server |
| `model-cache` | `/root/.cache/huggingface` | HuggingFace model cache (shared: inference-server + vLLM) |
| `easyocr-cache` | `/root/.EasyOCR` | EasyOCR model cache |
| `redis-data` | `/data` | Redis AOF persistence |

### Inference Server

The inference server (`inference_server.py`) centralizes all GPU models and processes requests from workers:

1. Workers LPUSH requests to `queue:doc:inference`
2. Server collects up to `INFERENCE_BATCH_SIZE` (default 4) or waits `INFERENCE_BATCH_TIMEOUT_MS` (default 100ms)
3. For each request: orientation correction (EasyOCR) → classification (EfficientNet) → VLM extraction
4. VLM backend is configurable:
   - **vLLM** (`VLLM_ENABLED=true`): async HTTP to vLLM container (`/v1/chat/completions`, OpenAI-compatible)
   - **HuggingFace** (`VLLM_ENABLED=false`): local Qwen2.5-VL via transformers (fallback/dev)
5. Publishes replies to individual Redis keys (`inference:result:{id}`)
6. Workers poll their reply key

Config: `INFERENCE_BATCH_SIZE=4`, `WORKER_CONCURRENT_JOBS=4`, `INFERENCE_TIMEOUT=120s`

## Configuration

All settings prefixed with `DOC_PIPELINE_` (see `doc_pipeline/config.py`):

| Setting | Default | Description |
|---------|---------|-------------|
| `CLASSIFIER_MODEL_TYPE` | efficientnet_b0 | b0, b2, or b4 |
| `EXTRACTOR_BACKEND` | qwen-vl | qwen-vl, easy-ocr, or hybrid |
| `CLASSIFIER_DEVICE` / `EXTRACTOR_DEVICE` | cuda:0 | cuda:N or cpu |
| `API_KEY` | - | Optional auth (requires `X-API-Key` header) |
| `INFERENCE_BATCH_SIZE` | 4 | Max requests per VLM batch |
| `INFERENCE_BATCH_TIMEOUT_MS` | 100 | Max wait to fill a batch (ms) |
| `WORKER_CONCURRENT_JOBS` | 4 | Parallel jobs per worker |
| `VLLM_ENABLED` | false | Use external vLLM server for VLM inference |
| `VLLM_BASE_URL` | http://vllm:8000/v1 | vLLM OpenAI-compatible API base URL |
| `VLLM_MODEL` | Qwen/Qwen2.5-VL-3B-Instruct | Model name served by vLLM |
| `VLLM_MAX_TOKENS` | 1024 | Max tokens for VLM generation |
| `VLLM_TIMEOUT` | 60.0 | HTTP timeout for vLLM requests (seconds) |

## Document Types

8 classes: `rg_frente`, `rg_verso`, `rg_aberto`, `rg_digital`, `cnh_frente`, `cnh_verso`, `cnh_aberta`, `cnh_digital` (plus CIN variants)

Extraction yields `RGData`, `CNHData`, or `CINData` based on document type (defined in `schemas.py`).

## Key Patterns

- **Lazy loading**: Models loaded on first access via `@property` (DocumentPipeline, workers)
- **Abstract base + factory**: `BaseExtractor` ABC; pipeline creates correct extractor from backend enum
- **Singletons**: `get_settings()`, `get_metrics()`, `get_queue_service()` return cached instances
- **JobContext dataclass**: Lightweight `@dataclass` (not Pydantic) for serializable job state through Redis
- **Shared OCR engine**: Single EasyOCR instance shared across extractors to avoid duplicate model loads
- **External dependency**: Requires `doc-classifier` package installed as editable (`pip install -e ../doc-classifier`)

## Monitoring (Grafana + Prometheus)

External Grafana at `https://speech-analytics-grafana-dev.paneas.com` (credentials in `.env`: `GRAFANA_URL`, `GRAFANA_TOKEN`).

Workers push metrics to Redis; API aggregates them at `/metrics` for Prometheus scraping.

### Grafana Folder Structure

```
Doc Pipeline (uid: bfbjyfdf0uhhcf)
├── Workers (uid: doc-pipeline-workers-nested)         → worker dashboards
└── Workers Alerts (uid: doc-pipeline-workers-alerts-nested) → worker alerts
```

### Managing Dashboards and Alerts

```bash
# Create/update via API scripts (uses .env credentials)
./monitoring/scripts/create-dashboards.sh
./monitoring/scripts/create-alerts.sh
```

Scripts are the source of truth for all alerts and dashboards. Alerts use `create-alerts.sh` with deterministic UIDs (`dp-*` prefix) and delete-before-create for idempotency.

### Adding a New Worker to Monitoring

1. Add dashboard call in `monitoring/scripts/create-dashboards.sh` (folderUid: `doc-pipeline-workers-nested`)
2. Add alert rules in `monitoring/scripts/create-alerts.sh` (folderUid: `doc-pipeline-workers-alerts-nested`)
3. Run both scripts against production Grafana

## Error Tracking (GlitchTip / Sentry)

All entry points (`api.py`, `worker_docid.py`, `worker_ocr.py`, `inference_server.py`) integrate with GlitchTip via `sentry-sdk`. Configured via:

- `DOC_PIPELINE_SENTRY_DSN` — if set, Sentry is enabled; if empty, it's a no-op
- `DOC_PIPELINE_SENTRY_ENVIRONMENT` — default `production`
- `DOC_PIPELINE_SENTRY_TRACES_SAMPLE_RATE` — default `0.1`

Each service uses a unique `server_name` in `sentry_sdk.init()` so errors in GlitchTip identify the source container. Handled exceptions are reported via `sentry_sdk.capture_exception(e)` in all worker error handlers.

## GPU / VRAM Requirements

With vLLM (default, production):
- **vLLM container**: ~16-20GB VRAM (Qwen2.5-VL-7B + KV cache, depends on `gpu-memory-utilization`)
- **Inference server**: ~3-4GB VRAM (EfficientNet ~200MB + EasyOCR ~2-4GB)
- **Workers**: 0 GB VRAM (stateless, CPU-only)
- **Total**: ~20-24GB minimum (single GPU)

Without vLLM (HuggingFace fallback, set `VLLM_ENABLED=false`):
- **Inference server**: ~20-24GB VRAM (classifier + EasyOCR + Qwen2.5-VL-7B via transformers)
- **Workers**: 0 GB VRAM
- **Total**: ~20-24GB minimum

Production: H200 (144GB) shared with ASR services.

## Technology Stack and Infrastructure Dependencies

| Component | Technology | Purpose | GPU Required |
|-----------|-----------|---------|:------------:|
| VLM | **Qwen2.5-VL-7B-Instruct** (7B param vision-language model) | Document data extraction | Yes |
| VLM Server | **vLLM** (continuous batching, PagedAttention, CUDA graphs) | High-throughput VLM inference | Yes (NVIDIA, CUDA 12+) |
| Classifier | **EfficientNet-B0** (PyTorch, ~200MB) | Document type classification (12 classes) | Yes |
| OCR Engine | **EasyOCR** (PyTorch + CRAFT text detection) | Text extraction, orientation correction | Yes |
| PDF Rendering | **PyMuPDF** (CPU) | PDF → image conversion (200 DPI) | No |
| API Framework | **FastAPI** + Uvicorn | REST API, rate limiting (SlowAPI) | No |
| Message Broker | **Redis 7** (AOF persistence) | Job queue, result store, inter-service messaging | No |
| GPU Runtime | **NVIDIA CUDA 12.8** + PyTorch 2.7 | GPU acceleration for all ML models | Yes |
| Container Runtime | **Docker** + Docker Compose (GPU passthrough via nvidia-container-toolkit) | Service orchestration | Host GPU |
| Container Images | `vllm/vllm-openai:latest` (vLLM), `python:3.11-slim` (workers), custom (inference-server, API) | Base images | - |
| Monitoring | **Prometheus** (metrics) + **Grafana** (dashboards/alerts) | Observability | No |
| Error Tracking | **Sentry SDK** → GlitchTip | Exception reporting | No |
| Model Hub | **HuggingFace Hub** (model downloads, `HF_TOKEN` auth) | ML model distribution | No |

### Infrastructure Requirements

- **GPU**: NVIDIA GPU with CUDA 12+ support (tested: RTX 5090, H200). vLLM requires compute capability sm_80+ (Ampere or newer).
- **VRAM**: Minimum 24GB for full stack with vLLM (7B model). Production recommended: 32GB+ with `gpu-memory-utilization=0.40-0.70`.
- **Docker**: Docker Engine with `nvidia-container-toolkit` for GPU passthrough. Compose v2+.
- **Network**: Internal Docker network (`doc-pipeline`). Only API port (9000) and optionally vLLM (8000) exposed externally.
- **Storage**: Shared volume (`temp-images`) for file passing between containers. Model cache volumes for HuggingFace and EasyOCR models (~5-10GB).
- **RAM**: ~8GB for vLLM container, ~2GB for inference server, ~200MB per worker. Total ~12GB system RAM.

## GPU Sharing (MPS)

NVIDIA MPS is enabled system-wide (`systemctl status nvidia-mps`) for fair GPU time-slicing between doc-pipeline, ASR, and other services on the H200. Check parent `../CLAUDE.md` for details.

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
