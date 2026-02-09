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
Client → api.py (port 9000, no GPU) → Redis queue
                                         ↓
                          worker_docid.py × N (lightweight, ~800MB)
                            ├── EfficientNet classifier (local)
                            ├── Orientation correction (EasyOCR textbox + docTR)
                            └── VLM extraction → inference_server.py (port 9020, GPU, ~14GB)
                                                  └── Batched Qwen2.5-VL inference

Client → api.py → Redis queue → worker_ocr.py (port 9011, ~2GB VRAM)
                                  └── EasyOCR + docTR orientation
```

**Key insight**: Workers are lightweight — they do classification and preprocessing locally, then delegate VLM inference to a centralized `inference_server.py` that batches requests for higher GPU throughput. Workers communicate with the inference server via Redis (LPUSH request → poll reply key).

**Entry points**: `cli.py`, `api.py`, `worker_docid.py`, `worker_ocr.py`, `inference_server.py`

### Core Modules

```
doc_pipeline/
├── pipeline.py              # DocumentPipeline orchestrator (lazy-loads models via @property)
├── config.py                # Pydantic Settings, DOC_PIPELINE_ env prefix, get_settings() singleton
├── schemas.py               # DocumentType enum, RGData/CNHData/CINData, result models
├── classifier/adapter.py    # Wraps external doc-classifier package (EfficientNet)
├── extractors/
│   ├── base.py              # BaseExtractor ABC — extend with extract_rg/extract_cnh/extract_cin
│   ├── qwen_vl.py           # QwenVLExtractor (~16GB VRAM, uses inference server in production)
│   ├── easyocr.py           # EasyOCRExtractor (~2GB VRAM, OCR + regex)
│   └── hybrid.py            # HybridExtractor — EasyOCR + VLM with CPF validation fallback
├── preprocessing/
│   └── orientation.py       # OrientationCorrector — hybrid: EasyOCR textbox direction (90°/270°) + docTR (180°)
├── ocr/
│   ├── engine.py            # OCREngine — EasyOCR wrapper with warmup
│   └── converter.py         # PDFConverter — PDF to image via PyMuPDF (150 DPI)
├── shared/
│   ├── job_context.py       # JobContext dataclass — serializable job state through Redis
│   ├── queue.py             # QueueService — Redis LPUSH/BRPOP operations
│   ├── delivery.py          # DeliveryService — sync (Redis polling) or webhook delivery
│   ├── inference_client.py  # InferenceClient — sends VLM requests to inference server via Redis
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

| Service | Port | Description |
|---------|------|-------------|
| API | 9000 | REST API (FastAPI) |
| DocID workers 1-5 | 9010, 9012, 9014, 9016, 9018 | Worker health/metrics |
| DocID workers 6-8 | 9022, 9024, 9026 | Scale-profile workers |
| OCR worker | 9011 | OCR worker health/metrics |
| Inference server | 9020 | Centralized VLM batching |

## Docker Deployment

```bash
# Core services (Redis + API + Worker 1 + Inference Server)
docker compose up -d

# ALL workers including scale-profile workers 6-8
docker compose --profile scale up -d

# Specific scale worker
docker compose --profile scale up -d worker-docid-6

# Rebuild after code changes
docker compose build && docker compose up -d

# After changing autoscaler script
docker compose build autoscaler && docker compose up -d autoscaler
```

**IMPORTANT**: Workers 6-8 use `profiles: [scale]` — you MUST pass `--profile scale` to manage them.

Services: **redis** (queue), **api** (stateless, no GPU), **inference-server** (batched VLM, GPU ~14GB), **worker-docid-1** (always on, ~800MB), **worker-docid-2 to 8** (managed by autoscaler), **worker-ocr**, **autoscaler** (monitors queue, scales workers)

### Inference Server

The inference server (`inference_server.py`) collects VLM requests into batches for higher GPU throughput:
1. Workers LPUSH requests to `queue:doc:inference`
2. Server collects up to `INFERENCE_BATCH_SIZE` (default 4) or waits `INFERENCE_BATCH_TIMEOUT_MS` (default 100ms)
3. Processes batch in single VLM forward pass
4. Publishes replies to individual Redis keys (`inference:result:{id}`)
5. Workers poll their reply key

Config: `INFERENCE_BATCH_SIZE=8`, `WORKER_CONCURRENT_JOBS=4`, `INFERENCE_TIMEOUT=30s`

### Autoscaler

Bash script (`scripts/autoscale.sh`) running as Docker container:
- Monitors `queue:doc:documents` depth via Redis
- Scales workers up when queue ≥ `SCALE_UP_THRESHOLD` (default 5)
- Scales down after queue empty for `SCALE_DOWN_DELAY` (default 120s)
- Never stops `worker-docid-1`; manages workers 1-8
- Respects `/warmup` API warmup requests
- Exports metrics to `/tmp/autoscaler-metrics/` (mounted as volume)

## Configuration

All settings prefixed with `DOC_PIPELINE_` (see `doc_pipeline/config.py`):

| Setting | Default | Description |
|---------|---------|-------------|
| `CLASSIFIER_MODEL_TYPE` | efficientnet_b0 | b0, b2, or b4 |
| `EXTRACTOR_BACKEND` | qwen-vl | qwen-vl, easy-ocr, or hybrid |
| `CLASSIFIER_DEVICE` / `EXTRACTOR_DEVICE` | cuda:0 | cuda:N or cpu |
| `API_KEY` | - | Optional auth (requires `X-API-Key` header) |
| `INFERENCE_BATCH_SIZE` | 4 | Max requests per VLM batch |
| `WORKER_CONCURRENT_JOBS` | 4 | Parallel jobs per worker |

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

Files: dashboards in `monitoring/grafana/dashboards/`, alerts in `monitoring/grafana/alerts/`

### Adding a New Worker to Monitoring

1. Add dashboard call in `monitoring/scripts/create-dashboards.sh` (folderUid: `doc-pipeline-workers-nested`)
2. Add alert rules in `monitoring/scripts/create-alerts.sh` (folderUid: `doc-pipeline-workers-alerts-nested`)
3. Run both scripts against production Grafana

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
