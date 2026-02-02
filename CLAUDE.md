# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

doc-pipeline is a document classification and data extraction pipeline for Brazilian identity documents (RG and CNH). It uses a two-stage architecture:
1. **Classification**: EfficientNet model classifies document type (8 classes)
2. **Extraction**: VLM (Qwen2.5-VL or GOT-OCR) extracts structured data via prompts

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Install dev dependencies (includes pre-commit and ruff)
pip install -e ".[dev]"

# Setup pre-commit hooks
pre-commit install

# Run CLI - full pipeline
python cli.py documento.jpg -m models/classifier.pth

# Run CLI - classification only
python cli.py documento.jpg -m models/classifier.pth --no-extraction

# Run CLI - with GOT-OCR backend (lower VRAM)
python cli.py documento.jpg -m models/classifier.pth --backend got-ocr

# Run API server (uses models/classifier.pth by default)
python api.py

# Run API server with ngrok (persists after SSH disconnect)
./start-server.sh                              # Start with random ngrok URL
./start-server.sh start meu-dominio.ngrok.io   # Start with custom domain (paid ngrok)
./start-server.sh status                       # Check status
./stop-server.sh                               # Stop all services
```

## Architecture

```
doc_pipeline/
├── pipeline.py         # DocumentPipeline - main orchestrator, lazy-loads models
├── config.py           # Pydantic Settings with DOC_PIPELINE_ env prefix
├── schemas.py          # DocumentType enum, RGData, CNHData, result models
├── classifier/
│   └── adapter.py      # Wraps external doc-classifier package
├── extractors/
│   ├── base.py         # BaseExtractor abstract class
│   ├── qwen_vl.py      # QwenVLExtractor (~16GB VRAM)
│   └── got_ocr.py      # GOTOCRExtractor (~2GB VRAM)
└── prompts/
    ├── rg.py           # RG extraction prompt template
    └── cnh.py          # CNH extraction prompt template
```

**Entry points**: `cli.py` (command-line), `api.py` (FastAPI REST server), `worker.py` (queue worker)

## Port Allocation

doc-pipeline uses the **9000 port range** to avoid conflicts with other services:

| Service | Port | Description |
|---------|------|-------------|
| API | 9000 | REST API (FastAPI) |
| Worker Health | 9010 | Worker health check endpoint |
| Redis | 6379 | Queue backend (internal) |

## Docker Deployment

```bash
# Start all services (Redis + API + Worker)
docker compose up -d

# View logs
docker compose logs -f

# Scale workers (if needed)
docker compose up -d --scale worker=2

# Stop services
docker compose down
```

Services:
- **redis**: Queue backend
- **api**: Stateless API that enqueues jobs (no GPU)
- **worker**: Processes jobs from queue (GPU required)

## GPU Sharing (MPS)

This server runs multiple GPU services (doc-pipeline, ASR, TTS, etc.) on the same NVIDIA H200. To avoid GPU contention and ensure fair time-slicing, we use **NVIDIA MPS (Multi-Process Service)**.

### Why MPS?

| Mode | VRAM Access | Concurrency | Use Case |
|------|-------------|-------------|----------|
| Default | Shared | Serialized (unfair) | Single service |
| **MPS** | **Shared (full)** | **Fair time-slicing** | **Multiple services** |
| MIG | Partitioned (fixed) | Parallel (isolated) | Fixed workloads <40GB |

MPS was chosen because:
- Models can use **full VRAM** (some need >40GB, ruling out MIG)
- **Fair scheduling** between services (ASR won't stall while OCR processes)
- Works with **any CUDA application** without code changes

### MPS Service

MPS is configured as a systemd service that starts on boot:

```bash
# Check MPS status
systemctl status nvidia-mps

# Verify processes are using MPS (look for M+C type)
nvidia-smi
# Type "M+C" = MPS + Compute (correct)
# Type "C" = Compute only (MPS not active)

# Restart MPS (requires stopping all GPU processes first)
sudo systemctl restart nvidia-mps
```

Service file: `/etc/systemd/system/nvidia-mps.service`

### Scaling Workers

With MPS, you can safely scale doc-pipeline workers:

```bash
# Scale to 2 workers (recommended for high load)
docker compose up -d --scale worker=2

# Check GPU memory before scaling more
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

Each worker uses ~17GB VRAM. The H200 (143GB) can handle multiple workers plus ASR services.

## Key Patterns

- **Lazy loading**: DocumentPipeline loads classifier/extractor only when first used (via `@property`)
- **Abstract base**: New extractors extend `BaseExtractor` with `extract()` method
- **Pydantic validation**: All data models use Pydantic; config uses pydantic-settings
- **External dependency**: Requires `doc-classifier` package installed as editable (`-e ../doc-classifier`)

## Configuration

Default model path: `models/classifier.pth`

Key settings (all prefixed with `DOC_PIPELINE_`):
- `CLASSIFIER_MODEL_TYPE`: efficientnet_b0, efficientnet_b2, efficientnet_b4
- `EXTRACTOR_BACKEND`: qwen-vl (default) or got-ocr
- `CLASSIFIER_DEVICE` / `EXTRACTOR_DEVICE`: cuda:N or cpu (supports multi-GPU)
- `API_KEY`: Optional API key for authentication (if set, requires `X-API-Key` header)

## Document Types

8 classes: `rg_frente`, `rg_verso`, `rg_aberto`, `rg_digital`, `cnh_frente`, `cnh_verso`, `cnh_aberta`, `cnh_digital`

Extraction yields `RGData` or `CNHData` based on document type (defined in `schemas.py`).

## Code Quality

Pre-commit hooks ensure consistent code style:
- **Ruff**: Linter and formatter (replaces black, isort, flake8)
- Run `pre-commit run --all-files` to check all files manually
- Hooks run automatically on `git commit`

## Documentation Lookup (Context7)

Always use Context7 MCP tools when searching for library documentation and best practices. This ensures you have up-to-date information beyond the knowledge cutoff.

### When to Use Context7

- Looking up API usage for libraries (Pydantic, FastAPI, PyTorch, Transformers, etc.)
- Checking current best practices or migration guides
- Finding code examples for specific functionality
- Verifying correct syntax or parameters for library functions

### How to Use

1. **Resolve the library ID first**:
   ```
   mcp__context7__resolve-library-id(libraryName="pydantic", query="how to define model with validators")
   ```

2. **Query the documentation**:
   ```
   mcp__context7__query-docs(libraryId="/pydantic/pydantic", query="how to define model with validators")
   ```

### Key Libraries for This Project

| Library | Typical Context7 ID |
|---------|---------------------|
| Pydantic | `/pydantic/pydantic` |
| FastAPI | `/fastapi/fastapi` |
| PyTorch | `/pytorch/pytorch` |
| Transformers | `/huggingface/transformers` |
| Pillow | `/python-pillow/pillow` |

### Tips

- Be specific in your query to get relevant results
- Limit to 3 calls per question to avoid excessive lookups
- If the user provides a library ID directly (e.g., `/org/project`), skip `resolve-library-id`
