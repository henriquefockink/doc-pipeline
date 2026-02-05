# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

doc-pipeline é um pipeline de processamento de documentos brasileiros (RG e CNH) com duas funcionalidades principais:

1. **Classificação + Extração** (Worker DocID)
   - Classifica tipo do documento usando EfficientNet (8 classes)
   - Extrai dados estruturados usando VLM (Qwen2.5-VL) ou OCR+regex (EasyOCR)
   - Use case: Processar documentos de identidade para obter dados como nome, CPF, data de nascimento

2. **OCR Genérico** (Worker OCR)
   - Extrai texto de qualquer PDF ou imagem usando EasyOCR
   - Suporte a português com diacríticos (ç, ã, á, etc.)
   - Use case: Extrair texto de contratos, faturas, documentos diversos

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

# Run CLI - with EasyOCR backend (lower VRAM)
python cli.py documento.jpg -m models/classifier.pth --backend easy-ocr

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
│   └── easyocr.py      # EasyOCRExtractor (~2GB VRAM)
└── prompts/
    ├── rg.py           # RG extraction prompt template
    └── cnh.py          # CNH extraction prompt template
```

**Entry points**: `cli.py` (command-line), `api.py` (FastAPI REST server), `worker_docid.py` (classification worker), `worker_ocr.py` (OCR worker)

## API Endpoints

| Endpoint | Method | Description | Worker |
|----------|--------|-------------|--------|
| `/classify` | POST | Classifica tipo do documento (RG/CNH) | worker |
| `/extract` | POST | Extrai dados estruturados | worker |
| `/process` | POST | Classifica + extrai (pipeline completo) | worker |
| `/ocr` | POST | OCR genérico de PDF/imagem | worker-ocr |
| `/health` | GET | Health check da API | - |
| `/metrics` | GET | Métricas Prometheus | - |

## Port Allocation

doc-pipeline uses the **9000 port range** to avoid conflicts with other services:

| Service | Port | Description |
|---------|------|-------------|
| API | 9000 | REST API (FastAPI) |
| Worker Health | 9010 | Classification worker health |
| Worker OCR Health | 9011 | OCR worker health |
| Redis | 6379 | Queue backend (internal) |

## Docker Deployment

```bash
# Start all services (Redis + API + Workers)
docker compose up -d

# View logs
docker compose logs -f

# Scale classification workers (if needed)
docker compose up -d --scale worker=2

# Stop services
docker compose down
```

Services:
- **redis**: Queue backend (filas separadas por worker)
- **api**: Stateless API que enfileira jobs (sem GPU)
- **worker-docid**: Worker de classificação/extração (RG/CNH)
- **worker-ocr**: Worker de OCR genérico

## Workers

O sistema usa workers separados para diferentes tipos de processamento:

### Worker DocID (`worker_docid.py`)

Processa documentos de identidade (RG e CNH).

| Característica | Valor |
|----------------|-------|
| Fila Redis | `queue:doc:documents` |
| Porta métricas | 9010 |
| Job Prometheus | `doc-pipeline-worker-docid` |
| Modelos | EfficientNet (classifier) + Qwen2.5-VL (extractor) |
| VRAM | ~16GB |
| Tempo/job | ~3s |

**Operações:**
- `classify` - Classifica tipo do documento (8 classes: rg_frente, rg_verso, cnh_frente, etc.)
- `extract` - Extrai dados estruturados (nome, CPF, data nascimento, etc.)
- `process` - Classifica + extrai em uma única chamada

**Exemplo de uso:**
```bash
curl -X POST http://localhost:9000/process \
  -F "arquivo=@documento.jpg" \
  -H "X-API-Key: $API_KEY"
```

### Worker OCR (`worker_ocr.py`)

OCR genérico para qualquer PDF ou imagem.

| Característica | Valor |
|----------------|-------|
| Fila Redis | `queue:doc:ocr` |
| Porta métricas | 9011 |
| Job Prometheus | `doc-pipeline-worker-ocr` |
| Modelo | EasyOCR (português) |
| VRAM | ~2GB |
| Tempo/job | ~40ms (imagem simples), ~4s/página (PDF) |

**Características:**
- Suporte a PDF multi-página (converte para imagem a 150 DPI)
- Bom reconhecimento de português (diacríticos: ç, ã, á, etc.)
- GPU opcional (mais rápido com CUDA)

**Exemplo de uso:**
```bash
curl -X POST http://localhost:9000/ocr \
  -F "arquivo=@contrato.pdf" \
  -F "max_pages=5" \
  -H "X-API-Key: $API_KEY"
```

### Métricas dos Workers

Todos os workers expõem métricas Prometheus em `/metrics`:

| Worker | Porta |
|--------|-------|
| DocID 1 | 9010 |
| DocID 2 | 9012 |
| DocID 3 | 9014 |
| DocID 4 | 9016 |
| DocID 5 | 9018 |
| OCR | 9011 |

**Métricas principais:**
- `doc_pipeline_jobs_processed_total{operation, status, delivery_mode}` - Jobs processados
- `doc_pipeline_worker_processing_seconds{operation}` - Tempo de processamento
- `doc_pipeline_queue_depth` - Profundidade da fila
- `doc_pipeline_queue_wait_seconds` - Tempo de espera na fila

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
- `EXTRACTOR_BACKEND`: qwen-vl (default) or easy-ocr
- `CLASSIFIER_DEVICE` / `EXTRACTOR_DEVICE`: cuda:N or cpu (supports multi-GPU)
- `API_KEY`: Optional API key for authentication (if set, requires `X-API-Key` header)

## Document Types

8 classes: `rg_frente`, `rg_verso`, `rg_aberto`, `rg_digital`, `cnh_frente`, `cnh_verso`, `cnh_aberta`, `cnh_digital`

Extraction yields `RGData` or `CNHData` based on document type (defined in `schemas.py`).

## Monitoring (Grafana + Prometheus)

O monitoramento usa Grafana e Prometheus **externos** (não locais):

- **Grafana**: https://speech-analytics-grafana-dev.paneas.com
- **Credenciais**: Configuradas em `.env` (`GRAFANA_URL`, `GRAFANA_TOKEN`)

O Prometheus de produção roda fora desta máquina e scrapea as métricas via rede:
- API: porta 9000
- Workers DocID: portas 9010, 9012, 9014, 9016, 9018
- Worker OCR: porta 9011

As métricas do autoscaler são expostas via `/metrics` da API.

### Autoscaler (Container)

O autoscaler roda como **container Docker** junto com o stack:

```bash
# Status
docker compose ps autoscaler

# Logs
docker compose logs -f autoscaler

# Reiniciar (após alterar scripts/autoscale.sh)
docker compose build autoscaler && docker compose up -d autoscaler
```

**Configuração** (via environment no `docker-compose.yml`):
- `MIN_WORKERS=1`: mínimo de workers
- `MAX_WORKERS=3`: limite normal de scaling
- `SCALE_UP_THRESHOLD=5`: queue depth para escalar
- `SCALE_DOWN_DELAY=120`: segundos antes de desescalar
- Warmup pode solicitar até 5 workers via API `/warmup`

**Workers disponíveis** (definidos em `scripts/autoscale.sh`):
- `worker-docid-1` a `worker-docid-5`

### Estrutura de Pastas no Grafana

```
📁 Doc Pipeline (uid: bfbjyfdf0uhhcf)           # Alertas gerais do pipeline
   📁 Workers (uid: doc-pipeline-workers-nested)        # Dashboards específicos de workers
   📁 Workers Alerts (uid: doc-pipeline-workers-alerts-nested)  # Alertas específicos de workers
```

Ao adicionar um novo worker, seguir este padrão:
- Dashboard vai em `Doc Pipeline / Workers` (folderUid: `doc-pipeline-workers-nested`)
- Alertas vão em `Doc Pipeline / Workers Alerts` (folderUid: `doc-pipeline-workers-alerts-nested`)

### Estrutura de Arquivos

```
monitoring/
├── grafana/
│   ├── dashboards/             # JSONs de dashboards (backup/modelo)
│   │   └── doc-pipeline.json   # Dashboard Overview
│   └── alerts/                 # YAMLs de alertas (backup/modelo)
│       └── doc-pipeline-alerts.yaml
└── scripts/
    ├── create-dashboards.sh    # Cria dashboards via API Grafana
    └── create-alerts.sh        # Cria alertas via API Grafana
```

**Onde guardar novos arquivos:**
- **Dashboards**: `monitoring/grafana/dashboards/`
- **Alertas**: `monitoring/grafana/alerts/`

### Criar/Atualizar Dashboards e Alertas via API

**IMPORTANTE**: Sempre usar os scripts para criar dashboards e alertas no Grafana de produção!

```bash
# Criar dashboards (usa .env para credenciais)
./monitoring/scripts/create-dashboards.sh

# Criar alertas (usa .env para credenciais)
./monitoring/scripts/create-alerts.sh

# Com argumentos explícitos
./monitoring/scripts/create-dashboards.sh https://grafana.example.com glsa_xxx
./monitoring/scripts/create-alerts.sh https://grafana.example.com glsa_xxx
```

Variáveis de ambiente (configuradas em `.env`):
- `GRAFANA_URL` - URL do Grafana (produção: https://speech-analytics-grafana-dev.paneas.com)
- `GRAFANA_TOKEN` - Token de API (glsa_xxx) ou user:password

### Processo para Adicionar Novo Worker

1. **Criar dashboard** em `monitoring/scripts/create-dashboards.sh`:
   - Adicionar chamada `create_dashboard "Worker X" "worker-x" "$FOLDER_WORKERS" '{...}'`
   - Painéis padrão: Queue Depth, Jobs/s, P95 Latency, Error Rate, Jobs (24h), Worker Status
   - Seções: Overview, Processing Metrics, Queue, Delivery

2. **Criar alertas** em `monitoring/scripts/create-alerts.sh`:
   - Adicionar seção "Criando alertas do X Worker..."
   - Alertas padrão: Worker Down, High Error Rate, High Latency, Queue Backup

3. **Executar scripts** no Grafana de produção:
   ```bash
   ./monitoring/scripts/create-dashboards.sh
   ./monitoring/scripts/create-alerts.sh
   ```

4. **Verificar** no Grafana:
   - Dashboard: https://speech-analytics-grafana-dev.paneas.com/d/worker-x/worker-x
   - Alertas: https://speech-analytics-grafana-dev.paneas.com/alerting/list

### Dashboards Disponíveis

| Dashboard | Pasta | UID | Descrição |
|-----------|-------|-----|-----------|
| Doc Pipeline - Overview | Doc Pipeline | doc-pipeline-overview | Overview geral, auto-scaler, métricas de negócio |
| Worker DocID | Doc Pipeline / Workers | worker-docid | Queue, processing time, confidence, document types |
| Worker OCR | Doc Pipeline / Workers | worker-ocr | Queue, processing time, delivery, errors |

### Alertas Configurados

**Doc Pipeline (alertas gerais):**
- High Error Rate (5xx), High Latency P95/P99, High Concurrency
- Classification Confidence, Queue Depth, Worker Errors, Webhook Failures

**Doc Pipeline / Workers Alerts:**

| Alerta | Worker | Condição |
|--------|--------|----------|
| DocID Worker Down | docid | Worker unreachable > 2min |
| DocID High Error Rate | docid | Error rate > 10% |
| DocID High Latency | docid | P95 > 30s |
| DocID Queue Backup | docid | Queue > 10 jobs |
| DocID Low Confidence | docid | Median confidence < 70% |
| OCR Worker Down | ocr | Worker unreachable > 2min |
| OCR High Error Rate | ocr | Error rate > 10% |
| OCR High Latency | ocr | P95 > 30s |
| OCR Queue Backup | ocr | Queue > 10 jobs |

## Code Quality

Pre-commit hooks ensure consistent code style:
- **Ruff**: Linter and formatter (replaces black, isort, flake8)
- Run `pre-commit run --all-files` to check all files manually
- Hooks run automatically on `git commit`

## Git Commits

**IMPORTANTE**: Ao criar commits, NÃO incluir `Co-Authored-By: Claude` ou qualquer menção ao Claude na mensagem de commit. Commits devem aparecer como se fossem feitos apenas pelo desenvolvedor.

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
