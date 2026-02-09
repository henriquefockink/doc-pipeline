# doc-pipeline

Pipeline de processamento de documentos brasileiros (RG, CNH e CIN) com classificação automática e extração de dados estruturados.

## Funcionalidades

- **Classificação** de 12 tipos de documentos via EfficientNet (rg/cnh/cin x frente/verso/aberto/digital)
- **Extração de dados** estruturados via VLM (Qwen2.5-VL-3B), OCR+regex (EasyOCR) ou modo híbrido
- **OCR genérico** para PDFs e imagens, com suporte a português
- **Correção automática de orientação** via EasyOCR (detecção de textbox, rotação 90°/270°)
- **Validação de CPF** com detecção de swap RG↔CPF
- **Batched inference** via servidor centralizado com vLLM (continuous batching) para alto throughput
- **Workers stateless** sem GPU — escaláveis horizontalmente em nós CPU-only

## Arquitetura

```
                              ┌──────────────────────────────────────────────────┐
                              │                   GPU Node                       │
                              │                                                  │
                              │  inference_server.py (port 9020, GPU ~3-4GB)     │
                              │    ├── EfficientNet classifier (~200MB)           │
                              │    ├── EasyOCR engine (~2-4GB)                    │
                              │    ├── OrientationCorrector (shares EasyOCR)      │
                              │    ├── PDFConverter (CPU)                          │
                              │    └── VLLMClient (HTTP) ──► vLLM (port 8000)     │
                              │                               GPU ~8-10GB         │
                              │                               Qwen2.5-VL-3B       │
                              │                               continuous batching  │
                              └──────────────────────────────────────────────────┘
                                            ▲                    │
                                  request   │                    │  reply
                                  (Redis)   │                    │  (Redis)
┌────────┐    ┌──────────────┐              │                    │
│ Client │───►│  API :9000   │──► Redis ────┘                    │
│        │    │  (sem GPU)   │    queue               ┌──────────▼───────────┐
└────────┘    └──────────────┘                        │                      │
                                           ┌──────────┴──────────────────┐   │
                                           │  worker_docid × N (sem GPU) │◄──┘
                                           │  worker_ocr (sem GPU)       │
                                           └─────────────────────────────┘
```

### Fluxo Principal (POST /process)

1. **Client** envia imagem/PDF para `api.py` via REST
2. **API** salva arquivo em volume compartilhado (`/tmp/doc-pipeline/`), enfileira job no Redis (`queue:doc:documents`)
3. **Worker DocID** consome job (BRPOP), envia request de inferência para `queue:doc:inference` via Redis LPUSH
4. **Inference Server** processa a request:
   - Carrega imagem do volume compartilhado
   - Corrige orientação (EasyOCR: detecção de textbox, rotação 90°/270°)
   - Classifica tipo do documento (EfficientNet, ~100ms)
   - Envia imagem + prompt ao **vLLM** via HTTP (`/v1/chat/completions`, API compatível com OpenAI)
   - **vLLM** processa com continuous batching + PagedAttention (tokens intercalados entre requests)
   - Parseia JSON do VLM, valida CPF (modo hybrid: cross-valida com EasyOCR)
   - Publica resultado na key Redis `inference:result:{id}`
5. **Worker** consulta a reply key, monta resposta final, publica no job result
6. **API** retorna resultado ao client (modo sync) ou POSTa no webhook (modo async)

> Para o fluxo detalhado de cada etapa, veja [docs/API_FLOW.md](docs/API_FLOW.md).

### Decisões de Design

- **Workers stateless e sem GPU**: todo processamento GPU é centralizado no inference server + vLLM. Workers escalam horizontalmente em nós CPU-only.
- **File passing via volume compartilhado**: imagens são escritas em `/tmp/doc-pipeline/` (volume Docker `temp-images`) e referenciadas por path via Redis, evitando payloads grandes na fila.
- **Redis como message bus**: toda comunicação inter-serviço vai pelo Redis (filas + reply keys). Não há HTTP direto entre workers e inference server.
- **Backend VLM swappable**: `DOC_PIPELINE_VLLM_ENABLED=true` usa vLLM (produção); `false` usa HuggingFace transformers local (dev/fallback).

## Quick Start

### Docker (Recomendado)

```bash
# Configurar variáveis
cp .env.example .env
# Editar .env: adicionar HF_TOKEN (obrigatório para download dos modelos)

# Stack completo COM vLLM (produção — continuous batching)
docker compose --profile vllm up -d

# Stack sem vLLM (dev — HuggingFace transformers local)
docker compose up -d

# Ver logs
docker compose logs -f

# Rebuild após mudanças no código
docker compose build && docker compose --profile vllm up -d

# Parar
docker compose down
```

Serviços disponíveis:
- **API**: http://localhost:9000 — REST API (FastAPI, sem GPU)
- **vLLM**: http://localhost:8000 — VLM server com continuous batching (GPU, profile `vllm`)
- **Inference Server**: http://localhost:9020 — EfficientNet + EasyOCR + proxy VLM (GPU)
- **Workers DocID 1-5**: Consumidores de fila stateless (sem GPU)
- **Worker OCR**: OCR genérico stateless (sem GPU)
- **Redis**: Message broker + result store

### Local (Desenvolvimento)

```bash
# Ambiente virtual
python3 -m venv venv
source venv/bin/activate

# Dependências
pip install -r requirements.txt

# Modelo do classificador
cp /caminho/para/modelo.pth models/classifier.pth

# API local
python api.py
```

## Uso

### CLI

```bash
# Pipeline completo (classificação + extração)
python cli.py documento.jpg

# Com EasyOCR (menor VRAM)
python cli.py documento.jpg --backend easy-ocr

# Apenas classificar
python cli.py documento.jpg --no-extraction

# Processar pasta com saída JSON
python cli.py ./documentos/ --json -o resultados.json
```

### API

```bash
# Pipeline completo
curl -X POST http://localhost:9000/process \
  -H "X-API-Key: $API_KEY" \
  -F "arquivo=@documento.jpg"

# Apenas classificar
curl -X POST http://localhost:9000/classify \
  -H "X-API-Key: $API_KEY" \
  -F "arquivo=@documento.jpg"

# Extrair dados (tipo conhecido)
curl -X POST "http://localhost:9000/extract?doc_type=rg_frente" \
  -H "X-API-Key: $API_KEY" \
  -F "arquivo=@rg.jpg"

# OCR genérico
curl -X POST http://localhost:9000/ocr \
  -H "X-API-Key: $API_KEY" \
  -F "arquivo=@contrato.pdf" \
  -F "max_pages=5"

# Health check (sem auth)
curl http://localhost:9000/health
```

### Python

```python
from doc_pipeline import DocumentPipeline

pipeline = DocumentPipeline(
    classifier_model_path="models/classifier.pth",
    extractor_backend="qwen-vl",
)

result = pipeline.process("documento.jpg")
print(f"Tipo: {result.document_type.value}")
print(f"Confiança: {result.classification.confidence:.1%}")
if result.data:
    print(f"Nome: {result.data.nome}")
    print(f"CPF: {result.data.cpf}")
```

## Endpoints

| Método | Endpoint | Descrição | Auth |
|--------|----------|-----------|:----:|
| POST | `/process` | Pipeline completo (classificação + extração) | API Key |
| POST | `/classify` | Apenas classificação | API Key |
| POST | `/extract?doc_type=rg_frente` | Apenas extração (tipo conhecido) | API Key |
| POST | `/ocr` | OCR genérico de PDF/imagem | API Key |
| GET | `/jobs/{id}/status` | Status de um job (sync polling) | API Key |
| POST | `/warmup` | Pré-escalar workers | Warmup Key |
| GET | `/warmup/status` | Status do warmup | Warmup Key |
| DELETE | `/warmup` | Cancelar warmup | Warmup Key |
| GET | `/health` | Status da API | Não |
| GET | `/metrics` | Métricas Prometheus | Não |
| GET | `/classes` | Lista classes suportadas | API Key |

### Modos de Entrega

| Modo | Comportamento |
|------|---------------|
| `sync` (default) | API faz polling no Redis até resultado (max 5min) |
| `webhook` | Retorna 202 imediatamente, POSTa resultado na URL informada |

### Resposta (POST /process)

```json
{
  "file_path": null,
  "classification": {
    "document_type": "rg_aberto",
    "confidence": 0.97
  },
  "extraction": {
    "document_type": "rg_aberto",
    "data": {
      "nome": "JOAO DA SILVA",
      "cpf": "123.456.789-00",
      "rg": "12.345.678-9",
      "data_nascimento": "15/03/1985",
      "nome_pai": "JOSE DA SILVA",
      "nome_mae": "MARIA DA SILVA",
      "naturalidade": "SAO PAULO-SP",
      "data_expedicao": "10/05/2020",
      "orgao_expedidor": "SSP-SP"
    },
    "backend": "hybrid"
  },
  "image_correction": {
    "was_corrected": true,
    "rotation_applied": 90,
    "correction_method": "easyocr_textbox",
    "confidence": 0.98
  },
  "success": true,
  "error": null
}
```

## Backends de Extração

| Backend | Estratégia | Velocidade | Precisão |
|---------|-----------|:----------:|:--------:|
| **hybrid** (default) | VLM extrai dados + EasyOCR valida CPF com algoritmo | ~5s | Melhor (CPF) |
| **vlm** | Qwen2.5-VL extrai tudo via prompt -> JSON | ~2.5s | Boa |
| **ocr** | EasyOCR + parsing com regex | ~2s | Menor |

O backend **hybrid** é o padrão em produção:
1. VLM (Qwen) extrai dados estruturados diretamente da imagem
2. Valida CPF usando o algoritmo oficial (módulo 11)
3. Se CPF inválido, faz fallback para EasyOCR e tenta corrigir
4. Detecta e corrige swap entre campos RG e CPF

## Documentos Suportados

### Tipos (12 classes)

| Documento | Variantes |
|-----------|-----------|
| **RG** (Registro Geral) | `rg_frente`, `rg_verso`, `rg_aberto`, `rg_digital` |
| **CNH** (Carteira Nacional de Habilitação) | `cnh_frente`, `cnh_verso`, `cnh_aberta`, `cnh_digital` |
| **CIN** (Carteira de Identidade Nacional) | `cin_frente`, `cin_verso`, `cin_aberta`, `cin_digital` |

### Campos Extraídos

**RG**: nome, nome_pai, nome_mae, data_nascimento, naturalidade, cpf, rg, data_expedicao, orgao_expedidor

**CNH**: nome, cpf, data_nascimento, doc_identidade, numero_registro, numero_espelho, validade, categoria, observacoes, primeira_habilitacao

**CIN**: nome, nome_pai, nome_mae, data_nascimento, naturalidade, cpf, data_expedicao, orgao_expedidor

## Preprocessamento

### Correção de Orientação (EasyOCR)

Antes de classificar e extrair, o pipeline corrige automaticamente a orientação da imagem:

1. **EXIF metadata** — Corrige rotação salva pela câmera
2. **EasyOCR textbox detection** — Detecta caixas de texto e infere orientação predominante (horizontal vs vertical). Se a maioria dos textboxes são mais altos que largos, a imagem está rotacionada 90° ou 270°.
3. Aplica rotação se necessário

Isso resolve documentos fotografados de lado ou de cabeça para baixo.

## Infraestrutura

### Containers Docker

| Container | Imagem Base | GPU | Descrição |
|-----------|------------|:---:|-----------|
| **redis** | redis:7-alpine | Não | Message broker + result store (AOF) |
| **api** | Dockerfile.api | Não | FastAPI REST API, rate limiting, enfileiramento |
| **vllm** | vllm/vllm-openai:latest | Sim (~8-10GB) | vLLM server, continuous batching, PagedAttention |
| **inference-server** | Dockerfile.inference-server | Sim (~3-4GB) | EfficientNet + EasyOCR + proxy VLM |
| **worker-docid-1..5** | Dockerfile.worker (python:3.11-slim) | Não | Consumidores de fila stateless |
| **worker-ocr** | Dockerfile.worker (python:3.11-slim) | Não | Consumidor OCR stateless |

### Portas

| Serviço | Porta | Descrição |
|---------|-------|-----------|
| API | 9000 | REST API (FastAPI) |
| vLLM | 8000 | VLM server (OpenAI-compatible API) |
| Workers DocID 1-5 | 9010, 9012, 9014, 9016, 9018 | Health + métricas |
| Worker OCR | 9011 | Health + métricas |
| Inference Server | 9020 | Saúde + métricas do servidor centralizado |

### Inference Server (Batching + vLLM)

O inference server centraliza todos os modelos GPU e processa requests dos workers:

```
Workers enviam requests → Redis queue → Inference Server:
                                          ├── Orientação (EasyOCR textbox)
                                          ├── Classificação (EfficientNet)
                                          └── Extração VLM ──► vLLM (HTTP)
                                                                  │
                                                    Continuous batching
                                                    + PagedAttention
                                                                  │
                                                          Replies ──► Redis keys ──► Workers
```

| Config | Default | Descrição |
|--------|---------|-----------|
| `INFERENCE_BATCH_SIZE` | 4 | Max requests por batch |
| `INFERENCE_BATCH_TIMEOUT_MS` | 100 | Max espera para encher batch |
| `WORKER_CONCURRENT_JOBS` | 4 | Jobs paralelos por worker |
| `VLLM_ENABLED` | false | Usar vLLM externo (true) ou HuggingFace local (false) |

### Stack Tecnológico e Dependências de Infraestrutura

| Componente | Tecnologia | Função | GPU |
|------------|-----------|--------|:---:|
| VLM | **Qwen2.5-VL-3B-Instruct** (3B params, vision-language) | Extração de dados de documentos | Sim |
| Servidor VLM | **vLLM** (continuous batching, PagedAttention, CUDA graphs) | Inferência VLM de alto throughput | Sim (NVIDIA, CUDA 12+) |
| Classificador | **EfficientNet-B0** (PyTorch, ~200MB) | Classificação de tipo de documento | Sim |
| OCR | **EasyOCR** (PyTorch + CRAFT text detection) | Extração de texto, correção de orientação | Sim |
| PDF | **PyMuPDF** (CPU) | Conversão PDF → imagem (200 DPI) | Não |
| API | **FastAPI** + Uvicorn | REST API, rate limiting (SlowAPI) | Não |
| Message Broker | **Redis 7** (AOF persistence) | Fila de jobs, store de resultados, mensageria | Não |
| Runtime GPU | **NVIDIA CUDA 12.8** + PyTorch 2.7 | Aceleração GPU para todos os modelos ML | Sim |
| Containers | **Docker** + Docker Compose (GPU via nvidia-container-toolkit) | Orquestração de serviços | Host GPU |
| Imagens Base | `vllm/vllm-openai:latest`, `python:3.11-slim`, custom | Imagens Docker | - |
| Monitoramento | **Prometheus** (métricas) + **Grafana** (dashboards/alertas) | Observabilidade | Não |
| Error Tracking | **Sentry SDK** → GlitchTip | Rastreamento de exceções | Não |
| Model Hub | **HuggingFace Hub** (download de modelos, auth via `HF_TOKEN`) | Distribuição de modelos ML | Não |

### Requisitos de Hardware

| Configuração | VRAM | Notas |
|--------------|------|-------|
| **Com vLLM** (produção) | ~12-14 GB | vLLM ~8-10GB + inference-server ~3-4GB |
| **Sem vLLM** (dev/fallback) | ~10-12 GB | inference-server com Qwen local |
| **Mínimo absoluto** | 12 GB | GPU NVIDIA com CUDA 12+, compute capability sm_80+ (Ampere+) |
| **Produção recomendada** | 16+ GB | `gpu-memory-utilization=0.70-0.90` para melhor throughput |

Outros requisitos:
- **Docker Engine** com `nvidia-container-toolkit` para passthrough de GPU
- **Docker Compose v2+** (necessário para suporte a `profiles`)
- **RAM sistema**: ~12GB (vLLM ~8GB, inference-server ~2GB, workers ~200MB cada)
- **Armazenamento**: ~5-10GB para cache de modelos (HuggingFace + EasyOCR)

### Warmup API

Permite escalar workers **antes** de uma carga esperada:

```bash
# Ativar: 5 workers por 30 minutos
curl -X POST http://localhost:9000/warmup \
  -H "X-Warmup-Key: $WARMUP_KEY" \
  -H "Content-Type: application/json" \
  -d '{"workers": 5, "duration_minutes": 30}'
```

### Autenticação

Se `DOC_PIPELINE_API_KEY` estiver configurada, todos os endpoints (exceto `/health` e `/metrics`) requerem `X-API-Key` no header.

### ngrok (Acesso Externo)

```bash
./start-server.sh                              # URL aleatória
./start-server.sh start meu-dominio.ngrok.io   # Domínio customizado
./stop-server.sh
```

## Configuração

Copie `.env.example` para `.env`:

```env
# Hugging Face (OBRIGATÓRIO para download de modelos)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# Classificador
DOC_PIPELINE_CLASSIFIER_MODEL_PATH=models/classifier.pth
DOC_PIPELINE_CLASSIFIER_MODEL_TYPE=efficientnet_b0

# Extrator
DOC_PIPELINE_EXTRACTOR_BACKEND=hybrid        # hybrid, qwen-vl, ou easy-ocr

# vLLM (produção)
DOC_PIPELINE_VLLM_ENABLED=true
DOC_PIPELINE_VLLM_BASE_URL=http://vllm:8000/v1
DOC_PIPELINE_VLLM_MODEL=Qwen/Qwen2.5-VL-3B-Instruct
DOC_PIPELINE_VLLM_MAX_TOKENS=1024
DOC_PIPELINE_VLLM_TIMEOUT=60.0

# Orientação
DOC_PIPELINE_ORIENTATION_ENABLED=true

# API
DOC_PIPELINE_API_HOST=0.0.0.0
DOC_PIPELINE_API_PORT=9000
DOC_PIPELINE_API_KEY=sua-api-key             # opcional
DOC_PIPELINE_WARMUP_API_KEY=sua-warmup-key   # para /warmup

# Inference Server
DOC_PIPELINE_INFERENCE_SERVER_ENABLED=true
DOC_PIPELINE_INFERENCE_BATCH_SIZE=4
DOC_PIPELINE_INFERENCE_TIMEOUT=120

# Sentry / GlitchTip
DOC_PIPELINE_SENTRY_DSN=                     # se vazio, Sentry desabilitado
```

## Monitoramento

### Métricas Prometheus

```bash
curl http://localhost:9000/metrics
```

Métricas principais:
- `doc_pipeline_jobs_processed_total{operation, status, delivery_mode}` — Jobs processados
- `doc_pipeline_worker_processing_seconds{operation}` — Tempo de processamento
- `doc_pipeline_queue_depth` — Profundidade da fila
- `inference_batch_size` — Tamanho dos batches no inference server

### Grafana

Dashboards disponíveis:
- **Doc Pipeline Overview** — Métricas gerais, fila
- **Worker DocID** — Processing time, confidence, document types
- **Worker OCR** — Processing time, delivery, errors

```bash
./monitoring/scripts/create-dashboards.sh
./monitoring/scripts/create-alerts.sh
```

## Desenvolvimento

```bash
pip install -e ".[dev]"
pre-commit install

# Linting
ruff check .
ruff format .
pre-commit run --all-files
```

## Licença

MIT
