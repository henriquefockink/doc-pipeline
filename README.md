# doc-pipeline

Pipeline de processamento de documentos brasileiros (RG, CNH e CIN) com classificação automática e extração de dados estruturados.

## Funcionalidades

- **Classificação** de 12 tipos de documentos via EfficientNet (rg/cnh/cin x frente/verso/aberto/digital)
- **Extração de dados** estruturados via VLM (Qwen2.5-VL), OCR+regex (EasyOCR) ou modo híbrido
- **OCR genérico** para PDFs e imagens, com suporte a português
- **Correção automática de orientação** via docTR (MobileNetV3)
- **Validação de CPF** com detecção de swap RG↔CPF
- **Autoscaling** de workers baseado na profundidade da fila
- **Warmup API** para pré-escalar workers antes de carga esperada
- **Batched inference** via servidor centralizado para alto throughput

## Arquitetura

```
                              ┌──────────────────────────────────────────────┐
                              │           INFERENCE SERVER (GPU)             │
                              │         inference_server.py :9020            │
                              │  Qwen2.5-VL-7B — batched forward pass       │
                              └──────────────▲──────────────┬───────────────┘
                                             │ request      │ reply
                                             │ (Redis)      │ (Redis)
┌────────┐    ┌──────────────┐    ┌──────────┴──────────────▼──────────────┐
│ Client │───►│  API :9000   │───►│         WORKER DocID × N               │
│        │    │  (sem GPU)   │    │  EfficientNet (classify)               │
│        │    │              │    │  docTR (orientação)                    │
│        │    │  Redis queue  │    │  EasyOCR (OCR/hybrid)                 │
└────────┘    └──────┬───────┘    └────────────────────────────────────────┘
                     │
                     │            ┌────────────────────────────────────────┐
                     └───────────►│         WORKER OCR                     │
                                  │  EasyOCR + docTR :9011                 │
                                  │  PDF multi-página (150 DPI)            │
                                  └────────────────────────────────────────┘
```

**Fluxo principal (POST /process)**:
1. **API** recebe imagem, salva em disco temporário, enfileira job no Redis
2. **Worker** consome job, corrige orientação (docTR), classifica (EfficientNet)
3. **Worker** envia imagem + prompt ao **Inference Server** via Redis
4. **Inference Server** agrupa requests em batches, processa no VLM, devolve resultado
5. **Worker** valida dados (CPF), entrega resultado ao cliente (sync ou webhook)

> Para o fluxo detalhado de cada etapa, veja [docs/API_FLOW.md](docs/API_FLOW.md).

## Quick Start

### Docker (Recomendado)

```bash
# Configurar variáveis
cp .env.example .env
# Editar .env: adicionar HF_TOKEN (obrigatório)

# Subir stack completo (Redis + API + Worker + Inference Server)
docker compose up -d

# Ver logs
docker compose logs -f

# Parar
docker compose down
```

Serviços disponíveis:
- **API**: http://localhost:9000
- **Worker DocID**: Classificação e extração de RG/CNH/CIN
- **Worker OCR**: OCR genérico de PDFs/imagens
- **Inference Server**: VLM centralizado com batching
- **Autoscaler**: Escala workers automaticamente

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

# Multi-GPU
python cli.py documento.jpg --classifier-device cuda:0 --extractor-device cuda:1
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
      "nome": "JOÃO DA SILVA",
      "cpf": "123.456.789-00",
      "rg": "12.345.678-9",
      "data_nascimento": "15/03/1985",
      "nome_pai": "JOSÉ DA SILVA",
      "nome_mae": "MARIA DA SILVA",
      "naturalidade": "SÃO PAULO-SP",
      "data_expedicao": "10/05/2020",
      "orgao_expedidor": "SSP-SP"
    },
    "backend": "hybrid"
  },
  "image_correction": {
    "was_corrected": true,
    "rotation_applied": 90,
    "correction_method": "doctr_classification",
    "confidence": 0.98
  },
  "success": true,
  "error": null
}
```

## Backends de Extração

| Backend | Estratégia | Velocidade | Precisão |
|---------|-----------|:----------:|:--------:|
| **hybrid** (default) | VLM extrai dados + EasyOCR valida CPF com algoritmo | ~15s | Melhor (CPF) |
| **vlm** | Qwen2.5-VL extrai tudo via prompt→JSON | ~5s | Boa |
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

### Correção de Orientação (docTR)

Antes de classificar e extrair, o pipeline corrige automaticamente a orientação da imagem:

1. **EXIF metadata** — Corrige rotação salva pela câmera
2. **docTR classification** — MobileNetV3 classifica orientação em 4 ângulos (0°, 90°, 180°, 270°)
3. Aplica rotação se confiança acima do threshold (default: 0.5)

Isso resolve documentos fotografados de lado ou de cabeça para baixo.

## Infraestrutura

### Portas

| Serviço | Porta | Descrição |
|---------|-------|-----------|
| API | 9000 | REST API (FastAPI) |
| Workers DocID 1-5 | 9010, 9012, 9014, 9016, 9018 | Health + métricas |
| Workers DocID 6-8 | 9022, 9024, 9026 | Scale-profile workers |
| Worker OCR | 9011 | Health + métricas |
| Inference Server | 9020 | VLM centralizado |

### Inference Server (Batching)

O inference server centraliza o VLM (Qwen2.5-VL) e processa requests em lotes:

```
Workers enviam requests → Redis queue → Inference Server agrupa em batch
                                                    ↓
                                        Forward pass único no GPU
                                                    ↓
                                        Replies via Redis keys → Workers
```

| Config | Default | Descrição |
|--------|---------|-----------|
| `INFERENCE_BATCH_SIZE` | 4 | Max requests por batch |
| `INFERENCE_BATCH_TIMEOUT_MS` | 100 | Max espera para encher batch |
| `WORKER_CONCURRENT_JOBS` | 4 | Jobs paralelos por worker |

Benefícios: workers ficam leves (~800MB), GPU é compartilhada eficientemente, throughput aumenta ~66% vs. sequencial.

### Autoscaler

Monitora profundidade da fila e escala workers automaticamente:

```
Queue depth ≥ 5  →  Scale up (até MAX_WORKERS)
Queue depth = 0  →  Aguarda 120s → Scale down (até MIN_WORKERS)
```

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `MIN_WORKERS` | 1 | Mínimo de workers ativos |
| `MAX_WORKERS` | 3 | Máximo de workers (scaling normal) |
| `SCALE_UP_THRESHOLD` | 5 | Queue depth para escalar |
| `SCALE_DOWN_DELAY` | 120 | Segundos antes de desescalar |

Workers 2-8 são pré-criados e ficam parados (0 recursos). O autoscaler faz start/stop para escalar rapidamente.

### Warmup API

Permite escalar workers **antes** de uma carga esperada:

```bash
# Ativar: 5 workers por 30 minutos
curl -X POST http://localhost:9000/warmup \
  -H "X-Warmup-Key: $WARMUP_KEY" \
  -H "Content-Type: application/json" \
  -d '{"workers": 5, "duration_minutes": 30}'

# Status
curl http://localhost:9000/warmup/status -H "X-Warmup-Key: $WARMUP_KEY"

# Cancelar
curl -X DELETE http://localhost:9000/warmup -H "X-Warmup-Key: $WARMUP_KEY"
```

### Autenticação

Se `DOC_PIPELINE_API_KEY` estiver configurada, todos os endpoints (exceto `/health` e `/metrics`) requerem `X-API-Key` no header.

### ngrok (Acesso Externo)

```bash
./start-server.sh                              # URL aleatória
./start-server.sh start meu-dominio.ngrok.io   # Domínio customizado
./start-server.sh status
./stop-server.sh
```

## Configuração

Copie `.env.example` para `.env`:

```env
# Hugging Face (OBRIGATÓRIO)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# Classificador
DOC_PIPELINE_CLASSIFIER_MODEL_PATH=models/classifier.pth
DOC_PIPELINE_CLASSIFIER_MODEL_TYPE=efficientnet_b0
DOC_PIPELINE_CLASSIFIER_DEVICE=cuda:0

# Extrator
DOC_PIPELINE_EXTRACTOR_BACKEND=hybrid        # hybrid, qwen-vl, ou easy-ocr
DOC_PIPELINE_EXTRACTOR_DEVICE=cuda:0
DOC_PIPELINE_EXTRACTOR_MODEL_QWEN=Qwen/Qwen2.5-VL-7B-Instruct

# Orientação
DOC_PIPELINE_ORIENTATION_ENABLED=true
DOC_PIPELINE_ORIENTATION_CONFIDENCE_THRESHOLD=0.5

# API
DOC_PIPELINE_API_HOST=0.0.0.0
DOC_PIPELINE_API_PORT=9000
DOC_PIPELINE_API_KEY=sua-api-key             # opcional
DOC_PIPELINE_WARMUP_API_KEY=sua-warmup-key   # para /warmup

# Inference Server
DOC_PIPELINE_INFERENCE_SERVER_ENABLED=true
DOC_PIPELINE_INFERENCE_BATCH_SIZE=8
DOC_PIPELINE_INFERENCE_TIMEOUT=30
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
- `doc_pipeline_autoscaler_workers_current` — Workers ativos
- `inference_batch_size` — Tamanho dos batches no inference server

### Grafana

Dashboards disponíveis:
- **Doc Pipeline Overview** — Métricas gerais, autoscaler, fila
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

## Requisitos de Hardware

| Configuração | VRAM |
|--------------|------|
| Inference Server (Qwen2.5-VL-7B) | ~14GB |
| Worker DocID (EfficientNet + docTR + EasyOCR) | ~800MB |
| Worker OCR (EasyOCR + docTR) | ~2GB |
| Stack completo (1 worker + inference) | ~16GB |

## Licença

MIT
