# Fluxo de Execução da API

Este documento detalha o fluxo completo de processamento quando um documento é enviado via API.

## Visão Geral

```
┌──────────┐     ┌──────────┐     ┌─────────────┐     ┌────────────┐     ┌───────┐     ┌──────────┐
│  Imagem  │ ──► │ FastAPI  │ ──► │   Worker    │ ──► │  Inference │ ──► │ vLLM  │ ──► │ Resposta │
│(RG/CNH/  │     │  (API)   │     │  (DocID)    │     │   Server   │     │ (GPU) │     │   JSON   │
│  CIN)    │     │          │     │ (sem GPU)   │     │   (GPU)    │     │       │     │          │
└──────────┘     └──────────┘     └─────────────┘     └────────────┘     └───────┘     └──────────┘
                  Enfileira         Stateless           EfficientNet      Continuous     Entrega via
                  no Redis          queue consumer      + EasyOCR         Batching       sync/webhook
```

## Fluxo Detalhado - POST /process (Produção)

### 1. Requisição HTTP → API

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CLIENT                                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ curl -X POST http://localhost:9000/process \                                │
│   -H "X-API-Key: sua-api-key" \                                             │
│   -F "arquivo=@rg_aberto.jpg" \                                             │
│   -F "delivery_mode=sync"                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. API — Validação e Enfileiramento

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ API (api.py) — Port 9000 — SEM GPU                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Autenticação: Verifica X-API-Key (se configurada)                        │
│ 2. Rate limiting: SlowAPI com backend Redis                                 │
│ 3. Validação: Formato de imagem (JPEG, PNG, HEIC, HEIF, AVIF, TIFF, BMP)  │
│ 4. Salva imagem em disco temporário (/tmp/doc-pipeline/)                    │
│    Volume Docker compartilhado: temp-images                                 │
│ 5. Cria JobContext com request_id, operation, delivery_mode                │
│ 6. LPUSH job na fila Redis: queue:doc:documents                             │
│                                                                             │
│ Se sync: Faz polling na key Redis job:result:{request_id} (max 5min)        │
│ Se webhook: Retorna 202 imediatamente                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. Worker — Consome Job

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ WORKER DOCID (worker_docid.py) — Ports 9010-9018 — SEM GPU                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. BRPOP job de queue:doc:documents (blocking wait)                         │
│ 2. Deserializa JobContext                                                   │
│ 3. Monta request de inferência com:                                         │
│    - Caminho da imagem no volume compartilhado                              │
│    - Operação solicitada (classify, extract, process)                       │
│    - Backend de extração (hybrid, vlm, ocr)                                │
│ 4. LPUSH request para queue:doc:inference                                   │
│ 5. Polling em inference:result:{inference_id} (aguarda resultado)           │
│                                                                             │
│ O worker NÃO carrega modelos ML — todo processamento GPU é delegado        │
│ ao inference server via Redis.                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4. Inference Server — Preprocessamento e Classificação

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ INFERENCE SERVER (inference_server.py) — Port 9020 — GPU ~3-4GB             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ BATCHING + PIPELINE OVERLAP:                                                │
│ 1. BRPOP primeiro request de queue:doc:inference                            │
│ 2. Coleta mais requests (RPOP não-bloqueante) até:                          │
│    • INFERENCE_BATCH_SIZE (default 16) ou                                   │
│    • INFERENCE_BATCH_TIMEOUT_MS (default 100ms)                             │
│ 3. Enquanto batch N processa, coleta batch N+1 concorrentemente             │
│                                                                             │
│ Para cada request no batch (PROCESSADOS EM PARALELO via asyncio.gather):                                                 │
│                                                                             │
│ ORIENTAÇÃO (EasyOCR textbox detection):                                     │
│ ┌───────────────────────────────────────────────────────────────────────┐   │
│ │ 1. Verifica EXIF metadata (câmera salvou orientação?)                 │   │
│ │    • Se sim: aplica rotação do EXIF                                   │   │
│ │ 2. EasyOCR detecta textboxes na imagem                                │   │
│ │    • Analisa aspect ratio dos bounding boxes                          │   │
│ │    • Se maioria dos boxes são mais altos que largos → rotação 90°     │   │
│ │    • Threshold: proporção de boxes verticais vs horizontais           │   │
│ │ 3. Aplica rotação se detectada (90° ou 270°)                          │   │
│ │ Retorna: ImageCorrection {was_corrected, rotation_applied, method}    │   │
│ └───────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│ CLASSIFICAÇÃO (EfficientNet-B0, ~100ms):                                    │
│ ┌───────────────────────────────────────────────────────────────────────┐   │
│ │ EfficientNet-B0 classifica imagem em 12 classes                       │   │
│ │ • Softmax → {document_type: "rg_aberto", confidence: 0.97}           │   │
│ │ • Se confiança < min_confidence (0.5): pula extração                  │   │
│ └───────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5. Inference Server — Extração VLM (vLLM Embedded)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ INFERENCE SERVER — vLLM IN-PROCESS (extração de dados)                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ 1. Monta prompt Qwen2.5-VL chat template com image placeholder             │
│ 2. Passa imagem PIL DIRETAMENTE para vLLM (zero-copy, sem base64)          │
│ 3. Chama vLLM in-process:                                                   │
│                                                                             │
│    llm.generate(                                                            │
│      {                                                                      │
│        "prompt": "<|im_start|>user\n<|vision_start|>...",                   │
│        "multi_modal_data": {"image": pil_image}  ← PIL direto!             │
│      },                                                                     │
│      sampling_params=SamplingParams(temperature=0.0, max_tokens=1024)       │
│    )                                                                        │
│                                                                             │
│ ┌───────────────────────────────────────────────────────────────────────┐   │
│ │ vLLM EMBEDDED (mesmo processo, mesma GPU)                             │   │
│ │                                                                       │   │
│ │ • Qwen2.5-VL-3B-Instruct (3B params, vision-language model)           │   │
│ │ • Roda DENTRO do inference server (sem container separado)            │   │
│ │ • Imagem PIL vai direto para o GPU (sem base64, sem HTTP)             │   │
│ │ • Continuous batching: tokens intercalados entre requests              │   │
│ │ • PagedAttention: KV cache fragmentado em páginas na VRAM             │   │
│ │ • CUDA graphs + FlashAttention para kernel otimizado                  │   │
│ │ • gpu-memory-utilization: controla % da VRAM para KV cache            │   │
│ │   (0.40 = produção com MPS, 0.90 = GPU dedicada)                     │   │
│ │                                                                       │   │
│ │ vs vLLM HTTP (modo antigo):                                           │   │
│ │   HTTP: PIL → base64 (300KB) → HTTP POST → decode → GPU → HTTP resp  │   │
│ │   Embedded: PIL ─────────────────────────────► GPU → texto direto     │   │
│ │   Ganho: +64% throughput em alta concorrência (H200 benchmarks)       │   │
│ │                                                                       │   │
│ │ Resposta:                                                             │   │
│ │ { "nome": "JOAO DA SILVA", "cpf": "123.456.789-00", ... }            │   │
│ └───────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│ 4. Parseia JSON da resposta do VLM                                          │
│ 5. SET resultado em inference:result:{id} no Redis                          │
│                                                                             │
│ PIPELINE OVERLAP:                                                           │
│ • Enquanto VLM processa batch N, já coleta batch N+1 do Redis              │
│ • Preprocessing paralelo: asyncio.gather para I/O + orientação + classify  │
│ • Elimina tempo ocioso entre batches                                        │
│                                                                             │
│ FALLBACKS:                                                                  │
│ • VLLM_ENABLED=true: vLLM via HTTP (container separado, port 9030)         │
│ • Ambos false: Qwen2.5-VL local via HuggingFace transformers (dev)         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6. Validação de CPF (Backend Hybrid)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CPF VALIDATION (utils/cpf.py) — Exclusivo do backend hybrid                 │
│ Executado dentro do Inference Server                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ 1. Valida CPF extraído pelo VLM (algoritmo módulo 11)                       │
│    └─► CPF válido? → Continua                                               │
│    └─► CPF inválido? → Fallback:                                            │
│                                                                             │
│ 2. FALLBACK: EasyOCR extrai texto bruto da imagem                           │
│    • Busca padrões numéricos: ###.###.###-##                                │
│    • Testa cada candidato com algoritmo de CPF                              │
│    • Se encontra válido: substitui o CPF do VLM                             │
│                                                                             │
│ 3. SWAP DETECTION: Verifica se RG e CPF estão trocados                      │
│    • Se campo "rg" contém formato de CPF e vice-versa → troca               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7. Worker — Recebe Resultado

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ WORKER DOCID — recebe resultado via Redis                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ 1. Polling em inference:result:{inference_id} retorna resultado             │
│ 2. Deserializa resposta (classificação + extração + correção de imagem)     │
│ 3. Monta resultado final (DocumentResult)                                   │
│ 4. Entrega resultado (ver passo 8)                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8. Entrega do Resultado

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DELIVERY SERVICE (shared/delivery.py)                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ SYNC (default):                                                             │
│   Worker SET resultado em job:result:{request_id} no Redis                  │
│   API (que estava fazendo polling) lê o resultado e retorna ao cliente       │
│                                                                             │
│ WEBHOOK:                                                                    │
│   Worker POST resultado para a webhook_url do job                           │
│   • Retries com backoff exponencial                                         │
│   • Timeout configurável                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9. Resposta JSON

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

---

## Fluxo por Backend

### Backend Hybrid (Padrão em Produção)

```
Imagem ──► Inference Server:
             ├── EasyOCR textbox (orientação)
             ├── EfficientNet (classificação)
             └── vLLM (extração VLM)
                       │
                  JSON bruto
                       │
                       ▼
            CPF válido? (módulo 11)
                 │           │
                SIM         NÃO
                 │           │
                 │     EasyOCR fallback
                 │     (busca CPF válido)
                 │           │
                 ▼           ▼
            Swap check (RG↔CPF)
                       │
                       ▼
                  RGData/CNHData/CINData
```

### Backend VLM

```
Imagem ──► Inference Server:
             ├── EasyOCR textbox (orientação)
             ├── EfficientNet (classificação)
             └── vLLM (extração VLM)
                       │
                  JSON direto
                       │
                       ▼
                  RGData/CNHData/CINData
```

### Backend OCR

```
Imagem ──► Inference Server:
             ├── EasyOCR textbox (orientação)
             ├── EfficientNet (classificação)
             └── EasyOCR (texto bruto)
                       │
                 Regex parsing
                       │
                       ▼
                  RGData/CNHData/CINData
```

---

## Comparativo dos Backends

```
┌─────────────────────────┬──────────────────┬──────────────────┬──────────────────┐
│                         │     HYBRID       │      VLM         │     OCR          │
├─────────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Modelo VLM              │ Qwen + EasyOCR   │ Qwen2.5-VL-3B   │ N/A              │
│ VRAM (workers)          │ 0 (sem GPU)      │ 0 (sem GPU)      │ 0 (sem GPU)      │
│ VRAM (inf. server)      │ ~3-4GB           │ ~3-4GB           │ ~3-4GB           │
│ VRAM (vLLM)             │ ~8-10GB          │ ~8-10GB          │ N/A              │
│ Tempo extração          │ ~5s              │ ~2.5s            │ ~2s              │
├─────────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Estratégia              │ VLM + fallback   │ Prompt → JSON    │ OCR → Regex      │
│                         │ OCR para CPF     │ direto           │                  │
├─────────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ CPF accuracy            │ Melhor           │ VLM erra         │ Depende OCR      │
│                         │ (validação +     │   dígitos às     │                  │
│                         │  fallback)       │   vezes          │                  │
├─────────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Campos contextuais      │ VLM entende      │ Excelente        │ Limitado         │
│ (nome_pai vs nome_mae)  │ contexto         │                  │ (depende ordem)  │
├─────────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Documentos danificados  │ VLM infere       │ Infere campos    │ Só visível       │
└─────────────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

---

## Guia de Escolha do Backend

| Cenário | Recomendação |
|---------|--------------|
| Produção (precisão CPF é crítica) | **hybrid** (default) |
| Alta velocidade, precisão aceitável | **vlm** |
| Sem vLLM disponível, GPU limitada | **ocr** |

---

## Tempos de Execução (vLLM Embedded, H200, single request)

| Etapa | Tempo |
|-------|-------|
| Upload + decode imagem | ~50ms |
| Correção de orientação (EasyOCR textbox) | ~200ms |
| Classificação (EfficientNet) | ~100ms |
| Extração VLM (vLLM embedded, single request) | ~1.5s |
| Validação CPF + fallback (hybrid) | ~50ms |
| **Total (hybrid, single request)** | **~1.9s** |
| **Total (vlm only, single request)** | **~1.8s** |

Throughput em produção (H200, `gpu-memory-utilization=0.40`):
- Single (c=1): 0.52 rps, P50 = 1.8s
- Alta concorrência (c=20): **1.38 rps**, P50 = 14.2s
- Stress (c=30): **1.38 rps**, P50 = 17.8s

Comparação com backends anteriores (H200):

| Backend | P50 (c=1) | P50 (c=20) | RPS (c=20) |
|---------|-----------|------------|------------|
| HuggingFace (static batch) | 4.8s | 13.9s | 1.11 |
| vLLM HTTP (container separado) | 2.4s | 24.9s | 0.60 |
| **vLLM Embedded (in-process)** | **1.8s** | **14.2s** | **1.38** |

---

## Fluxo de Erro

### Baixa Confiança

```json
{
  "classification": {
    "document_type": "rg_frente",
    "confidence": 0.35
  },
  "extraction": null,
  "success": false,
  "error": "Confiança (35.0%) abaixo do mínimo (50.0%)"
}
```

### Timeout (Sync Mode)

Se o worker não processar dentro de 5 minutos, a API retorna timeout. O job pode ainda estar sendo processado — o cliente pode consultar `/jobs/{id}/status`.

---

## Endpoints Disponíveis

| Método | Endpoint | Classificador | Extrator | Inference Server | Auth |
|--------|----------|:-------------:|:--------:|:----------------:|:----:|
| POST | `/process` | Sim | Sim | Sim | Sim |
| POST | `/classify` | Sim | Não | Sim | Sim |
| POST | `/extract?doc_type=...` | Não | Sim | Sim | Sim |
| POST | `/ocr` | Não | Não | Sim | Sim |
| GET | `/jobs/{id}/status` | Não | Não | Não | Sim |
| GET | `/health` | Não | Não | Não | Não |
| GET | `/metrics` | Não | Não | Não | Não |

---

## Autenticação

Se `DOC_PIPELINE_API_KEY` estiver configurada, a API requer `X-API-Key` em todos os endpoints exceto `/health` e `/metrics`.

```
Request ──► API Key configurada? ──► NÃO ──► Acesso liberado
                  │
                 SIM
                  │
                  ▼
           Header X-API-Key? ──► NÃO ──► 401 Unauthorized
                  │
                 SIM
                  │
                  ▼
           Key válida? ──► NÃO ──► 403 Forbidden
                  │
                 SIM ──► Acesso liberado
```

---

## Infraestrutura e Comunicação entre Serviços

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Docker Network: doc-pipeline                      │
│                                                                             │
│  ┌─────────┐    ┌──────────────────┐    ┌─────────────────────┐            │
│  │  Redis   │◄───│      API         │    │   Workers (N+1)     │            │
│  │  :6379   │    │     :9000        │    │   DocID :9010-9018   │            │
│  │          │◄───│                  │    │   OCR   :9011        │            │
│  │ AOF      │    │  FastAPI         │    │   python:3.11-slim   │            │
│  │ persist  │◄───┤  SlowAPI         │    │   SEM GPU            │            │
│  └────┬─────┘    │  Uvicorn         │    └───────┬─────────────┘            │
│       │          └──────────────────┘            │                          │
│       │                                          │                          │
│       │  queue:doc:documents ◄───── API          │                          │
│       │  queue:doc:documents ─────► Workers       │                          │
│       │  queue:doc:inference ◄───── Workers       │                          │
│       │  queue:doc:inference ─────► Inference Svr  │                          │
│       │  inference:result:{id} ◄─── Inference Svr  │                          │
│       │  inference:result:{id} ───► Workers        │                          │
│       │  job:result:{id} ◄───────── Workers        │                          │
│       │  job:result:{id} ─────────► API            │                          │
│       │                                            │                          │
│       │    ┌─────────────────────────────┐         │                          │
│       └────┤  Inference Server :9020     │◄────────┘                          │
│            │  GPU ~3-4GB                 │    (via Redis)                     │
│            │  EfficientNet (~200MB)      │                                    │
│            │  EasyOCR (~2-4GB)           │    ┌────────────────────┐          │
│            │  VLLMEmbeddedClient         │                                    │
│            │  (in-process, zero-copy)    │                                    │
│            │  vLLM LLM class             │                                    │
│            │  Qwen2.5-VL-3B (~10-16GB)   │                                    │
│            │  Continuous Batching         │                                    │
│            │  PagedAttention              │                                    │
│            └─────────────────────────────┘                                    │
│                                                                             │
│            Volume: temp-images                                              │
│            (/tmp/doc-pipeline)                                              │
│            Compartilhado: API, Workers,                                      │
│            Inference Server                                                  │
│                                                                             │
│            Volume: model-cache                                              │
│            (/root/.cache/huggingface)                                       │
│            Compartilhado: Inference Server, vLLM                            │
└─────────────────────────────────────────────────────────────────────────────┘
```
