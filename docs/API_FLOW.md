# Fluxo de Execução da API

Este documento detalha o fluxo completo de processamento quando um documento é enviado via API.

## Visão Geral

```
┌──────────┐     ┌──────────┐     ┌─────────────┐     ┌────────────┐     ┌──────────────┐
│  Imagem  │ ──► │ FastAPI  │ ──► │   Worker    │ ──► │  Inference │ ──► │   Resposta   │
│(RG/CNH/  │     │  (API)   │     │  (DocID)    │     │   Server   │     │    JSON      │
│  CIN)    │     │          │     │             │     │  (VLM GPU) │     │              │
└──────────┘     └──────────┘     └─────────────┘     └────────────┘     └──────────────┘
                  Enfileira         Classifica +        Batched           Entrega via
                  no Redis          Preprocessa         Inference         sync/webhook
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
│ API (api.py) — Port 9000                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. Autenticação: Verifica X-API-Key (se configurada)                        │
│ 2. Rate limiting: SlowAPI com backend Redis                                 │
│ 3. Validação: Formato de imagem (JPEG, PNG, HEIC, HEIF, AVIF, TIFF, BMP)  │
│ 4. Salva imagem em disco temporário (/tmp/doc-pipeline/)                    │
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
│ WORKER DOCID (worker_docid.py) — Ports 9010-9026                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1. BRPOP job de queue:doc:documents (blocking wait)                         │
│ 2. Carrega imagem do disco                                                  │
│ 3. Converte para RGB (suporte HEIC/HEIF via pillow-heif)                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4. Preprocessamento — Correção de Orientação

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ ORIENTATION CORRECTOR (preprocessing/orientation.py)                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Imagem ──► EXIF check ──► docTR MobileNetV3 ──► Rotação corrigida         │
│                                                                             │
│ Passo 1: Verifica EXIF metadata (câmera salvou orientação?)                 │
│   • Se sim: aplica rotação do EXIF                                          │
│                                                                             │
│ Passo 2: Classificação docTR (modelo ~6MB, MobileNetV3)                     │
│   • Input: imagem 512×512                                                   │
│   • Output: ângulo (0°, 90°, 180°, 270°) + confiança                       │
│   • Se confiança ≥ threshold (0.5): aplica rotação                          │
│                                                                             │
│ Retorna: ImageCorrection {was_corrected, rotation_applied, method, conf}    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5. Classificação — EfficientNet

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CLASSIFICADOR (classifier/adapter.py)                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ EfficientNet-B0 processa imagem (local no worker, ~100ms)                   │
│                                                                             │
│ • Softmax sobre 12 classes                                                  │
│ • Retorna:                                                                  │
│   {                                                                         │
│     document_type: "rg_aberto",                                             │
│     confidence: 0.97                                                        │
│   }                                                                         │
│                                                                             │
│ Se confiança < min_confidence (0.5):                                        │
│   → Pula extração, retorna erro "baixa confiança"                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6. Extração — Inference Server (Backend Hybrid)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ WORKER envia request para INFERENCE SERVER                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│ Worker (lightweight, ~800MB):                                                │
│ 1. Monta prompt baseado no doc_type (RG/CNH/CIN)                            │
│ 2. Serializa imagem + prompt                                                 │
│ 3. LPUSH para queue:doc:inference                                            │
│ 4. Polling em inference:result:{inference_id} (aguarda resultado)            │
│                                                                              │
│ ┌─────────────────────────────────────────────────────────────────────────┐  │
│ │ INFERENCE SERVER (inference_server.py) — Port 9020 — GPU ~14GB         │  │
│ ├─────────────────────────────────────────────────────────────────────────┤  │
│ │                                                                         │  │
│ │ 1. BRPOP primeiro request de queue:doc:inference                        │  │
│ │ 2. Coleta mais requests (RPOP não-bloqueante) até:                      │  │
│ │    • INFERENCE_BATCH_SIZE (default 8) ou                                │  │
│ │    • INFERENCE_BATCH_TIMEOUT_MS (default 100ms)                         │  │
│ │ 3. Processa batch em forward pass único (Qwen2.5-VL-7B)                │  │
│ │ 4. SET resultado em inference:result:{id} para cada request             │  │
│ │                                                                         │  │
│ │ Prompt exemplo (RG):                                                    │  │
│ │ ┌───────────────────────────────────────────────────────────────────┐   │  │
│ │ │ "Extraia os campos deste RG brasileiro:                          │   │  │
│ │ │  nome, cpf, rg, data_nascimento, nome_pai, nome_mae,            │   │  │
│ │ │  naturalidade, data_expedicao, orgao_expedidor                   │   │  │
│ │ │  Retorne JSON. Use null se ausente."                             │   │  │
│ │ └───────────────────────────────────────────────────────────────────┘   │  │
│ │                                                                         │  │
│ │ Resposta VLM:                                                           │  │
│ │ {                                                                       │  │
│ │   "nome": "JOÃO DA SILVA",                                              │  │
│ │   "cpf": "123.456.789-00",                                              │  │
│ │   "rg": "12.345.678-9",                                                 │  │
│ │   "data_nascimento": "15/03/1985",                                      │  │
│ │   ...                                                                   │  │
│ │ }                                                                       │  │
│ └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 7. Validação de CPF (Backend Hybrid)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CPF VALIDATION (utils/cpf.py) — Exclusivo do backend hybrid                 │
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

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ RESPOSTA JSON                                                                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ {                                                                                               │
│   "file_path": null,                                                                            │
│   "classification": {                                                                           │
│     "document_type": "rg_aberto",                                                               │
│     "confidence": 0.97                                                                          │
│   },                                                                                            │
│   "extraction": {                                                                               │
│     "document_type": "rg_aberto",                                                               │
│     "data": {                                                                                   │
│       "nome": "JOÃO DA SILVA",                                                                  │
│       "cpf": "123.456.789-00",                                                                  │
│       "rg": "12.345.678-9",                                                                     │
│       "data_nascimento": "15/03/1985",                                                          │
│       "nome_pai": "JOSÉ DA SILVA",                                                              │
│       "nome_mae": "MARIA DA SILVA",                                                             │
│       "naturalidade": "SÃO PAULO-SP",                                                           │
│       "data_expedicao": "10/05/2020",                                                           │
│       "orgao_expedidor": "SSP-SP"                                                               │
│     },                                                                                          │
│     "raw_text": null,                                                                           │
│     "backend": "hybrid"                                                                         │
│   },                                                                                            │
│   "image_correction": {                                                                         │
│     "was_corrected": true,                                                                      │
│     "rotation_applied": 90,                                                                     │
│     "correction_method": "doctr_classification",                                                │
│     "confidence": 0.98                                                                          │
│   },                                                                                            │
│   "success": true,                                                                              │
│   "error": null                                                                                 │
│ }                                                                                               │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Fluxo por Backend

### Backend Hybrid (Padrão em Produção)

```
Imagem ──► docTR (orientação) ──► EfficientNet (classificação)
                                         │
                                         ▼
                               Inference Server (VLM)
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
Imagem ──► docTR (orientação) ──► EfficientNet (classificação)
                                         │
                                         ▼
                               Inference Server (VLM)
                                         │
                                    JSON direto
                                         │
                                         ▼
                                    RGData/CNHData/CINData
```

### Backend OCR

```
Imagem ──► docTR (orientação) ──► EfficientNet (classificação)
                                         │
                                         ▼
                                   EasyOCR (texto)
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
│ Modelo                  │ Qwen + EasyOCR   │ Qwen2.5-VL-7B   │ EasyOCR          │
│ VRAM (worker)           │ ~800MB           │ ~800MB           │ ~2GB             │
│ VRAM (inference server) │ ~14GB            │ ~14GB            │ N/A              │
│ Tempo extração          │ ~15s             │ ~5s              │ ~2s              │
├─────────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Estratégia              │ VLM + fallback   │ Prompt → JSON    │ OCR → Regex      │
│                         │ OCR para CPF     │ direto           │                  │
├─────────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ CPF accuracy            │ ✅ Melhor        │ ⚠️ VLM erra     │ ⚠️ Depende OCR   │
│                         │ (validação +     │   dígitos às     │                  │
│                         │  fallback)       │   vezes          │                  │
├─────────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Campos contextuais      │ ✅ VLM entende   │ ✅ Excelente     │ ⚠️ Limitado      │
│ (nome_pai vs nome_mae)  │ contexto         │                  │ (depende ordem)  │
├─────────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Documentos danificados  │ ✅ VLM infere    │ ✅ Infere campos │ ❌ Só visível    │
└─────────────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

---

## Guia de Escolha do Backend

| Cenário | Recomendação |
|---------|--------------|
| Produção (precisão CPF é crítica) | **hybrid** (default) |
| Alta velocidade, precisão aceitável | **vlm** |
| GPU limitada (<8GB), sem inference server | **ocr** |
| Multi-GPU (classificador + extrator separados) | **hybrid** ou **vlm** |

---

## Tempos de Execução Típicos

| Etapa | Tempo |
|-------|-------|
| Upload + decode imagem | ~50ms |
| Correção de orientação (docTR) | ~50ms |
| Classificação (EfficientNet) | ~100ms |
| Extração (Hybrid: VLM + CPF validation) | ~15s |
| Extração (VLM only) | ~5s |
| Extração (EasyOCR) | ~2s |
| **Total com Hybrid** | **~15-16s** |
| **Total com VLM** | **~5-6s** |
| **Total com EasyOCR** | **~2-3s** |

Com batching (batch_size=8), throughput: ~3-4 docs/s vs ~0.2 docs/s sequencial.

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
| POST | `/process` | ✅ | ✅ | ✅ | ✅ |
| POST | `/classify` | ✅ | ❌ | ❌ | ✅ |
| POST | `/extract?doc_type=...` | ❌ | ✅ | ✅ | ✅ |
| POST | `/ocr` | ❌ | ❌ | ❌ | ✅ |
| GET | `/jobs/{id}/status` | ❌ | ❌ | ❌ | ✅ |
| GET | `/health` | ❌ | ❌ | ❌ | ❌ |
| GET | `/metrics` | ❌ | ❌ | ❌ | ❌ |

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
