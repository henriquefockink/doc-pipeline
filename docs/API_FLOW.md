# Fluxo de Execução da API

Este documento detalha o fluxo completo de processamento quando um documento é enviado via API.

## Visão Geral

```
┌──────────┐     ┌──────────┐     ┌─────────────────┐     ┌──────────────┐
│  Imagem  │ ──► │ FastAPI  │ ──► │  Classificador  │ ──► │   Extrator   │
│  (RG/CNH)│     │          │     │  (EfficientNet) │     │ (VLM ou OCR) │
└──────────┘     └──────────┘     └─────────────────┘     └──────────────┘
                                          │                       │
                                          ▼                       ▼
                                   ┌─────────────┐         ┌─────────────┐
                                   │ Tipo + Conf │         │ Dados JSON  │
                                   └─────────────┘         └─────────────┘
```

## Fluxo Detalhado - POST /process

### 1. Requisição HTTP

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ REQUISIÇÃO HTTP                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ curl -X POST http://localhost:8001/process -F "arquivo=@rg_aberto.jpg"      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. FastAPI - Recepção

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ FastAPI (api.py:147)                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ async def process(arquivo: UploadFile, extract: bool = True, ...)           │
│   • Lê bytes do arquivo: contents = await arquivo.read()                    │
│   • Converte para PIL: image = Image.open(io.BytesIO(contents)).convert("RGB")
│   • Chama pipeline.process(image, extract=True, min_confidence=0.5)         │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. Pipeline - Orquestração

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DocumentPipeline.process() (pipeline.py:145)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│ def process(image, extract=True, min_confidence=0.5):                       │
│   • classification = self.classify(image)  ──────────────────────┐          │
│   • if classification.confidence >= min_confidence:              │          │
│       extraction = self.extract(image, classification.document_type)        │
│   • return PipelineResult(classification, extraction)                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4. Classificação + Extração (Paralelo)

```
                    ┌─────────────────────────────────────┐
                    ▼                                     ▼
┌───────────────────────────────────┐   ┌───────────────────────────────────┐
│ 4a. CLASSIFICAÇÃO                 │   │ 4b. EXTRAÇÃO                      │
│     (classifier/adapter.py)       │   │     (pipeline.py:91)              │
├───────────────────────────────────┤   ├───────────────────────────────────┤
│ ClassifierAdapter.classify()      │   │ @property extractor:              │
│                                   │   │ if backend == QWEN_VL:            │
│ • EfficientNet-B0 processa imagem │   │     return QwenVLExtractor()      │
│ • Softmax sobre 8 classes         │   │ elif backend == GOT_OCR:          │
│ • Retorna:                        │   │     return GOTOCRExtractor()      │
│   {                               │   │                                   │
│     document_type: "rg_aberto",   │   │ extractor.extract(image, doc_type)│
│     confidence: 0.97,             │   │         │                         │
│     all_probabilities: {...}      │   │         ▼                         │
│   }                               │   │   document_type.is_rg → extract_rg│
└───────────────────────────────────┘   └───────────────────────────────────┘
```

### 5. Backends de Extração

```
                          ┌───────────────────────────────────────────────────┐
                          ▼                                                   ▼
┌──────────────────────────────────────────────┐   ┌──────────────────────────────────────────────┐
│ 5a. QWEN-VL (extractors/qwen_vl.py)          │   │ 5b. GOT-OCR (extractors/got_ocr.py)          │
│     ~16GB VRAM                               │   │     ~2GB VRAM                                │
├──────────────────────────────────────────────┤   ├──────────────────────────────────────────────┤
│ extract_rg():                                │   │ extract_rg():                                │
│                                              │   │                                              │
│ 1. Carrega Qwen2.5-VL-7B-Instruct (lazy)     │   │ 1. Carrega GOT-OCR-2.0-hf (lazy)             │
│                                              │   │                                              │
│ 2. Monta mensagem com imagem + prompt:       │   │ 2. Extrai texto com OCR:                     │
│    ┌────────────────────────────────────┐    │   │    ┌────────────────────────────────────┐    │
│    │ "Extraia os campos deste RG:       │    │   │    │ model.chat(tokenizer, image,       │    │
│    │  nome, cpf, rg, data_nascimento... │    │   │    │            ocr_type="format")      │    │
│    │  Retorne JSON, use null se ausente"│    │   │    └────────────────────────────────────┘    │
│    └────────────────────────────────────┘    │   │                                              │
│                                              │   │ 3. Retorno OCR (texto puro):                 │
│ 3. VLM gera resposta estruturada:            │   │    ┌────────────────────────────────────┐    │
│    ┌────────────────────────────────────┐    │   │    │ REPÚBLICA FEDERATIVA DO BRASIL     │    │
│    │ {                                  │    │   │    │ REGISTRO GERAL                     │    │
│    │   "nome": "JOÃO DA SILVA",         │    │   │    │ 12.345.678-9                       │    │
│    │   "cpf": "123.456.789-00",         │    │   │    │ NOME                               │    │
│    │   "rg": "12.345.678-9",            │    │   │    │ JOÃO DA SILVA                      │    │
│    │   "data_nascimento": "15/03/1985", │    │   │    │ FILIAÇÃO                           │    │
│    │   ...                              │    │   │    │ JOSÉ DA SILVA                      │    │
│    │ }                                  │    │   │    │ MARIA DA SILVA                     │    │
│    └────────────────────────────────────┘    │   │    │ CPF 123.456.789-00                 │    │
│                                              │   │    │ ...                                │    │
│ 4. Parseia JSON → RGData                     │   │    └────────────────────────────────────┘    │
│                                              │   │                                              │
│                                              │   │ 4. Parseia com REGEX (_parse_rg_from_text):  │
│                                              │   │    • CPF: r"(\d{3}\.\d{3}\.\d{3}-\d{2})"     │
│                                              │   │    • Data: r"(\d{2}/\d{2}/\d{4})"            │
│                                              │   │    • Órgão: r"SSP[\-/]?[A-Z]{2}"             │
│                                              │   │    • Nome: linha após "NOME"                │
│                                              │   │                                              │
│                                              │   │ 5. Monta RGData com campos encontrados       │
└──────────────────────────────────────────────┘   └──────────────────────────────────────────────┘
```

### 6. Resposta JSON

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│ RESPOSTA JSON                                                                                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────┤
│ {                                                                                               │
│   "file_path": null,                                                                            │
│   "classification": {                                                                           │
│     "document_type": "rg_aberto",                                                               │
│     "confidence": 0.97,                                                                         │
│     "all_probabilities": { "rg_aberto": 0.97, "rg_frente": 0.01, ... }                          │
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
│     "backend": "qwen-vl"  ◄─── ou "got-ocr"                                                     │
│   },                                                                                            │
│   "success": true,                                                                              │
│   "error": null                                                                                 │
│ }                                                                                               │
└─────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Comparativo dos Backends

```
┌─────────────────────────┬────────────────────────────┬────────────────────────────┐
│                         │         QWEN-VL            │         GOT-OCR            │
├─────────────────────────┼────────────────────────────┼────────────────────────────┤
│ Modelo                  │ Qwen2.5-VL-7B-Instruct     │ GOT-OCR-2.0-hf             │
│ VRAM                    │ ~16GB                      │ ~2GB                       │
│ Tempo extração          │ ~3-5s                      │ ~1-2s                      │
├─────────────────────────┼────────────────────────────┼────────────────────────────┤
│ Estratégia              │ Prompt → JSON direto       │ OCR → Regex parsing        │
├─────────────────────────┼────────────────────────────┼────────────────────────────┤
│ Campos contextuais      │ ✅ Excelente               │ ⚠️ Limitado                │
│ (nome_pai vs nome_mae)  │ Entende contexto           │ Depende da ordem no texto  │
├─────────────────────────┼────────────────────────────┼────────────────────────────┤
│ Documentos danificados  │ ✅ Infere campos           │ ❌ Só extrai o visível     │
├─────────────────────────┼────────────────────────────┼────────────────────────────┤
│ Custo GPU               │ Alto                       │ Baixo                      │
└─────────────────────────┴────────────────────────────┴────────────────────────────┘
```

---

## Guia de Escolha do Backend

| Cenário | Recomendação |
|---------|--------------|
| Produção com GPU potente (24GB+) | **qwen-vl** |
| GPU limitada (8GB) | **got-ocr** |
| Alta precisão em campos contextuais | **qwen-vl** |
| Alto volume, menor precisão aceitável | **got-ocr** |
| Multi-GPU (classificador + extrator separados) | **qwen-vl** em GPU dedicada |

---

## Tempos de Execução Típicos

| Etapa | Tempo |
|-------|-------|
| Upload + decode imagem | ~50ms |
| Classificação (EfficientNet) | ~100ms |
| Extração (Qwen-VL) | ~3-5s |
| Extração (GOT-OCR) | ~1-2s |
| **Total com Qwen-VL** | **~3-6s** |
| **Total com GOT-OCR** | **~1-3s** |

---

## Fluxo de Erro

Quando a confiança está abaixo do mínimo:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ RESPOSTA - BAIXA CONFIANÇA                                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│ {                                                                           │
│   "file_path": null,                                                        │
│   "classification": {                                                       │
│     "document_type": "rg_frente",                                           │
│     "confidence": 0.35,                                                     │
│     "all_probabilities": { ... }                                            │
│   },                                                                        │
│   "extraction": null,  ◄─── Extração não executada                          │
│   "success": false,                                                         │
│   "error": "Confiança (35.0%) abaixo do mínimo (50.0%)"                     │
│ }                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Endpoints Disponíveis

| Método | Endpoint | Descrição | Usa Classificador | Usa Extrator |
|--------|----------|-----------|:-----------------:|:------------:|
| POST | `/process` | Pipeline completo | ✅ | ✅ |
| POST | `/classify` | Apenas classificação | ✅ | ❌ |
| POST | `/extract?doc_type=...` | Apenas extração | ❌ | ✅ |
| GET | `/health` | Status da API | ❌ | ❌ |
| GET | `/classes` | Lista classes | ❌ | ❌ |
