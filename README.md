# doc-pipeline

Pipeline de classificação e extração de dados de documentos brasileiros (RG/CNH).

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              doc-pipeline                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐                                                               │
│  │  Imagem  │──────────────────────────────────────────────────────────┐    │
│  └──────────┘                                                          │    │
│                                                                        ▼    │
│  ┌──────────┐     ┌─────────────┐      ┌────────────────────────────────┐   │
│  │   PDF    │────▶│  PyMuPDF    │─────▶│         GOT-OCR2               │   │
│  └──────────┘     │ (converter) │      │   (OCR puro, ~2GB VRAM)        │   │
│                   └─────────────┘      └────────────────────────────────┘   │
│                         │                            │                       │
│                         │                            ▼                       │
│                         │              ┌────────────────────────────────┐   │
│                         │              │   GenericExtractionResult      │   │
│                         │              │   { raw_text, pages[] }        │   │
│                         │              └────────────────────────────────┘   │
│                         │                                                    │
│  ════════════════════════════════════════════════════════════════════════   │
│                                                                              │
│  ┌──────────┐     ┌─────────────────┐      ┌────────────────────────────┐   │
│  │  Imagem  │────▶│  Classificador  │─────▶│  Tipo: RG/CNH              │   │
│  │ (RG/CNH) │     │  EfficientNet   │      │  + Confiança               │   │
│  └──────────┘     └─────────────────┘      └────────────────────────────┘   │
│                                                       │                      │
│                                                       ▼                      │
│                                            ┌────────────────────────────┐   │
│                                            │  Qwen2.5-VL ou GOT-OCR2    │   │
│                                            │  (extração estruturada)    │   │
│                                            └────────────────────────────┘   │
│                                                       │                      │
│                                                       ▼                      │
│                                            ┌────────────────────────────┐   │
│                                            │  ExtractionResult          │   │
│                                            │  { RGData | CNHData }      │   │
│                                            └────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Fluxos

| Entrada | Tipo | Backend | Saída |
|---------|------|---------|-------|
| Imagem (RG/CNH) | `rg_*`, `cnh_*` | Qwen-VL (default) | Dados estruturados (nome, cpf, etc) |
| Imagem (qualquer) | `generic` | Qwen-VL | Texto bruto |
| **PDF** | `generic` | **GOT-OCR** (automático) | Texto bruto por página |

> **Nota:** PDFs são automaticamente convertidos para imagem via PyMuPDF (~50-100ms/página) e processados pelo GOT-OCR, que é mais leve (~2GB VRAM).

> Para uma visão detalhada do fluxo de execução da API, consulte [docs/API_FLOW.md](docs/API_FLOW.md).

## Instalação

```bash
# Clone o repositório
git clone <repo-url>
cd doc-pipeline

# Crie e ative o ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# .\venv\Scripts\activate  # Windows

# Instale as dependências
pip install -r requirements.txt

# Copie o modelo do classificador para a pasta models/
cp /caminho/para/modelo.pth models/classifier.pth

# Configure as variáveis de ambiente
cp .env.example .env
# Edite o .env e adicione seu HF_TOKEN (obrigatório)
```

### Configuração do Hugging Face Token

O token do Hugging Face é **obrigatório** para download dos modelos VLM. Sem ele:
- Downloads serão mais lentos e com rate limits
- Modelos gated (como PyAnnote) não funcionarão

1. Crie um token em: https://huggingface.co/settings/tokens
2. Adicione ao arquivo `.env`:
   ```env
   HF_TOKEN=hf_seu_token_aqui
   ```

## Quick Start

```bash
# Ativar venv
source venv/bin/activate

# Subir a API (usa models/classifier.pth por padrão)
python api.py

# A API estará disponível em http://localhost:8001
```

### Gerenciar servidor

```bash
# Iniciar API (background)
./start-server.sh

# Iniciar em foreground (logs no terminal, Ctrl+C para parar)
./start-server.sh start -f

# Iniciar com ngrok (túnel externo)
./start-server.sh start --ngrok                # URL aleatória
./start-server.sh start meu-dominio.ngrok.io   # Domínio fixo (ngrok pago)

# Ver status
./start-server.sh status

# Parar tudo
./start-server.sh stop
# ou
./stop-server.sh
```

**Opções do script:**

| Flag | Descrição |
|------|-----------|
| `-f`, `--foreground` | Roda em foreground (logs no terminal) |
| `--ngrok` | Habilita túnel ngrok (URL aleatória) |
| `--ngrok=DOMINIO` | Habilita ngrok com domínio fixo |

## Uso

### CLI

```bash
# Pipeline completo (classificação + extração)
python cli.py documento.jpg

# Usar GOT-OCR2 ao invés de Qwen (menor uso de VRAM)
python cli.py documento.jpg --backend got-ocr

# Apenas classificar
python cli.py documento.jpg --no-extraction

# Processar pasta com saída JSON
python cli.py ./documentos/ --json -o resultados.json

# Multi-GPU: classificador em cuda:0, extractor em cuda:1
python cli.py documento.jpg --classifier-device cuda:0 --extractor-device cuda:1
```

### API

```bash
# Iniciar servidor (usa models/classifier.pth por padrão)
python api.py

# Com GOT-OCR (menor VRAM)
DOC_PIPELINE_EXTRACTOR_BACKEND=got-ocr python api.py

# Com modelo em outro caminho
DOC_PIPELINE_CLASSIFIER_MODEL_PATH=/outro/caminho/modelo.pth python api.py
```

#### Endpoints

| Método | Endpoint | Descrição | Auth |
|--------|----------|-----------|:----:|
| POST | `/process` | Pipeline completo (classificação + extração) | Sim |
| POST | `/classify` | Apenas classificação | Sim |
| POST | `/extract?doc_type=rg_frente` | Extração estruturada (tipo conhecido) | Sim |
| POST | `/extract?doc_type=generic` | **OCR puro** (texto bruto) - suporta PDF | Sim |
| GET | `/health` | Status da API | Não |
| GET | `/classes` | Lista classes suportadas | Sim |

#### Autenticação

Se `DOC_PIPELINE_API_KEY` estiver configurada no `.env`, todos os endpoints (exceto `/health`) requerem o header `X-API-Key`:

```bash
# Com autenticação
curl -X POST http://localhost:8001/process \
  -H "X-API-Key: sua-api-key-aqui" \
  -F "arquivo=@documento.jpg"
```

#### Exemplos

```bash
# Pipeline completo
curl -X POST http://localhost:8001/process \
  -H "X-API-Key: $DOC_PIPELINE_API_KEY" \
  -F "arquivo=@documento.jpg"

# Apenas classificar
curl -X POST http://localhost:8001/classify \
  -H "X-API-Key: $DOC_PIPELINE_API_KEY" \
  -F "arquivo=@documento.jpg"

# Extrair dados (tipo conhecido)
curl -X POST "http://localhost:8001/extract?doc_type=rg_frente" \
  -H "X-API-Key: $DOC_PIPELINE_API_KEY" \
  -F "arquivo=@rg.jpg"

# OCR genérico (texto bruto) - imagem
curl -X POST "http://localhost:8001/extract?doc_type=generic" \
  -H "X-API-Key: $DOC_PIPELINE_API_KEY" \
  -F "arquivo=@documento.jpg"

# OCR genérico (texto bruto) - PDF multi-página
curl -X POST "http://localhost:8001/extract?doc_type=generic" \
  -H "X-API-Key: $DOC_PIPELINE_API_KEY" \
  -F "arquivo=@documento.pdf"

# Health check (não requer auth)
curl http://localhost:8001/health
```

### Python

```python
from doc_pipeline import DocumentPipeline

# Inicializa pipeline
pipeline = DocumentPipeline(
    classifier_model_path="models/classifier.pth",
    extractor_backend="qwen-vl",  # ou "got-ocr"
)

# Pipeline completo
result = pipeline.process("documento.jpg")
print(f"Tipo: {result.document_type.value}")
print(f"Confiança: {result.classification.confidence:.1%}")
if result.data:
    print(f"Nome: {result.data.nome}")
    print(f"CPF: {result.data.cpf}")

# Apenas classificar
classification = pipeline.classify("documento.jpg")

# Apenas extrair (tipo conhecido)
from doc_pipeline.schemas import DocumentType
extraction = pipeline.extract("rg.jpg", DocumentType.RG_FRENTE)
```

## Backends VLM

| Backend | Modelo | VRAM | Uso |
|---------|--------|------|-----|
| **qwen-vl** (default) | Qwen/Qwen2.5-VL-7B-Instruct | ~16GB | Extração contextualizada com prompts |
| **got-ocr** | stepfun-ai/GOT-OCR-2.0-hf | ~2GB | OCR puro, suporta markdown |

### OCR Genérico (doc_type=generic)

Para extração de texto bruto sem estruturação, use `doc_type=generic`. Suporta imagens e **PDFs**.

**Resposta para imagem:**
```json
{
  "document_type": "generic",
  "raw_text": "Texto extraído do documento...",
  "pages": [{"page": 1, "text": "Texto..."}],
  "total_pages": 1,
  "backend": "qwen-vl"
}
```

**Resposta para PDF (multi-página):**
```json
{
  "document_type": "generic",
  "raw_text": "Texto de todas as páginas concatenado...",
  "pages": [
    {"page": 1, "text": "Texto da página 1..."},
    {"page": 2, "text": "Texto da página 2..."}
  ],
  "total_pages": 2,
  "backend": "got-ocr"
}
```

> **Nota:** PDFs são automaticamente processados pelo GOT-OCR (mais leve). A conversão PDF→imagem é feita via PyMuPDF (~50-100ms/página).

## Campos Extraídos

### RG

| Campo | Descrição |
|-------|-----------|
| nome | Nome completo |
| nome_pai | Nome do pai |
| nome_mae | Nome da mãe |
| data_nascimento | Data de nascimento (DD/MM/AAAA) |
| naturalidade | Cidade/Estado de nascimento |
| cpf | CPF (###.###.###-##) |
| rg | Número do RG |
| data_expedicao | Data de expedição |
| orgao_expedidor | Órgão expedidor (ex: SSP-SP) |

### CNH

| Campo | Descrição |
|-------|-----------|
| nome | Nome completo |
| cpf | CPF (###.###.###-##) |
| data_nascimento | Data de nascimento (DD/MM/AAAA) |
| numero_registro | Número de registro da CNH |
| numero_espelho | Número do espelho |
| validade | Data de validade |
| categoria | Categoria (A, B, AB, C, D, E) |
| observacoes | Observações/restrições |
| primeira_habilitacao | Data da primeira habilitação |

## Configuração

### Environment Variables

Copie `.env.example` para `.env` e configure:

```bash
cp .env.example .env
```

```env
# Hugging Face (OBRIGATÓRIO)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# Classificador
DOC_PIPELINE_CLASSIFIER_MODEL_PATH=models/classifier.pth
DOC_PIPELINE_CLASSIFIER_MODEL_TYPE=efficientnet_b0
DOC_PIPELINE_CLASSIFIER_DEVICE=cuda:0
DOC_PIPELINE_CLASSIFIER_FP8=false

# Extractor
DOC_PIPELINE_EXTRACTOR_BACKEND=qwen-vl  # ou got-ocr
DOC_PIPELINE_EXTRACTOR_DEVICE=cuda:0
DOC_PIPELINE_EXTRACTOR_MODEL_QWEN=Qwen/Qwen2.5-VL-7B-Instruct
DOC_PIPELINE_EXTRACTOR_MODEL_GOT=stepfun-ai/GOT-OCR-2.0-hf

# API
DOC_PIPELINE_API_HOST=0.0.0.0
DOC_PIPELINE_API_PORT=8001
DOC_PIPELINE_API_KEY=sua-api-key-secreta  # Autenticação (opcional)

# Geral
DOC_PIPELINE_MIN_CONFIDENCE=0.5
DOC_PIPELINE_WARMUP_ON_START=true  # Carrega modelos na inicialização
```

## Requisitos de Hardware

| Configuração | GPU | VRAM |
|--------------|-----|------|
| Classifier + GOT-OCR | 1x | 8GB+ |
| Classifier + Qwen-VL | 1x | 24GB+ |
| Multi-GPU | 2x | 8GB + 16GB |

## Classes de Documentos

- `cnh_aberta` - CNH aberta (frente e verso visíveis)
- `cnh_digital` - CNH digital
- `cnh_frente` - CNH frente
- `cnh_verso` - CNH verso
- `rg_aberto` - RG aberto (frente e verso visíveis)
- `rg_digital` - RG digital
- `rg_frente` - RG frente
- `rg_verso` - RG verso
- `generic` - Documento genérico (OCR puro, sem extração estruturada)

## Estrutura do Projeto

```
doc-pipeline/
├── api.py                 # FastAPI REST server
├── cli.py                 # Command-line interface
├── start-server.sh        # Script para iniciar API + ngrok
├── stop-server.sh         # Script para parar todos os serviços
├── models/
│   └── classifier.pth     # Modelo EfficientNet (não versionado)
├── doc_pipeline/
│   ├── pipeline.py        # Orquestrador principal
│   ├── config.py          # Configurações (pydantic-settings)
│   ├── schemas.py         # Modelos Pydantic (RGData, CNHData, etc.)
│   ├── classifier/        # Adapter para doc-classifier
│   ├── extractors/        # Backends VLM (Qwen, GOT-OCR)
│   └── prompts/           # Templates de extração
└── docs/
    └── API_FLOW.md        # Documentação detalhada do fluxo
```

## Dependências

- Python 3.12+
- PyTorch 2.0+
- transformers 4.49+
- python-dotenv 1.0+
- PyMuPDF 1.24+ (para suporte a PDF)
- [doc-classifier](https://github.com/henriquefockink/doc-classifier) (dependência local)

### Opcionais

```bash
# Flash Attention 2 (melhora performance ~20%)
pip install -e ".[flash]"
```

## Desenvolvimento

### Setup

```bash
# Instalar dependências de desenvolvimento
pip install -e ".[dev]"

# Configurar pre-commit hooks
pre-commit install
```

### Code Quality

O projeto usa [Ruff](https://docs.astral.sh/ruff/) para linting e formatação (substitui black, isort e flake8).

```bash
# Rodar linter
ruff check .

# Rodar linter com auto-fix
ruff check . --fix

# Formatar código
ruff format .

# Rodar todos os hooks (inclui ruff)
pre-commit run --all-files
```

### Regras configuradas

| Regra | Descrição |
|-------|-----------|
| E | pycodestyle errors |
| F | pyflakes |
| I | isort (ordenação de imports) |
| B | flake8-bugbear |
| UP | pyupgrade |
| SIM | flake8-simplify |

## Licença

MIT
