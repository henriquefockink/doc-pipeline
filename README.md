# doc-pipeline

Pipeline de classificação e extração de dados de documentos brasileiros (RG/CNH).

## Arquitetura

```
[Imagem] → [Classificador EfficientNet] → [Tipo: RG/CNH]
                                               ↓
                                    [VLM: Qwen2.5-VL ou GOT-OCR2]
                                               ↓
                                    [Dados Estruturados JSON]
```

> Para uma visão detalhada do fluxo de execução da API, consulte [docs/API_FLOW.md](docs/API_FLOW.md).

## Instalação

```bash
# Clone o repositório
git clone <repo-url>
cd doc-pipeline

# Instale as dependências (inclui doc-classifier como dependência editável)
pip install -r requirements.txt
```

## Uso

### CLI

```bash
# Pipeline completo (classificação + extração)
python cli.py documento.jpg -m ../doc-classifier/modelos/modelo.pth

# Usar GOT-OCR2 ao invés de Qwen (menor uso de VRAM)
python cli.py documento.jpg -m ../doc-classifier/modelos/modelo.pth --backend got-ocr

# Apenas classificar
python cli.py documento.jpg -m ../doc-classifier/modelos/modelo.pth --no-extraction

# Processar pasta com saída JSON
python cli.py ./documentos/ -m ../doc-classifier/modelos/modelo.pth --json -o resultados.json

# Multi-GPU: classificador em cuda:0, extractor em cuda:1
python cli.py documento.jpg -m ../doc-classifier/modelos/modelo.pth \
  --classifier-device cuda:0 --extractor-device cuda:1
```

### API

```bash
# Iniciar servidor
DOC_PIPELINE_CLASSIFIER_MODEL_PATH=../doc-classifier/modelos/modelo.pth python api.py

# Ou com GOT-OCR
DOC_PIPELINE_CLASSIFIER_MODEL_PATH=../doc-classifier/modelos/modelo.pth \
  DOC_PIPELINE_EXTRACTOR_BACKEND=got-ocr python api.py
```

#### Endpoints

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/process` | Pipeline completo (classificação + extração) |
| POST | `/classify` | Apenas classificação |
| POST | `/extract?doc_type=rg_frente` | Apenas extração (tipo conhecido) |
| GET | `/health` | Status da API |
| GET | `/classes` | Lista classes suportadas |

#### Exemplos

```bash
# Pipeline completo
curl -X POST http://localhost:8001/process -F "arquivo=@documento.jpg"

# Apenas classificar
curl -X POST http://localhost:8001/classify -F "arquivo=@documento.jpg"

# Extrair dados (tipo conhecido)
curl -X POST "http://localhost:8001/extract?doc_type=rg_frente" -F "arquivo=@rg.jpg"
```

### Python

```python
from doc_pipeline import DocumentPipeline

# Inicializa pipeline
pipeline = DocumentPipeline(
    classifier_model_path="../doc-classifier/modelos/modelo.pth",
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

```env
# Classificador
DOC_PIPELINE_CLASSIFIER_MODEL_PATH=../doc-classifier/modelos/modelo.pth
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

# Geral
DOC_PIPELINE_MIN_CONFIDENCE=0.5
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

## Dependências

- Python 3.12+
- PyTorch 2.0+
- transformers 4.49+
- doc-classifier (dependência local)

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
