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
python cli.py documento.jpg -m /path/to/model.pth

# Run CLI - classification only
python cli.py documento.jpg -m /path/to/model.pth --no-extraction

# Run CLI - with GOT-OCR backend (lower VRAM)
python cli.py documento.jpg -m /path/to/model.pth --backend got-ocr

# Run API server
DOC_PIPELINE_CLASSIFIER_MODEL_PATH=/path/to/model.pth python api.py
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

**Entry points**: `cli.py` (command-line), `api.py` (FastAPI REST server)

## Key Patterns

- **Lazy loading**: DocumentPipeline loads classifier/extractor only when first used (via `@property`)
- **Abstract base**: New extractors extend `BaseExtractor` with `extract()` method
- **Pydantic validation**: All data models use Pydantic; config uses pydantic-settings
- **External dependency**: Requires `doc-classifier` package installed as editable (`-e ../doc-classifier`)

## Configuration

Required environment variable: `DOC_PIPELINE_CLASSIFIER_MODEL_PATH`

Key settings (all prefixed with `DOC_PIPELINE_`):
- `CLASSIFIER_MODEL_TYPE`: efficientnet_b0, efficientnet_b2, efficientnet_b4
- `EXTRACTOR_BACKEND`: qwen-vl (default) or got-ocr
- `CLASSIFIER_DEVICE` / `EXTRACTOR_DEVICE`: cuda:N or cpu (supports multi-GPU)

## Document Types

8 classes: `rg_frente`, `rg_verso`, `rg_aberto`, `rg_digital`, `cnh_frente`, `cnh_verso`, `cnh_aberta`, `cnh_digital`

Extraction yields `RGData` or `CNHData` based on document type (defined in `schemas.py`).

## Code Quality

Pre-commit hooks ensure consistent code style:
- **Ruff**: Linter and formatter (replaces black, isort, flake8)
- Run `pre-commit run --all-files` to check all files manually
- Hooks run automatically on `git commit`

## Documentation Lookup

Always use Context7 MCP tools when searching for library documentation and best practices:
1. First call `resolve-library-id` to get the Context7-compatible library ID
2. Then call `query-docs` with the library ID to fetch up-to-date documentation and code examples
