"""
Extractor usando GOT-OCR2 para OCR puro de documentos.
"""

import json
import re
from pathlib import Path

from PIL import Image

from ..schemas import CNHData, RGData
from .base import BaseExtractor


class GOTOCRExtractor(BaseExtractor):
    """Extractor usando GOT-OCR-2.0 para OCR de documentos."""

    backend_name = "got-ocr"

    def __init__(
        self,
        model_name: str = "stepfun-ai/GOT-OCR-2.0-hf",
        device: str = "cuda:0",
    ):
        """
        Inicializa o extractor GOT-OCR.

        Args:
            model_name: Nome do modelo no HuggingFace
            device: Device para inferência
        """
        self.model_name = model_name
        self.device = device

        self._model = None
        self._processor = None

    def load_model(self) -> None:
        """Carrega o modelo GOT-OCR."""
        if self._model is not None:
            return

        from transformers import AutoModel, AutoTokenizer
        import torch

        print(f"Carregando modelo {self.model_name}...")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
        )

        self._model = AutoModel.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            low_cpu_mem_usage=True,
        )
        self._model = self._model.eval()

        print(f"Modelo {self.model_name} carregado em {self.device}")

    def unload_model(self) -> None:
        """Descarrega o modelo para liberar memória."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def extract_text(self, image: str | Path | Image.Image) -> str:
        """Extrai texto bruto da imagem usando OCR."""
        self.load_model()

        # GOT-OCR espera path ou PIL Image
        if isinstance(image, (str, Path)):
            image_path = str(image)
        else:
            # Salva temporariamente se for PIL Image
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                image.save(f.name)
                image_path = f.name

        # Usa modo "ocr" para extração de texto
        result = self._model.chat(
            self._tokenizer,
            image_path,
            ocr_type="ocr",
        )

        return result

    def _extract_with_format(self, image: str | Path | Image.Image) -> str:
        """Extrai texto com formatação markdown."""
        self.load_model()

        if isinstance(image, (str, Path)):
            image_path = str(image)
        else:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                image.save(f.name)
                image_path = f.name

        # Usa modo "format" para extração estruturada
        result = self._model.chat(
            self._tokenizer,
            image_path,
            ocr_type="format",
        )

        return result

    def _parse_rg_from_text(self, text: str) -> dict:
        """Parseia campos de RG do texto extraído."""
        data = {}
        text_lower = text.lower()
        lines = text.split("\n")

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()

            # Nome - geralmente após "NOME" ou é a primeira linha grande
            if "nome" in line_lower and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and len(next_line) > 5:
                    data["nome"] = next_line

            # Filiação
            if "filiação" in line_lower or "pai" in line_lower or "mae" in line_lower:
                if i + 1 < len(lines):
                    data["nome_pai"] = lines[i + 1].strip()
                if i + 2 < len(lines):
                    data["nome_mae"] = lines[i + 2].strip()

            # Data de nascimento - formato DD/MM/AAAA
            date_match = re.search(r"(\d{2}/\d{2}/\d{4})", line)
            if date_match:
                if "nascimento" in line_lower or "nasc" in line_lower:
                    data["data_nascimento"] = date_match.group(1)
                elif "expedição" in line_lower or "exped" in line_lower:
                    data["data_expedicao"] = date_match.group(1)

            # CPF
            cpf_match = re.search(r"(\d{3}\.\d{3}\.\d{3}-\d{2})", line)
            if cpf_match:
                data["cpf"] = cpf_match.group(1)

            # RG
            if "registro geral" in line_lower or line_lower.startswith("rg"):
                rg_match = re.search(r"[\d\.\-]+", line)
                if rg_match:
                    data["rg"] = rg_match.group(0)

            # Naturalidade
            if "naturalidade" in line_lower or "natural" in line_lower:
                if i + 1 < len(lines):
                    data["naturalidade"] = lines[i + 1].strip()

            # Órgão expedidor
            if "ssp" in line_lower or "secretaria" in line_lower:
                org_match = re.search(r"SSP[\-/]?[A-Z]{2}", line.upper())
                if org_match:
                    data["orgao_expedidor"] = org_match.group(0)

        return data

    def _parse_cnh_from_text(self, text: str) -> dict:
        """Parseia campos de CNH do texto extraído."""
        data = {}
        lines = text.split("\n")

        for i, line in enumerate(lines):
            line_lower = line.lower().strip()

            # Nome
            if "nome" in line_lower and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if next_line and len(next_line) > 5:
                    data["nome"] = next_line

            # CPF
            cpf_match = re.search(r"(\d{3}\.\d{3}\.\d{3}-\d{2})", line)
            if cpf_match:
                data["cpf"] = cpf_match.group(1)

            # Datas
            date_match = re.search(r"(\d{2}/\d{2}/\d{4})", line)
            if date_match:
                if "nascimento" in line_lower or "nasc" in line_lower:
                    data["data_nascimento"] = date_match.group(1)
                elif "validade" in line_lower or "valid" in line_lower:
                    data["validade"] = date_match.group(1)
                elif "primeira" in line_lower or "1ª" in line_lower:
                    data["primeira_habilitacao"] = date_match.group(1)

            # Registro
            if "registro" in line_lower or "n°" in line_lower:
                reg_match = re.search(r"(\d{9,12})", line)
                if reg_match:
                    data["numero_registro"] = reg_match.group(1)

            # Espelho
            if "espelho" in line_lower:
                esp_match = re.search(r"(\d{9,12})", line)
                if esp_match:
                    data["numero_espelho"] = esp_match.group(1)

            # Categoria
            cat_match = re.search(r"\b([ABCDE]{1,2})\b", line.upper())
            if cat_match and ("categoria" in line_lower or "cat" in line_lower):
                data["categoria"] = cat_match.group(1)

            # Observações
            if "observ" in line_lower or "obs" in line_lower:
                if i + 1 < len(lines):
                    data["observacoes"] = lines[i + 1].strip()

        return data

    def extract_rg(self, image: str | Path | Image.Image) -> RGData:
        """Extrai dados de um RG usando OCR."""
        text = self._extract_with_format(image)
        data = self._parse_rg_from_text(text)

        return RGData(
            nome=data.get("nome"),
            nome_pai=data.get("nome_pai"),
            nome_mae=data.get("nome_mae"),
            data_nascimento=data.get("data_nascimento"),
            naturalidade=data.get("naturalidade"),
            cpf=data.get("cpf"),
            rg=data.get("rg"),
            data_expedicao=data.get("data_expedicao"),
            orgao_expedidor=data.get("orgao_expedidor"),
        )

    def extract_cnh(self, image: str | Path | Image.Image) -> CNHData:
        """Extrai dados de uma CNH usando OCR."""
        text = self._extract_with_format(image)
        data = self._parse_cnh_from_text(text)

        return CNHData(
            nome=data.get("nome"),
            cpf=data.get("cpf"),
            data_nascimento=data.get("data_nascimento"),
            numero_registro=data.get("numero_registro"),
            numero_espelho=data.get("numero_espelho"),
            validade=data.get("validade"),
            categoria=data.get("categoria"),
            observacoes=data.get("observacoes"),
            primeira_habilitacao=data.get("primeira_habilitacao"),
        )
