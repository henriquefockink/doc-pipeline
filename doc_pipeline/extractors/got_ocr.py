"""
Extractor usando GOT-OCR2 para OCR puro de documentos.

Usa a versão integrada ao transformers (GOT-OCR-2.0-hf).
"""

import re
from pathlib import Path

import structlog
from PIL import Image

from ..schemas import CNHData, RGData
from .base import BaseExtractor

logger = structlog.get_logger("got_ocr")


class GOTOCRExtractor(BaseExtractor):
    """Extractor usando GOT-OCR-2.0 para OCR de documentos."""

    backend_name = "got-ocr"
    supports_pdf = True

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
        """Carrega o modelo GOT-OCR (versão integrada ao transformers)."""
        if self._model is not None:
            return

        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info("loading_model", model=self.model_name)

        self._processor = AutoProcessor.from_pretrained(self.model_name)

        import torch
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            device_map=self.device,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        self._model = self._model.eval()

        logger.info("model_loaded", model=self.model_name, device=self.device)

    def unload_model(self) -> None:
        """Descarrega o modelo para liberar memória."""
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _run_ocr(self, image: Image.Image) -> str:
        """Executa OCR em uma imagem PIL."""
        return self._run_ocr_batch([image])[0]

    def _run_ocr_batch(self, images: list[Image.Image]) -> list[str]:
        """Executa OCR em batch de imagens PIL."""
        import torch

        self.load_model()

        # Processa as imagens em batch
        inputs = self._processor(images, return_tensors="pt").to(self.device)

        # Gera o texto com inference_mode para performance
        with torch.inference_mode():
            generate_ids = self._model.generate(
                **inputs,
                do_sample=False,
                tokenizer=self._processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=1024,  # Reduzido - maioria das páginas não precisa mais
                num_beams=1,
                use_cache=True,
                pad_token_id=self._processor.tokenizer.pad_token_id,
            )

        # Decodifica em batch
        results = self._processor.batch_decode(
            generate_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        return [r.strip() if r else "" for r in results]

    def extract_text(self, image: str | Path | Image.Image) -> str:
        """Extrai texto bruto da imagem usando OCR."""
        # Carrega a imagem se for um caminho
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image = image.convert("RGB")

        return self._run_ocr(image)

    def _extract_with_format(self, image: str | Path | Image.Image) -> str:
        """Extrai texto com formatação markdown."""
        # Para GOT-OCR-2.0-hf, o formato é controlado pelo prompt
        # Por enquanto, retorna texto simples
        return self.extract_text(image)

    def extract_text_from_pdf(self, pdf_path: str | Path, batch_size: int = 4) -> list[str]:
        """
        Extrai texto de cada página de um PDF usando batch processing.

        Args:
            pdf_path: Caminho do arquivo PDF
            batch_size: Número de páginas por batch (default: 4)

        Returns:
            Lista de textos, um por página
        """
        import fitz  # PyMuPDF

        self.load_model()

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

        doc = fitz.open(pdf_path)
        total_pages = len(doc)

        # Renderiza todas as páginas como imagens (100 DPI - rápido, qualidade OK)
        dpi = 100
        mat = fitz.Matrix(dpi / 72, dpi / 72)

        images = []
        try:
            for page_num in range(total_pages):
                page = doc[page_num]
                pix = page.get_pixmap(matrix=mat)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                images.append(img)
        finally:
            doc.close()

        # Processa em batches
        page_texts = []
        for i in range(0, total_pages, batch_size):
            batch = images[i:i + batch_size]
            logger.debug("processing_batch", start=i + 1, end=min(i + batch_size, total_pages), total=total_pages)
            texts = self._run_ocr_batch(batch)
            page_texts.extend(texts)

        return page_texts

    def _parse_rg_from_text(self, text: str) -> dict:
        """Parseia campos de RG do texto extraído."""
        data = {}
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
        text = self.extract_text(image)
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
        text = self.extract_text(image)
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
