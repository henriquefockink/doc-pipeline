"""
Hybrid extractor: VLM first, OCR fallback for CPF validation.

This extractor is for use via CLI (`cli.py`) and local testing.
Production uses `VLLMEmbeddedClient` via `inference_server.py` with
hybrid CPF validation built into the inference server.

Strategy:
1. VLM (Qwen2.5-VL) extracts data directly from the image
2. Validates CPF with checksum algorithm
3. If CPF invalid, tries EasyOCR to extract numbers
"""

import json
import re
from pathlib import Path

from PIL import Image

from ..observability import get_logger
from ..prompts import CIN_EXTRACTION_PROMPT, CNH_EXTRACTION_PROMPT, RG_EXTRACTION_PROMPT
from ..schemas import CINData, CNHData, RGData
from ..utils import fix_cpf_rg_swap, is_valid_cpf
from .base import BaseExtractor

logger = get_logger("hybrid_extractor")


class HybridExtractor(BaseExtractor):
    """
    Extractor híbrido: VLM + fallback OCR.

    1. VLM extrai dados vendo a imagem (ignora fundo de segurança)
    2. Valida CPF extraído
    3. Se inválido, tenta EasyOCR para números
    """

    backend_name = "hybrid"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda:0",
        ocr_engine=None,
    ):
        self.model_name = model_name
        self.device = device
        self._ocr_engine = ocr_engine
        self._model = None
        self._processor = None

    def _get_ocr_engine(self):
        """Get or lazy load OCR engine."""
        if self._ocr_engine is None:
            from ..ocr import OCREngine

            self._ocr_engine = OCREngine(lang="pt", use_gpu=True)
        return self._ocr_engine

    def _get_model_class(self):
        """Get the appropriate model class based on model name."""
        if "Qwen3-VL" in self.model_name:
            from transformers import Qwen3VLForConditionalGeneration

            return Qwen3VLForConditionalGeneration
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration

            return Qwen2_5_VLForConditionalGeneration

    def load_model(self) -> None:
        """Carrega o modelo VLM."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor

        logger.info("loading_model", model=self.model_name)

        ModelClass = self._get_model_class()

        try:
            self._model = ModelClass.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                attn_implementation="flash_attention_2",
            )
        except Exception:
            logger.info("flash_attention_unavailable_using_sdpa")
            self._model = ModelClass.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                attn_implementation="sdpa",
            )

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        logger.info("model_loaded", device=self.device)

    def unload_model(self) -> None:
        """Descarrega o modelo."""
        if self._model is not None:
            del self._model
            del self._processor
            self._model = None
            self._processor = None

            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _generate_with_image(self, image: Image.Image, prompt: str) -> str:
        """Gera resposta do VLM vendo a imagem diretamente."""
        from qwen_vl_utils import process_vision_info

        self.load_model()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        generated_ids = self._model.generate(**inputs, max_new_tokens=1024)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]

        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]

    def _extract_ocr_text(self, image: Image.Image) -> str:
        """Extrai texto da imagem usando EasyOCR."""
        ocr = self._get_ocr_engine()
        text, confidence = ocr.extract_text(image)
        return text

    def _parse_json(self, text: str) -> dict:
        """Extrai e parseia JSON da resposta."""
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            text = json_match.group(1)

        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            text = json_match.group(0)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def _normalize_cpf(self, cpf: str | None) -> str | None:
        """Normaliza CPF para formato ###.###.###-##"""
        if not cpf:
            return None

        # Remove tudo exceto dígitos
        digits = re.sub(r"\D", "", cpf)

        if len(digits) != 11:
            return None

        return f"{digits[:3]}.{digits[3:6]}.{digits[6:9]}-{digits[9:11]}"

    def _extract_cpf_from_text(self, text: str) -> str | None:
        """Extrai CPF do texto usando regex. Retorna CPF normalizado ou None."""
        # Padrão 1: CPF completo com pontuação (###.###.###-##)
        cpf_pattern = r"\b(\d{3}\.\d{3}\.\d{3}-\d{2})\b"
        matches = re.findall(cpf_pattern, text)
        if matches:
            return matches[0]

        # Padrão 2: CPF com barra (#########/##) - comum em RGs antigos
        cpf_barra_pattern = r"\b(\d{9})/(\d{2})\b"
        matches = re.findall(cpf_barra_pattern, text)
        if matches:
            digits = matches[0][0]
            verifier = matches[0][1]
            return f"{digits[:3]}.{digits[3:6]}.{digits[6:9]}-{verifier}"

        # Padrão 3: 11 dígitos consecutivos (sem formatação)
        digits_pattern = r"\b(\d{11})\b"
        matches = re.findall(digits_pattern, text)
        for match in matches:
            cpf = self._normalize_cpf(match)
            if cpf and is_valid_cpf(cpf):
                return cpf

        # Padrão 4: CPF fragmentado em linhas próximas ao label "CPF"
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if "CPF" in line.upper() and i + 3 < len(lines):
                parts = []
                for j in range(i + 1, min(i + 6, len(lines))):
                    next_line = lines[j].strip()
                    digits = re.findall(r"[\d]+", next_line)
                    if digits:
                        parts.extend(digits)
                    joined = "".join(parts)
                    if len(joined) >= 11:
                        break

                all_digits = "".join(parts)
                if len(all_digits) >= 11:
                    cpf = self._normalize_cpf(all_digits[:11])
                    if cpf:
                        return cpf

        return None

    def _extract_registro_from_text(self, text: str) -> str | None:
        """Extrai número de registro (11 dígitos sem pontuação)."""
        lines = text.split("\n")
        for line in lines:
            # Ignora linhas com formato de CPF
            if re.search(r"\d{3}\.\d{3}\.\d{3}-\d{2}", line):
                continue
            match = re.search(r"\b(\d{11})\b", line)
            if match:
                return match.group(1)
        return None

    def extract_text(self, image: str | Path | Image.Image) -> str:
        """Extrai texto bruto usando EasyOCR."""
        img = self._load_image(image)
        return self._extract_ocr_text(img)

    def extract_rg(self, image: str | Path | Image.Image) -> RGData:
        """
        Extrai dados de um RG.

        Estratégia:
        1. VLM extrai dados vendo a imagem
        2. Valida CPF
        3. Se inválido, tenta OCR para extrair CPF
        """
        img = self._load_image(image)

        # Step 1: VLM extracts data from the image
        logger.info("vlm_extracting", doc_type="rg")
        response = self._generate_with_image(img, RG_EXTRACTION_PROMPT)
        logger.debug("vlm_response", response=response)
        data = self._parse_json(response)

        # Fix CPF/RG swap before validation
        data = fix_cpf_rg_swap(data)

        # Step 2: Validate CPF
        vlm_cpf = self._normalize_cpf(data.get("cpf"))
        cpf_valid = is_valid_cpf(vlm_cpf)
        logger.info("vlm_cpf_result", cpf=vlm_cpf, valid=cpf_valid)

        # Step 3: If CPF invalid, try OCR
        if not cpf_valid:
            logger.info("cpf_invalid_trying_ocr")
            ocr_text = self._extract_ocr_text(img)
            logger.debug("ocr_text", text=ocr_text[:500])

            ocr_cpf = self._extract_cpf_from_text(ocr_text)
            if ocr_cpf and is_valid_cpf(ocr_cpf):
                logger.info("ocr_cpf_found", cpf=ocr_cpf)
                data["cpf"] = ocr_cpf
            elif ocr_cpf:
                logger.info("ocr_cpf_also_invalid", cpf=ocr_cpf)
                if vlm_cpf:
                    data["cpf"] = vlm_cpf
        else:
            data["cpf"] = vlm_cpf

        logger.debug("final_rg_data", data=data)

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
        """
        Extrai dados de uma CNH.

        Estratégia:
        1. VLM extrai dados vendo a imagem
        2. Valida CPF
        3. Se inválido, tenta OCR para extrair CPF e registro
        """
        img = self._load_image(image)

        # Step 1: VLM extracts data from the image
        logger.info("vlm_extracting", doc_type="cnh")
        response = self._generate_with_image(img, CNH_EXTRACTION_PROMPT)
        logger.debug("vlm_response", response=response)
        data = self._parse_json(response)

        # Step 2: Validate CPF
        vlm_cpf = self._normalize_cpf(data.get("cpf"))
        cpf_valid = is_valid_cpf(vlm_cpf)
        logger.info("vlm_cpf_result", cpf=vlm_cpf, valid=cpf_valid)

        # Step 3: If CPF invalid, try OCR
        if not cpf_valid:
            logger.info("cpf_invalid_trying_ocr")
            ocr_text = self._extract_ocr_text(img)
            logger.debug("ocr_text", text=ocr_text[:500])

            ocr_cpf = self._extract_cpf_from_text(ocr_text)
            if ocr_cpf and is_valid_cpf(ocr_cpf):
                logger.info("ocr_cpf_found", cpf=ocr_cpf)
                data["cpf"] = ocr_cpf
            elif ocr_cpf:
                logger.info("ocr_cpf_also_invalid", cpf=ocr_cpf)
                if vlm_cpf:
                    data["cpf"] = vlm_cpf

            # Also try extracting registration number via OCR
            ocr_registro = self._extract_registro_from_text(ocr_text)
            if ocr_registro and not data.get("numero_registro"):
                logger.info("ocr_registro_found", registro=ocr_registro)
                data["numero_registro"] = ocr_registro
        else:
            data["cpf"] = vlm_cpf

        logger.debug("final_cnh_data", data=data)

        return CNHData(
            nome=data.get("nome"),
            cpf=data.get("cpf"),
            data_nascimento=data.get("data_nascimento"),
            doc_identidade=data.get("doc_identidade"),
            numero_registro=data.get("numero_registro"),
            numero_espelho=data.get("numero_espelho"),
            validade=data.get("validade"),
            categoria=data.get("categoria"),
            observacoes=data.get("observacoes"),
            primeira_habilitacao=data.get("primeira_habilitacao"),
        )

    def extract_cin(self, image: str | Path | Image.Image) -> CINData:
        """
        Extrai dados de uma CIN.

        Estratégia:
        1. VLM extrai dados vendo a imagem
        2. Valida CPF
        3. Se inválido, tenta OCR para extrair CPF
        """
        img = self._load_image(image)

        # Step 1: VLM extracts data from the image
        logger.info("vlm_extracting", doc_type="cin")
        response = self._generate_with_image(img, CIN_EXTRACTION_PROMPT)
        logger.debug("vlm_response", response=response)
        data = self._parse_json(response)

        # Fix CPF swap before validation
        data = fix_cpf_rg_swap(data)

        # Step 2: Validate CPF
        vlm_cpf = self._normalize_cpf(data.get("cpf"))
        cpf_valid = is_valid_cpf(vlm_cpf)
        logger.info("vlm_cpf_result", cpf=vlm_cpf, valid=cpf_valid)

        # Step 3: If CPF invalid, try OCR
        if not cpf_valid:
            logger.info("cpf_invalid_trying_ocr")
            ocr_text = self._extract_ocr_text(img)
            logger.debug("ocr_text", text=ocr_text[:500])

            ocr_cpf = self._extract_cpf_from_text(ocr_text)
            if ocr_cpf and is_valid_cpf(ocr_cpf):
                logger.info("ocr_cpf_found", cpf=ocr_cpf)
                data["cpf"] = ocr_cpf
            elif ocr_cpf:
                logger.info("ocr_cpf_also_invalid", cpf=ocr_cpf)
                if vlm_cpf:
                    data["cpf"] = vlm_cpf
        else:
            data["cpf"] = vlm_cpf

        logger.debug("final_cin_data", data=data)

        return CINData(
            nome=data.get("nome"),
            nome_pai=data.get("nome_pai"),
            nome_mae=data.get("nome_mae"),
            data_nascimento=data.get("data_nascimento"),
            naturalidade=data.get("naturalidade"),
            cpf=data.get("cpf"),
            data_expedicao=data.get("data_expedicao"),
            orgao_expedidor=data.get("orgao_expedidor"),
        )
