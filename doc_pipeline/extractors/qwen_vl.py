"""
Extractor usando Qwen3-VL para extração contextualizada de dados.
"""

import json
import re
from pathlib import Path

from PIL import Image

from ..prompts import CIN_EXTRACTION_PROMPT, CNH_EXTRACTION_PROMPT, RG_EXTRACTION_PROMPT
from ..schemas import CINData, CNHData, RGData
from ..utils import fix_cpf_rg_swap
from .base import BaseExtractor


class QwenVLExtractor(BaseExtractor):
    """Extractor usando Qwen3-VL-8B-Instruct."""

    backend_name = "qwen-vl"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda:0",
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
    ):
        """
        Inicializa o extractor Qwen-VL.

        Args:
            model_name: Nome do modelo no HuggingFace
            device: Device para inferência
            min_pixels: Mínimo de pixels para redimensionamento
            max_pixels: Máximo de pixels para redimensionamento
        """
        self.model_name = model_name
        self.device = device
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self._model = None
        self._processor = None

    @property
    def is_loaded(self) -> bool:
        """Whether the model is loaded and ready for inference."""
        return self._model is not None

    def _get_model_class(self):
        """Get the appropriate model class based on model name."""
        if "Qwen3-VL" in self.model_name:
            from transformers import Qwen3VLForConditionalGeneration

            return Qwen3VLForConditionalGeneration
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration

            return Qwen2_5_VLForConditionalGeneration

    def load_model(self) -> None:
        """Carrega o modelo Qwen-VL."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor

        print(f"Carregando modelo {self.model_name}...")

        ModelClass = self._get_model_class()

        # Tenta usar flash_attention_2, senão usa sdpa (PyTorch nativo)
        try:
            self._model = ModelClass.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                attn_implementation="flash_attention_2",
            )
        except Exception:
            print("Flash Attention 2 não disponível, usando SDPA...")
            self._model = ModelClass.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                attn_implementation="sdpa",
            )

        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        print(f"Modelo {self.model_name} carregado em {self.device}")

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

    def _generate(self, image: Image.Image, prompt: str) -> str:
        """Gera resposta do modelo para uma imagem e prompt."""
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

    def _generate_batch(self, images: list[Image.Image], prompts: list[str]) -> list[str]:
        """Generate responses for a batch of images and prompts in a single forward pass."""
        from qwen_vl_utils import process_vision_info

        self.load_model()

        all_texts = []
        all_image_inputs = []

        for image, prompt in zip(images, prompts, strict=True):
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
            all_texts.append(text)

            img_inputs, _ = process_vision_info(messages)
            if img_inputs:
                all_image_inputs.extend(img_inputs)

        inputs = self._processor(
            text=all_texts,
            images=all_image_inputs if all_image_inputs else None,
            videos=None,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        generated_ids = self._model.generate(**inputs, max_new_tokens=1024)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=True)
        ]

        return self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def _parse_json(self, text: str) -> dict:
        """Extrai e parseia JSON da resposta do modelo."""
        # Tenta extrair JSON de blocos de código
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if json_match:
            text = json_match.group(1)

        # Tenta encontrar objeto JSON
        json_match = re.search(r"\{[\s\S]*\}", text)
        if json_match:
            text = json_match.group(0)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def extract_text(self, image: str | Path | Image.Image) -> str:
        """Extrai texto bruto da imagem usando OCR."""
        img = self._load_image(image)
        prompt = "Extraia todo o texto visível nesta imagem de documento. Retorne apenas o texto, preservando a estrutura original."
        return self._generate(img, prompt)

    def extract_rg(self, image: str | Path | Image.Image) -> RGData:
        """Extrai dados de um RG."""
        img = self._load_image(image)
        response = self._generate(img, RG_EXTRACTION_PROMPT)
        data = self._parse_json(response)

        # Fix CPF/RG swap (VLM sometimes confuses them) and normalize CPF
        data = fix_cpf_rg_swap(data)

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
        """Extrai dados de uma CNH."""
        img = self._load_image(image)
        response = self._generate(img, CNH_EXTRACTION_PROMPT)
        data = self._parse_json(response)

        # Fix CPF/doc_identidade swap and normalize CPF
        data = fix_cpf_rg_swap(data)

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
        """Extrai dados de uma CIN."""
        img = self._load_image(image)
        response = self._generate(img, CIN_EXTRACTION_PROMPT)
        data = self._parse_json(response)

        # Fix CPF swap and normalize CPF
        data = fix_cpf_rg_swap(data)

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
