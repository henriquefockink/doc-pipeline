"""
Extractor híbrido: EasyOCR para texto + VLM para estruturar.

Combina a precisão do OCR para dígitos com a inteligência do VLM para contexto.
"""

import json
import re
from pathlib import Path

from PIL import Image

from ..schemas import CNHData, RGData
from .base import BaseExtractor


# Prompts específicos para o modo híbrido (recebe texto, não imagem)
CNH_HYBRID_PROMPT = """Você deve extrair dados de uma CNH brasileira. O texto abaixo foi extraído por OCR.

TEXTO DO DOCUMENTO:
{ocr_text}

TAREFA: Encontre no texto acima os seguintes valores e copie-os EXATAMENTE como aparecem:

1. NOME: Nome completo da pessoa (ex: RAUL DE OLIVEIRA)
2. CPF: Procure um número no formato ###.###.###-## (11 dígitos com pontos e hífen)
3. DATA NASCIMENTO: Data no formato DD/MM/AAAA
4. N° REGISTRO: 11 dígitos consecutivos sem pontuação
5. VALIDADE: Data futura
6. CATEGORIA: Letra(s) da categoria (A, B, AB, etc)

REGRA CRÍTICA: Copie os números EXATAMENTE como aparecem no texto. NÃO modifique, NÃO corrija, NÃO recalcule nenhum dígito.

JSON:
{{
    "nome": "copie exatamente",
    "cpf": "copie exatamente no formato ###.###.###-##",
    "data_nascimento": "DD/MM/AAAA",
    "numero_registro": "11 dígitos",
    "validade": "DD/MM/AAAA",
    "categoria": "letra(s)",
    "primeira_habilitacao": "DD/MM/AAAA"
}}"""

RG_HYBRID_PROMPT = """Analise o texto abaixo extraído de um RG (Registro Geral) brasileiro e estruture os dados.

TEXTO EXTRAÍDO DO DOCUMENTO:
{ocr_text}

IMPORTANTE:
- Use EXATAMENTE os valores que aparecem no texto acima
- NÃO modifique números ou dígitos - copie exatamente como estão
- Se um campo não estiver no texto, retorne null
- O CPF deve estar no formato ###.###.###-##
- Datas no formato DD/MM/AAAA

Retorne APENAS um JSON válido:
{{
    "nome": "nome completo da pessoa",
    "nome_pai": "nome do pai",
    "nome_mae": "nome da mãe",
    "data_nascimento": "DD/MM/AAAA",
    "naturalidade": "cidade/estado",
    "cpf": "CPF exatamente como aparece no texto",
    "rg": "número do RG",
    "data_expedicao": "DD/MM/AAAA",
    "orgao_expedidor": "órgão expedidor"
}}

Responda SOMENTE com o JSON."""


class HybridExtractor(BaseExtractor):
    """
    Extractor híbrido: EasyOCR + VLM.

    1. EasyOCR extrai texto bruto (preciso para dígitos)
    2. VLM estrutura o texto em JSON (bom para contexto)
    """

    backend_name = "hybrid"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda:0",
    ):
        self.model_name = model_name
        self.device = device
        self._ocr_engine = None
        self._model = None
        self._processor = None

    def _get_ocr_engine(self):
        """Lazy load OCR engine."""
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

        from transformers import AutoProcessor
        import torch

        print(f"[Hybrid] Carregando modelo {self.model_name}...")

        ModelClass = self._get_model_class()

        try:
            self._model = ModelClass.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                attn_implementation="flash_attention_2",
            )
        except Exception:
            print("[Hybrid] Flash Attention 2 não disponível, usando SDPA...")
            self._model = ModelClass.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                attn_implementation="sdpa",
            )

        self._processor = AutoProcessor.from_pretrained(self.model_name)
        print(f"[Hybrid] Modelo carregado em {self.device}")

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

    def _extract_ocr_text(self, image: Image.Image) -> str:
        """Extrai texto da imagem usando EasyOCR."""
        ocr = self._get_ocr_engine()
        text, confidence = ocr.extract_text(image)
        return text

    def _generate_text_only(self, prompt: str) -> str:
        """Gera resposta do modelo apenas com texto (sem imagem)."""
        self.load_model()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        generated_ids = self._model.generate(**inputs, max_new_tokens=1024)

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]

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

    def _extract_cpf_from_text(self, text: str) -> str | None:
        """Extrai CPF do texto usando regex. Formato: ###.###.###-##"""
        # Padrão 1: CPF completo com pontuação
        cpf_pattern = r'\b(\d{3}\.\d{3}\.\d{3}-\d{2})\b'
        matches = re.findall(cpf_pattern, text)
        if matches:
            return matches[0]

        # Padrão 2: CPF fragmentado em linhas (comum em OCR)
        # Procura pelo label "CPF" e junta os próximos números
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if 'CPF' in line.upper() and i + 3 < len(lines):
                # Procura nas próximas linhas por partes do CPF
                parts = []
                for j in range(i + 1, min(i + 6, len(lines))):
                    next_line = lines[j].strip()
                    # Extrai apenas dígitos e hífen
                    digits = re.findall(r'[\d-]+', next_line)
                    if digits:
                        parts.extend(digits)
                    # Se já temos o suficiente, para
                    joined = ''.join(parts).replace('-', '')
                    if len(joined) >= 11:
                        break

                # Tenta formar o CPF
                all_digits = ''.join(parts).replace('-', '').replace('.', '')
                if len(all_digits) >= 11:
                    # Pega os primeiros 11 dígitos e formata
                    cpf_digits = all_digits[:11]
                    formatted = f"{cpf_digits[:3]}.{cpf_digits[3:6]}.{cpf_digits[6:9]}-{cpf_digits[9:11]}"
                    return formatted

        return None

    def _extract_registro_from_text(self, text: str) -> str | None:
        """Extrai número de registro (11 dígitos sem pontuação)."""
        # Procura por 11 dígitos consecutivos que NÃO fazem parte de um CPF
        lines = text.split('\n')
        for line in lines:
            # Ignora linhas com formato de CPF
            if re.search(r'\d{3}\.\d{3}\.\d{3}-\d{2}', line):
                continue
            # Procura 11 dígitos consecutivos
            match = re.search(r'\b(\d{11})\b', line)
            if match:
                return match.group(1)
        return None

    def extract_text(self, image: str | Path | Image.Image) -> str:
        """Extrai texto bruto usando EasyOCR."""
        img = self._load_image(image)
        return self._extract_ocr_text(img)

    def extract_rg(self, image: str | Path | Image.Image) -> RGData:
        """Extrai dados de um RG usando modo híbrido."""
        img = self._load_image(image)

        # Step 1: OCR
        ocr_text = self._extract_ocr_text(img)
        print(f"[Hybrid] OCR extracted text:\n{ocr_text}\n")

        # Step 2: Extrai CPF via regex (mais confiável para números)
        cpf_regex = self._extract_cpf_from_text(ocr_text)
        print(f"[Hybrid] Regex CPF: {cpf_regex}")

        # Step 3: VLM estrutura o resto
        prompt = RG_HYBRID_PROMPT.format(ocr_text=ocr_text)
        response = self._generate_text_only(prompt)
        print(f"[Hybrid] VLM response:\n{response}\n")
        data = self._parse_json(response)

        # Override CPF com valor do regex (mais confiável)
        if cpf_regex:
            data["cpf"] = cpf_regex

        print(f"[Hybrid] Final RG data: {data}")

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
        """Extrai dados de uma CNH usando modo híbrido."""
        img = self._load_image(image)

        # Step 1: OCR
        ocr_text = self._extract_ocr_text(img)
        print(f"[Hybrid] OCR extracted text:\n{ocr_text}\n")

        # Step 2: Extrai CPF e Registro via regex (mais confiável para números)
        cpf_regex = self._extract_cpf_from_text(ocr_text)
        registro_regex = self._extract_registro_from_text(ocr_text)
        print(f"[Hybrid] Regex CPF: {cpf_regex}, Registro: {registro_regex}")

        # Step 3: VLM estrutura o resto
        prompt = CNH_HYBRID_PROMPT.format(ocr_text=ocr_text)
        response = self._generate_text_only(prompt)
        print(f"[Hybrid] VLM response:\n{response}\n")
        data = self._parse_json(response)

        # Override com valores do regex (mais confiáveis)
        if cpf_regex:
            data["cpf"] = cpf_regex
        if registro_regex:
            data["numero_registro"] = registro_regex

        print(f"[Hybrid] Final data: {data}")

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
