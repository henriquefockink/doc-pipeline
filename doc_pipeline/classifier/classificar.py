#!/usr/bin/env python3
"""
Classificador de Documentos - Inferência
========================================
Classifica imagens de RG/CNH usando EfficientNet treinado.

Copiado de: https://github.com/org/doc-classifier
Apenas código de inferência. Código de treino permanece no repo original.
"""

import torch
from torchvision import transforms, models
from PIL import Image
from pathlib import Path

# FP8 support (optional)
try:
    from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight
    TORCHAO_AVAILABLE = True
except ImportError:
    TORCHAO_AVAILABLE = False


class ClassificadorDocumentos:
    """Classifica documentos brasileiros (RG, CNH)"""

    CLASSES = [
        "cnh_aberta",
        "cnh_digital",
        "cnh_frente",
        "cnh_verso",
        "rg_aberto",
        "rg_digital",
        "rg_frente",
        "rg_verso",
    ]

    INPUT_SIZES = {
        "efficientnet_b0": 224,
        "efficientnet_b2": 260,
        "efficientnet_b4": 380,
    }

    def __init__(self, modelo_path: str, modelo_tipo: str = "efficientnet_b0", device: str = None, fp8: bool = False):
        """
        Args:
            modelo_path: Caminho para o arquivo .pth do modelo treinado
            modelo_tipo: Tipo do modelo (efficientnet_b0, b2, b4)
            device: 'cuda' ou 'cpu' (auto-detecta se None)
            fp8: Se True, quantiza modelo para FP8 (requer GPU Hopper/Blackwell)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.modelo_tipo = modelo_tipo
        self.input_size = self.INPUT_SIZES[modelo_tipo]
        self.fp8_enabled = False

        self._load_model(modelo_path)
        self._setup_transforms()

        if fp8:
            self._quantize_fp8()

        print(f"Classificador carregado: {modelo_tipo} em {self.device}" + (" [FP8]" if self.fp8_enabled else ""))

    def _load_model(self, modelo_path: str):
        """Carrega modelo treinado"""
        checkpoint = torch.load(modelo_path, map_location=self.device, weights_only=False)

        # Carrega classes do checkpoint se disponível
        if "classes" in checkpoint:
            self.classes = checkpoint["classes"]
        else:
            self.classes = self.CLASSES

        num_classes = len(self.classes)

        # Cria modelo
        if self.modelo_tipo == "efficientnet_b0":
            self.model = models.efficientnet_b0(weights=None)
        elif self.modelo_tipo == "efficientnet_b2":
            self.model = models.efficientnet_b2(weights=None)
        elif self.modelo_tipo == "efficientnet_b4":
            self.model = models.efficientnet_b4(weights=None)

        # Ajusta classifier
        in_features = self.model.classifier[1].in_features
        self.model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3, inplace=True),
            torch.nn.Linear(in_features, num_classes),
        )

        # Carrega pesos
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

    def _setup_transforms(self):
        """Configura transforms de inferência"""
        self.transform = transforms.Compose([
            transforms.Resize(int(self.input_size * 1.1)),
            transforms.CenterCrop(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def _quantize_fp8(self):
        """Quantiza modelo para FP8 (Hopper/Blackwell)"""
        if not TORCHAO_AVAILABLE:
            print("Aviso: torchao não instalado. FP8 desabilitado.")
            return

        if self.device == "cpu":
            print("Aviso: FP8 requer GPU. Ignorando quantização.")
            return

        # Verifica se GPU suporta FP8 (sm_89+ = Ada, sm_90+ = Hopper, sm_100+ = Blackwell)
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability[0] < 9:
                print(f"Aviso: GPU (sm_{capability[0]}{capability[1]}) não suporta FP8. Requer Hopper (sm_90+) ou Blackwell (sm_100+).")
                return

        try:
            quantize_(self.model, float8_dynamic_activation_float8_weight())
            self.model = torch.compile(self.model, mode="max-autotune")
            self.fp8_enabled = True
        except Exception as e:
            print(f"Aviso: Falha ao quantizar FP8: {e}")

    def classificar(self, imagem) -> dict:
        """
        Classifica uma imagem.

        Args:
            imagem: Path da imagem, PIL.Image, ou tensor

        Returns:
            dict com 'classe', 'confianca', e 'probabilidades'
        """
        # Carrega imagem se necessário
        if isinstance(imagem, (str, Path)):
            imagem = Image.open(imagem).convert("RGB")
        elif not isinstance(imagem, Image.Image):
            raise ValueError("Imagem deve ser path, PIL.Image ou tensor")

        # Preprocessa
        img_tensor = self.transform(imagem).unsqueeze(0).to(self.device)

        # Inferência
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, idx = probs.max(dim=1)

        classe = self.classes[idx.item()]
        confianca = conf.item()

        # Todas as probabilidades
        probabilidades = {
            self.classes[i]: probs[0, i].item()
            for i in range(len(self.classes))
        }

        return {
            "classe": classe,
            "confianca": confianca,
            "probabilidades": probabilidades,
        }

    def classificar_batch(self, imagens: list) -> list:
        """Classifica múltiplas imagens"""
        return [self.classificar(img) for img in imagens]
