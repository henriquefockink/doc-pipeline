"""
In-process vLLM backend for zero-overhead VLM inference.

Uses vLLM's LLM class directly — PIL images go straight to the GPU
without base64 encoding or HTTP serialization.
"""

from PIL import Image

from ..observability import get_logger
from .vllm_client import VLLMClient

logger = get_logger("vllm_embedded")


class VLLMEmbeddedClient:
    """In-process vLLM client using the offline LLM class.

    Passes PIL images directly to vLLM via multi_modal_data,
    eliminating base64 + HTTP overhead that hurts high-concurrency performance.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        gpu_memory_utilization: float = 0.40,
        max_model_len: int = 4096,
        max_tokens: int = 1024,
        max_num_seqs: int = 8,
    ):
        self.model_name = model
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_tokens = max_tokens
        self.max_num_seqs = max_num_seqs
        self._llm = None
        self._sampling_params = None

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    def start(self) -> None:
        """Load the vLLM model in-process (blocking, call once at startup)."""
        from vllm import LLM, SamplingParams

        logger.info(
            "vllm_embedded_loading",
            model=self.model_name,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
        )

        self._llm = LLM(
            model=self.model_name,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            gpu_memory_utilization=self.gpu_memory_utilization,
            mm_processor_kwargs={
                "min_pixels": 28 * 28,
                "max_pixels": 1280 * 28 * 28,
            },
            limit_mm_per_prompt={"image": 1},
            dtype="auto",
            trust_remote_code=True,
        )

        self._sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_tokens,
        )

        logger.info("vllm_embedded_loaded", model=self.model_name)

    def stop(self) -> None:
        """Release the vLLM model and free GPU memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._sampling_params = None

            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("vllm_embedded_stopped")

    def _build_prompt(self, text: str) -> str:
        """Build Qwen2.5-VL chat prompt with image placeholder."""
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>"
            f"{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def _resize_image(self, image: Image.Image, max_side: int = 1280) -> Image.Image:
        """Resize image if too large (same logic as VLLMClient HTTP)."""
        w, h = image.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def generate(self, image: Image.Image, prompt: str) -> str:
        """Generate VLM response for a single image+prompt (blocking)."""
        image = self._resize_image(image)
        formatted_prompt = self._build_prompt(prompt)

        outputs = self._llm.generate(
            {"prompt": formatted_prompt, "multi_modal_data": {"image": image}},
            sampling_params=self._sampling_params,
        )

        return outputs[0].outputs[0].text

    def generate_batch(self, images: list[Image.Image], prompts: list[str]) -> list[str]:
        """Generate VLM responses for a batch (blocking, vLLM batches internally)."""
        inputs = []
        for image, prompt in zip(images, prompts, strict=True):
            image = self._resize_image(image)
            formatted_prompt = self._build_prompt(prompt)
            inputs.append({"prompt": formatted_prompt, "multi_modal_data": {"image": image}})

        outputs = self._llm.generate(inputs, sampling_params=self._sampling_params)

        return [out.outputs[0].text for out in outputs]

    @staticmethod
    def parse_json(text: str) -> dict:
        """Parse JSON from VLM response (reuses VLLMClient logic)."""
        return VLLMClient.parse_json(text)
