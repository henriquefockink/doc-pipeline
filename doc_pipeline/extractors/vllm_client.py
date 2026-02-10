"""
Async HTTP client for vLLM OpenAI-compatible API.

This client is for use via CLI or when running vLLM as a separate HTTP server.
Production uses `VLLMEmbeddedClient` via `inference_server.py` (in-process, zero overhead).

Sends VLM requests to an external vLLM server with continuous batching.
vLLM handles batching internally — this client fires N requests in parallel.
"""

import asyncio
import base64
import io

import httpx
from PIL import Image

from ..observability import get_logger

logger = get_logger("vllm_client")


class VLLMClient:
    """Async HTTP client for vLLM OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str = "http://vllm:8000/v1",
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        """Create the HTTP connection pool."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout, connect=10.0),
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
        logger.info("vllm_client_started", base_url=self.base_url, model=self.model)

    async def stop(self) -> None:
        """Close the HTTP connection pool."""
        if self._client:
            await self._client.aclose()
            self._client = None
        logger.info("vllm_client_stopped")

    def _encode_image(self, image: Image.Image, max_side: int = 1280) -> str:
        """Encode PIL Image to base64 JPEG data URL, resizing if needed.

        Large images generate too many vision tokens for the model context.
        A 3024x4032 image produces ~15k tokens vs max_model_len=4096.
        Resizing to max 1280px keeps tokens within limits.
        """
        w, h = image.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"

    async def generate(self, image: Image.Image, prompt: str) -> str:
        """Send a single VLM request to vLLM and return the generated text."""
        data_url = self._encode_image(image)

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
        }

        last_error = None
        for attempt in range(3):
            try:
                resp = await self._client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                result = resp.json()
                return result["choices"][0]["message"]["content"]
            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_error = e
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code < 500:
                    logger.error(
                        "vllm_client_error",
                        status=e.response.status_code,
                        body=e.response.text[:500],
                    )
                    raise
                if attempt < 2:
                    logger.warning(
                        "vllm_request_retry",
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    await asyncio.sleep(1.0 * (attempt + 1))

        raise last_error

    async def generate_batch(self, images: list[Image.Image], prompts: list[str]) -> list[str]:
        """Send N requests in parallel — vLLM handles continuous batching internally.

        Uses return_exceptions=True so one failed request doesn't kill the whole batch.
        Individual failures are re-raised as strings that the caller can detect.
        """
        tasks = [self.generate(img, prompt) for img, prompt in zip(images, prompts, strict=True)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert results: keep strings, re-raise exceptions as error marker strings
        output = []
        for r in results:
            if isinstance(r, BaseException):
                # Return a sentinel that _build_extraction_result will handle
                output.append(f"__ERROR__: {r}")
            else:
                output.append(r)
        return output

    async def health_check(self) -> bool:
        """Check if vLLM server is ready by querying /models."""
        try:
            resp = await self._client.get("/models")
            return resp.status_code == 200
        except Exception:
            return False

    @staticmethod
    def parse_json(text: str) -> dict:
        """Extract and parse JSON from VLM response text."""
        from .vlm_utils import parse_vlm_json

        return parse_vlm_json(text)
