# Refactor Plan: Centralized Inference Server + vLLM Migration

## Goal

Move ALL GPU models (EfficientNet, EasyOCR, Qwen VLM) to the inference server.
Workers become stateless, zero-GPU queue consumers. Optionally migrate VLM to vLLM for better throughput.

## Current Architecture (before)

```
Worker DocID × 5 (GPU ~6-8GB each = ~35GB total)
  ├── EfficientNet classifier (~200MB)
  ├── EasyOCR reader (~2-4GB) — shared across orientation + extractors
  ├── OrientationCorrector (uses EasyOCR detect)
  └── InferenceClient → sends VLM requests to inference server

Inference Server × 1 (GPU ~12GB)
  └── Qwen2.5-VL (batched inference)

Worker OCR × 1 (GPU ~1GB)
  └── EasyOCR reader + OrientationCorrector

Total GPU: ~48GB
```

## Target Architecture (after)

```
Worker DocID × N (ZERO GPU, ~100MB RAM)
  └── InferenceClient → sends ALL requests to inference server

Inference Server × 1 (GPU ~15GB)
  ├── Qwen2.5-VL (extraction)
  ├── EfficientNet (classification)
  └── EasyOCR (orientation + OCR + hybrid fallback)

Worker OCR × N (ZERO GPU, ~100MB RAM)
  └── InferenceClient → sends OCR requests to inference server

Total GPU: ~15GB
```

---

## Phase 1: Centralize EfficientNet + EasyOCR on Inference Server

### 1.1 New Queue Protocol

Currently the inference server only handles one operation: `extract` (VLM inference).
We need to support multiple operation types via the same Redis queue.

**Request format** (add `operation` field):

```python
# Existing (extract only):
{
    "inference_id": "uuid",
    "request_id": "uuid",
    "document_type": "rg_frente",
    "image_path": "/tmp/img.jpg",
}

# New (multi-operation):
{
    "inference_id": "uuid",
    "request_id": "uuid",
    "operation": "classify" | "process" | "ocr" | "extract",
    "image_path": "/tmp/img.jpg",
    "document_type": "rg_frente",       # only for extract
    "auto_rotate": true,                 # for classify/process
    "backend": "hybrid",                 # for process (extraction backend)
    "max_pages": 10,                     # for ocr (PDF)
}
```

**Operations:**

| Operation | Models used | Returns |
|-----------|------------|---------|
| `classify` | EfficientNet + EasyOCR (orientation) | `{document_type, confidence, image_correction}` |
| `extract` | Qwen VLM (existing) | `{result: RGData/CNHData/CINData}` |
| `process` | All three (classify → orient → extract) | `{document_type, confidence, result, image_correction}` |
| `ocr` | EasyOCR + orientation | `{pages: [{page, text, confidence}]}` |

**Key optimization**: The `process` operation runs classify + orient + extract in a single roundtrip,
avoiding 3 separate Redis queue hops. This is the main operation for `/process` endpoint.

### 1.2 Inference Server Changes (`inference_server.py`)

**New models to load in `start()`:**

```python
async def start(self):
    # Existing: VLM
    self.extractor = QwenVLExtractor(...)
    self.extractor.load_model()

    # NEW: Classifier
    self.classifier = ClassifierAdapter(
        model_type=settings.classifier_model_type,
        model_path=settings.classifier_model_path,
        device=settings.extractor_device,
    )

    # NEW: EasyOCR engine (shared)
    self.ocr_engine = OCREngine(
        lang=settings.ocr_language,
        use_gpu=True,
        show_log=False,
    )

    # NEW: Orientation corrector (uses shared OCR engine)
    self.orientation_corrector = OrientationCorrector(
        use_text_detection=settings.orientation_enabled,
        device=settings.extractor_device,
        confidence_threshold=settings.orientation_confidence_threshold,
        ocr_engine=self.ocr_engine,
    )
```

**New routing in main loop:**

```python
async def _process_single(self, request: dict):
    operation = request.get("operation", "extract")  # backward compat

    if operation == "classify":
        await self._handle_classify(request)
    elif operation == "process":
        await self._handle_process(request)
    elif operation == "ocr":
        await self._handle_ocr(request)
    else:  # "extract"
        await self._handle_extract(request)  # existing logic
```

**New handlers:**

```python
async def _handle_classify(self, request: dict):
    """Classify document type. Fast (~100ms)."""
    image = Image.open(request["image_path"]).convert("RGB")

    # Orientation correction
    correction = None
    if request.get("auto_rotate", True):
        result = self.orientation_corrector.correct(image)
        if result.was_corrected:
            image = result.image
        correction = result.to_dict()

    # Classification
    classification = self.classifier.classify(image)

    reply = {
        "inference_id": request["inference_id"],
        "success": True,
        "result": {
            "document_type": classification.document_type.value,
            "confidence": classification.confidence,
            "image_correction": correction,
        },
    }
    # ... publish reply

async def _handle_process(self, request: dict):
    """Full pipeline: orient → classify → extract. Single roundtrip."""
    image = Image.open(request["image_path"]).convert("RGB")

    # 1. Orientation correction
    correction = None
    if request.get("auto_rotate", True):
        result = self.orientation_corrector.correct(image)
        if result.was_corrected:
            image = result.image
            # Save corrected image for VLM
            corrected_path = request["image_path"] + ".corrected.jpg"
            image.save(corrected_path)
            request["image_path"] = corrected_path
        correction = result.to_dict()

    # 2. Classification
    classification = self.classifier.classify(image)
    doc_type = classification.document_type.value

    # 3. Extraction (VLM)
    prompt = self._get_prompt(doc_type)
    raw_text = self.extractor._generate(image, prompt)
    extraction = self._build_result(raw_text, doc_type)

    # 4. Hybrid CPF validation (if backend=hybrid)
    if request.get("backend") == "hybrid":
        extraction = self._validate_cpf_with_ocr(image, extraction, doc_type)

    reply = {
        "inference_id": request["inference_id"],
        "success": True,
        "result": {
            "document_type": doc_type,
            "confidence": classification.confidence,
            "extracted_data": extraction,
            "image_correction": correction,
        },
    }

async def _handle_ocr(self, request: dict):
    """Generic OCR. Handles images and multi-page PDFs."""
    file_path = Path(request["image_path"])
    max_pages = request.get("max_pages", 10)

    if is_pdf(file_path):
        images = self.pdf_converter.convert(file_path, max_pages=max_pages)
    else:
        images = [Image.open(file_path)]

    pages = []
    for i, img in enumerate(images, start=1):
        if request.get("auto_rotate", True):
            result = self.orientation_corrector.correct(img)
            img = result.image
        text, confidence = self.ocr_engine.extract_text(img)
        pages.append({"page": i, "text": text, "confidence": confidence})

    reply = {
        "inference_id": request["inference_id"],
        "success": True,
        "result": {"total_pages": len(pages), "pages": pages},
    }
```

### 1.3 Batching Strategy

**Problem:** Currently all requests go to one batch queue. With multiple operation types,
we can't batch classify + extract together (different models).

**Solution:** Separate queues OR smart routing:

**Option A — Single queue, route by type (simpler, recommended):**
- Keep single `queue:doc:inference` queue
- Collect batch, group by operation type
- Process each group sequentially: classify batch → extract batch → ocr batch
- Pro: No queue changes. Con: Can't batch different types.

**Option B — Separate queues per operation:**
- `queue:doc:inference:classify`
- `queue:doc:inference:extract`
- `queue:doc:inference:ocr`
- Server round-robins between queues
- Pro: Clean separation. Con: More complex queue management.

**Recommendation: Option A** — simpler, and `process` (the most common operation)
runs all 3 models sequentially anyway, so batching across types isn't useful.

For the `process` operation, the full pipeline runs synchronously per request:
orient (~100ms) → classify (~100ms) → VLM extract (~3-5s). The VLM is still the bottleneck,
so the added overhead of classification is negligible.

**VLM batching still works:** Multiple `process` or `extract` requests can batch their
VLM inference step. The server can:
1. Collect N `process` requests
2. Run orientation + classification for each (fast, sequential, ~200ms total)
3. Batch all N VLM extractions in one forward pass (existing logic)

### 1.4 Worker DocID Changes (`worker_docid.py`)

Workers become thin:

```python
class DocumentWorker:
    def __init__(self):
        self.queue = QueueService(queue_name=QueueName.DOCID)
        self.delivery = DeliveryService(queue_service=self.queue)
        self._inference_client = InferenceClient(
            queue_service=self.queue,
            timeout=settings.inference_timeout_seconds,
        )
        # NO classifier, NO ocr_engine, NO orientation_corrector
        # NO pipeline instances

    async def _process_job(self, job: JobContext):
        """Send everything to inference server."""
        operation = job.operation  # "classify", "extract", "process"

        result = await self._inference_client.request(
            operation=operation,
            image_path=job.image_path,
            request_id=job.request_id,
            document_type=job.document_type,
            backend=job.extra_params.get("backend", "hybrid"),
            auto_rotate=job.extra_params.get("auto_rotate", True),
        )

        job.result = result
        await self.delivery.deliver(job)
```

**Key simplification:**
- Remove ALL lazy-loaded models: `_vlm_pipeline`, `_ocr_pipeline`, `_hybrid_pipeline`
- Remove `_shared_ocr_engine`, `orientation_corrector`
- Remove `_get_pipeline()`, `_classify()`, `_extract_local()`, `_extract_remote()`
- Remove `_correct_orientation()`
- `_process_job()` becomes ~30 lines instead of ~300

### 1.5 Worker OCR Changes (`worker_ocr.py`)

Same simplification:

```python
class OCRWorker:
    def __init__(self):
        self.queue = QueueService(queue_name=QueueName.OCR)
        self.delivery = DeliveryService(queue_service=self.queue)
        self._inference_client = InferenceClient(...)
        # NO ocr_engine, NO pdf_converter, NO orientation_corrector

    async def _process_job(self, job: JobContext):
        max_pages = job.extra_params.get("max_pages", 10)
        result = await self._inference_client.request(
            operation="ocr",
            image_path=job.image_path,
            request_id=job.request_id,
            max_pages=max_pages,
        )
        job.result = result
        await self.delivery.deliver(job)
```

### 1.6 InferenceClient Changes (`shared/inference_client.py`)

Generalize the client to support all operations:

```python
class InferenceClient:
    async def request(
        self,
        operation: str,      # "classify", "extract", "process", "ocr"
        image_path: str,
        request_id: str,
        document_type: str | None = None,
        backend: str | None = None,
        auto_rotate: bool = True,
        max_pages: int = 10,
    ) -> dict:
        inference_id = str(uuid.uuid4())
        request = {
            "inference_id": inference_id,
            "request_id": request_id,
            "operation": operation,
            "image_path": image_path,
            "document_type": document_type,
            "backend": backend,
            "auto_rotate": auto_rotate,
            "max_pages": max_pages,
        }
        await self._queue.redis.lpush(QueueName.INFERENCE, json.dumps(request))
        return await self._poll_reply(inference_id)
```

### 1.7 Docker/Deployment Changes

**docker-compose.yml:**
- Workers: remove `runtime: nvidia`, remove `NVIDIA_VISIBLE_DEVICES`
- Workers: remove GPU-related env vars (`CLASSIFIER_DEVICE`, `EXTRACTOR_DEVICE`, `ORIENTATION_DEVICE`)
- Inference server: add classifier model path volume mount
- Inference server: add EasyOCR model cache volume

**Dockerfile:**
- Workers no longer need CUDA, torch, torchvision, EasyOCR, doc-classifier
- Create lightweight worker image (just Redis + httpx + pydantic)
- Inference server Dockerfile stays heavy (all models)

### 1.8 Config Changes (`config.py`)

- Remove from worker-relevant settings: `classifier_device`, `orientation_device`, `orientation_use_torch_compile`
- Add to inference server: `classifier_model_path`, `classifier_model_type` (already exist, just used by server now)
- `inference_server_enabled` can be removed (always true — workers always use inference server)

### 1.9 Migration/Backward Compatibility

- Add `operation` field to request format, default to `"extract"` for backward compat
- Deploy new inference server first (understands both old and new format)
- Then deploy new lightweight workers
- Rollback: revert workers to old version (they still have models locally)

### 1.10 Files to Modify

| File | Change |
|------|--------|
| `inference_server.py` | Add classifier, OCR engine, orientation; new handlers; smart batching |
| `worker_docid.py` | Strip to thin queue consumer; remove all model loading |
| `worker_ocr.py` | Strip to thin queue consumer; remove all model loading |
| `doc_pipeline/shared/inference_client.py` | Generalize for multi-operation |
| `doc_pipeline/config.py` | Clean up worker-only settings |
| `docker-compose.yml` | Remove GPU from workers, add model paths to inference server |
| `Dockerfile` | Create lightweight worker image variant |
| `requirements.txt` | Split into `requirements-worker.txt` (light) + `requirements-inference.txt` (heavy) |

---

## Phase 2: vLLM Migration (Optional, Higher Throughput)

### 2.1 Why vLLM?

| Feature | Current (HuggingFace) | vLLM |
|---------|----------------------|------|
| Batching | Fixed batch (collect N, process) | Continuous batching (process as they arrive) |
| Memory | Full KV cache pre-allocated | PagedAttention (dynamic, less waste) |
| Throughput | ~1.7s/item in batch | ~0.8-1.2s/item (2x faster) |
| API | Custom Redis protocol | OpenAI-compatible HTTP API |
| Quantization | Manual | Built-in AWQ, GPTQ, FP8 |

### 2.2 Architecture with vLLM

```
Inference Server (our code, ~3GB VRAM)
  ├── EfficientNet classifier
  ├── EasyOCR engine
  ├── OrientationCorrector
  └── HTTP client → vLLM server

vLLM Server (separate process, ~10-12GB VRAM)
  └── Qwen2.5-VL with continuous batching
      OpenAI-compatible API at localhost:8000
```

**Key change:** Our inference server no longer loads the VLM directly.
It sends HTTP requests to vLLM's OpenAI API:

```python
# Instead of:
raw_text = self.extractor._generate(image, prompt)

# We do:
import httpx

response = await httpx.AsyncClient().post(
    "http://vllm:8000/v1/chat/completions",
    json={
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "messages": [
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64img}"}},
                {"type": "text", "text": prompt},
            ]}
        ],
        "max_tokens": 1024,
    }
)
raw_text = response.json()["choices"][0]["message"]["content"]
```

### 2.3 vLLM Docker Service

```yaml
# docker-compose.yml
vllm:
  image: vllm/vllm-openai:latest
  runtime: nvidia
  environment:
    - NVIDIA_VISIBLE_DEVICES=all
  command: >
    --model Qwen/Qwen2.5-VL-3B-Instruct
    --dtype auto
    --max-model-len 4096
    --gpu-memory-utilization 0.85
    --port 8000
  volumes:
    - huggingface-cache:/root/.cache/huggingface
  ports:
    - "8000:8000"
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
```

### 2.4 Benefits for 5090 (32GB)

With vLLM + centralized inference:
- vLLM (Qwen2.5-VL-3B): ~8GB (with PagedAttention, less than HF)
- Inference server (EfficientNet + EasyOCR): ~3GB
- **Total: ~11GB** — leaves 21GB free
- Could run Qwen2.5-VL-7B (~14GB) for better accuracy and still fit

### 2.5 vLLM Files to Add/Modify

| File | Change |
|------|--------|
| `inference_server.py` | Replace `QwenVLExtractor` with HTTP client to vLLM |
| `docker-compose.yml` | Add `vllm` service |
| `doc_pipeline/extractors/vllm_client.py` | NEW: vLLM OpenAI API client |
| `doc_pipeline/config.py` | Add `vllm_base_url`, `vllm_model` settings |

---

## Implementation Order

### Step 1: Phase 1 — Centralize models (main effort)
1. Modify `inference_server.py` to load classifier + EasyOCR + orientation
2. Add new operation handlers (`classify`, `process`, `ocr`)
3. Generalize `InferenceClient` for multi-operation
4. Simplify `worker_docid.py` to thin consumer
5. Simplify `worker_ocr.py` to thin consumer
6. Update docker-compose (remove GPU from workers)
7. Test locally with 1 worker

### Step 2: Phase 2 — vLLM migration (optional)
1. Add vLLM service to docker-compose
2. Create `vllm_client.py` with OpenAI API calls
3. Replace `QwenVLExtractor` usage in inference server
4. Benchmark: vLLM vs current HF throughput
5. Test with Qwen2.5-VL-7B if VRAM allows

### Step 3: Cleanup
1. Remove unused extractor code from workers
2. Split requirements into light (worker) and heavy (inference)
3. Create separate Dockerfile for lightweight workers
4. Update CLAUDE.md with new architecture
5. Update Grafana dashboards (workers no longer have GPU metrics)

---

## VRAM Estimates

| Setup | VRAM |
|-------|------|
| Phase 1 (HF, centralized) | ~15GB |
| Phase 2 (vLLM 3B, centralized) | ~11GB |
| Phase 2 (vLLM 7B, centralized) | ~17GB |
| Current (distributed) | ~48GB |

## GPU Compatibility

**Development**: RTX 5090 (Blackwell, 32GB, sm_100, CUDA 12.8+)
**Production**: NVIDIA H200 (Hopper, 141GB, sm_90, CUDA 12.0+)

**Rules:**
- Do NOT use Blackwell-only features (FP4, etc.)
- Use `dtype=auto` or `dtype=float16` — works on both architectures
- vLLM: use `--dtype auto` (auto-selects bf16 on both Hopper and Blackwell)
- PyTorch: ensure version supports both sm_90 (Hopper) and sm_100 (Blackwell) — PyTorch 2.2+
- EasyOCR/EfficientNet: FP32 by default, no compatibility issues
- Docker base image: use `nvidia/cuda:12.4.0-runtime-ubuntu22.04` or later (supports both)
- Do NOT pin CUDA to 12.8+ (Hopper needs 12.0+)
- Test on 5090 with `CUDA_VISIBLE_DEVICES=0`, deploy to H200 with same images
- MPS is enabled on H200 production (multiple services share GPU); not needed for 5090 dev

## Risk Mitigation

- **Single point of failure**: If inference server goes down, all processing stops.
  Mitigation: health checks + auto-restart + alerting (already configured).
- **Latency increase for classification**: Adding Redis roundtrip (~1ms) to classification.
  Negligible vs the 100ms classify time.
- **Rollback plan**: Keep old worker code in a branch. Workers can be reverted independently.
- **Backward compatibility**: New inference server handles old request format (no `operation` field = `extract`).
