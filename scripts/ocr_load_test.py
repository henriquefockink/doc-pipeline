#!/usr/bin/env python3
"""
Teste de carga simples para OCR via upload multipart.

Exemplo:
  python3 scripts/ocr_load_test.py \
    --file "/path/to/documento.jpg" \
    --requests 100 \
    --concurrency 10 \
    --rps 5 \
    --per-request \
    --print-response \
    --api-key "$OCR_API_KEY"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx


DEFAULT_URL = "https://ocr.paneas.com/process"
DEFAULT_TIMEOUT_S = 800.0
DEFAULT_RPS = 5.0
DEFAULT_CONCURRENCY = 10
DEFAULT_REQUESTS = 100
DEFAULT_MIN_CONFIDENCE = 0.9


@dataclass(frozen=True)
class RequestResult:
    index: int
    ok: bool
    status_code: Optional[int]
    latency_s: float
    error: Optional[str]
    response_text: Optional[str]


def _mask_secret(secret: str, keep: int = 4) -> str:
    if not secret:
        return ""
    if len(secret) <= keep * 2:
        return "*" * len(secret)
    return f"{secret[:keep]}{'*' * (len(secret) - keep * 2)}{secret[-keep:]}"


def _build_url(base_url: str, extract: bool, min_confidence: float) -> str:
    parts = urlsplit(base_url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.setdefault("extract", "true" if extract else "false")
    query.setdefault("min_confidence", str(min_confidence))
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _percentile_ms(values_ms: list[float], percentile: float) -> float:
    if not values_ms:
        return 0.0
    if percentile <= 0:
        return min(values_ms)
    if percentile >= 100:
        return max(values_ms)
    values_ms_sorted = sorted(values_ms)
    k = (len(values_ms_sorted) - 1) * (percentile / 100.0)
    f = int(k)
    c = min(f + 1, len(values_ms_sorted) - 1)
    if f == c:
        return values_ms_sorted[f]
    d0 = values_ms_sorted[f] * (c - k)
    d1 = values_ms_sorted[c] * (k - f)
    return d0 + d1


async def _run_load_test(
    *,
    url: str,
    api_key: str,
    file_path: str,
    field_name: str,
    requests_total: int,
    concurrency: int,
    rps: float,
    timeout_s: float,
    verify_tls: bool,
    per_request: bool,
    print_response: bool,
    max_response_chars: int,
    preload_file: bool,
) -> list[RequestResult]:
    if requests_total <= 0:
        raise ValueError("--requests precisa ser > 0")
    if concurrency <= 0:
        raise ValueError("--concurrency precisa ser > 0")
    if rps <= 0:
        raise ValueError("--rps precisa ser > 0")

    file_name = os.path.basename(file_path)
    mime_type, _ = mimetypes.guess_type(file_path)
    mime_type = mime_type or "application/octet-stream"

    file_bytes: Optional[bytes] = None
    if preload_file:
        with open(file_path, "rb") as f:
            file_bytes = f.read()

    headers: dict[str, str] = {
        "accept": "application/json",
        "user-agent": "ocr-load-test/1.0 (+python httpx)",
        "referer": "http://localhost:8001/docs",
        "x-api-key": api_key,
    }

    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )
    timeout = httpx.Timeout(timeout_s)

    results: list[Optional[RequestResult]] = [None] * requests_total
    queue: asyncio.Queue[Optional[int]] = asyncio.Queue()

    async with httpx.AsyncClient(
        headers=headers,
        limits=limits,
        timeout=timeout,
        verify=verify_tls,
        follow_redirects=True,
    ) as client:

        async def producer() -> None:
            start = time.perf_counter()
            for i in range(requests_total):
                scheduled = start + (i / rps)
                delay = scheduled - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                await queue.put(i)
            for _ in range(concurrency):
                await queue.put(None)

        async def worker(worker_id: int) -> None:
            while True:
                index = await queue.get()
                if index is None:
                    return

                t0 = time.perf_counter()
                try:
                    if file_bytes is not None:
                        files = {field_name: (file_name, file_bytes, mime_type)}
                        response = await client.post(url, files=files)
                    else:
                        with open(file_path, "rb") as f:
                            files = {field_name: (file_name, f, mime_type)}
                            response = await client.post(url, files=files)

                    latency_s = time.perf_counter() - t0
                    response_text: Optional[str] = None
                    if print_response:
                        text = response.text
                        if max_response_chars > 0 and len(text) > max_response_chars:
                            text = text[:max_response_chars] + "…"
                        response_text = text

                    ok = 200 <= response.status_code < 300
                    result = RequestResult(
                        index=index,
                        ok=ok,
                        status_code=response.status_code,
                        latency_s=latency_s,
                        error=None if ok else f"HTTP {response.status_code}",
                        response_text=response_text,
                    )
                except Exception as e:  # noqa: BLE001
                    latency_s = time.perf_counter() - t0
                    result = RequestResult(
                        index=index,
                        ok=False,
                        status_code=None,
                        latency_s=latency_s,
                        error=f"{type(e).__name__}: {e}",
                        response_text=None,
                    )

                results[index] = result
                if per_request:
                    status = result.status_code if result.status_code is not None else "ERR"
                    ms = result.latency_s * 1000.0
                    msg = f"[{index+1:>4}/{requests_total}] worker={worker_id:>2} status={status} {ms:>8.1f}ms"
                    if result.error and (result.status_code is None or result.status_code >= 400):
                        msg += f" error={result.error}"
                    print(msg, flush=True)
                if result.response_text is not None:
                    print(f"--- response[{index+1}] ---\n{result.response_text}\n", flush=True)

        tasks = [asyncio.create_task(worker(i + 1)) for i in range(concurrency)]
        producer_task = asyncio.create_task(producer())

        await producer_task
        await asyncio.gather(*tasks)

    final_results: list[RequestResult] = []
    for item in results:
        if item is None:
            raise RuntimeError("Resultado faltando (bug no runner).")
        final_results.append(item)
    return final_results


def _print_summary(results: list[RequestResult], started_s: float, finished_s: float) -> None:
    duration_s = max(0.000001, finished_s - started_s)
    ok_results = [r for r in results if r.ok]
    fail_results = [r for r in results if not r.ok]

    lat_ms_all = [r.latency_s * 1000.0 for r in results]
    lat_ms_ok = [r.latency_s * 1000.0 for r in ok_results]

    status_counts: dict[str, int] = {}
    for r in results:
        key = str(r.status_code) if r.status_code is not None else "ERR"
        status_counts[key] = status_counts.get(key, 0) + 1

    print("\n=== Summary ===")
    print(f"total={len(results)} ok={len(ok_results)} fail={len(fail_results)}")
    print(f"duration={duration_s:.2f}s achieved_rps={len(results)/duration_s:.2f}")
    print(f"status_counts={json.dumps(status_counts, ensure_ascii=False, sort_keys=True)}")

    def show_lat(label: str, values: list[float]) -> None:
        if not values:
            print(f"{label}: n=0")
            return
        mean = statistics.fmean(values)
        p50 = _percentile_ms(values, 50)
        p95 = _percentile_ms(values, 95)
        p99 = _percentile_ms(values, 99)
        print(
            f"{label}: n={len(values)} min={min(values):.1f}ms mean={mean:.1f}ms p50={p50:.1f}ms p95={p95:.1f}ms p99={p99:.1f}ms max={max(values):.1f}ms"
        )

    show_lat("latency_all", lat_ms_all)
    show_lat("latency_ok", lat_ms_ok)

    if fail_results:
        # Mostra até 10 erros únicos
        err_counts: dict[str, int] = {}
        for r in fail_results:
            err = r.error or "unknown"
            err_counts[err] = err_counts.get(err, 0) + 1
        top = sorted(err_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        print("top_errors=" + json.dumps(top, ensure_ascii=False))


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OCR load test (multipart upload) with RPS limiter.")
    parser.add_argument("--file", required=True, help="Caminho do arquivo a enviar.")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"URL base (default: {DEFAULT_URL}).")
    parser.add_argument("--field", default="arquivo", help="Nome do campo multipart (default: arquivo).")
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS, help="Total de requisições.")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Requisições em paralelo.")
    parser.add_argument("--rps", type=float, default=DEFAULT_RPS, help="Requisições por segundo (rate limit).")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S, help="Timeout por requisição (s).")
    parser.add_argument("--api-key", default=os.getenv("OCR_API_KEY", ""), help="X-API-Key (ou env OCR_API_KEY).")
    parser.add_argument(
        "--extract",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Param extract=true/false (default: true).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help=f"Param min_confidence (default: {DEFAULT_MIN_CONFIDENCE}).",
    )
    parser.add_argument(
        "--verify-tls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verifica TLS (default: true). Use --no-verify-tls para desabilitar.",
    )
    parser.add_argument("--per-request", action="store_true", help="Imprime uma linha por requisição.")
    parser.add_argument("--print-response", action="store_true", help="Imprime o corpo da resposta (truncado).")
    parser.add_argument(
        "--max-response-chars",
        type=int,
        default=1000,
        help="Máximo de caracteres do body ao usar --print-response (default: 1000).",
    )
    parser.add_argument(
        "--preload-file",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Carrega o arquivo em memória e reutiliza (default: true).",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)

    if not os.path.exists(args.file):
        print(f"Arquivo não encontrado: {args.file}", file=sys.stderr)
        return 2
    if not args.api_key:
        print(
            "API key não informada. Use --api-key ou defina OCR_API_KEY no ambiente.",
            file=sys.stderr,
        )
        return 2

    url = _build_url(args.url, extract=args.extract, min_confidence=args.min_confidence)

    print("=== OCR Load Test ===")
    print(f"url={url}")
    print(f"file={args.file}")
    print(f"field={args.field}")
    print(f"requests={args.requests} concurrency={args.concurrency} rps={args.rps}")
    print(f"timeout={args.timeout}s verify_tls={args.verify_tls} preload_file={args.preload_file}")
    print(f"api_key={_mask_secret(args.api_key)}")
    if args.print_response:
        print(f"print_response=true max_response_chars={args.max_response_chars}")
    if args.per_request:
        print("per_request=true")

    started = time.perf_counter()
    try:
        results = asyncio.run(
            _run_load_test(
                url=url,
                api_key=args.api_key,
                file_path=args.file,
                field_name=args.field,
                requests_total=args.requests,
                concurrency=args.concurrency,
                rps=args.rps,
                timeout_s=args.timeout,
                verify_tls=args.verify_tls,
                per_request=args.per_request,
                print_response=args.print_response,
                max_response_chars=args.max_response_chars,
                preload_file=args.preload_file,
            )
        )
    except KeyboardInterrupt:
        print("\nInterrompido.", file=sys.stderr)
        return 130
    finished = time.perf_counter()

    _print_summary(results, started_s=started, finished_s=finished)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
