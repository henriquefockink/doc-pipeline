#!/usr/bin/env python3
"""Benchmark script for doc-pipeline /process endpoint."""

import asyncio
import statistics
import sys
import time

import httpx

API_URL = "http://localhost:9000/process"
TEST_IMAGE = "test_files/90.jpg"

SCENARIOS = [
    {"name": "Single", "requests": 10, "concurrency": 1},
    {"name": "Low", "requests": 30, "concurrency": 5},
    {"name": "Medium", "requests": 50, "concurrency": 10},
    {"name": "High", "requests": 50, "concurrency": 20},
    {"name": "Stress", "requests": 60, "concurrency": 30},
]


async def send_request(client: httpx.AsyncClient, image_path: str) -> tuple[float, bool]:
    """Send a single /process request, return (latency_ms, success)."""
    start = time.perf_counter()
    try:
        with open(image_path, "rb") as f:
            resp = await client.post(
                API_URL,
                files={"image": ("test.jpg", f, "image/jpeg")},
                data={"delivery_mode": "sync"},
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        success = resp.status_code == 200
        if not success:
            print(f"  FAIL status={resp.status_code} body={resp.text[:200]}")
        return elapsed_ms, success
    except Exception as e:
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"  ERROR: {e}")
        return elapsed_ms, False


async def run_scenario(name: str, total: int, concurrency: int) -> dict:
    """Run a benchmark scenario with controlled concurrency."""
    sem = asyncio.Semaphore(concurrency)
    latencies = []
    ok = 0
    fail = 0

    async def bounded_request(client):
        nonlocal ok, fail
        async with sem:
            lat, success = await send_request(client, TEST_IMAGE)
            latencies.append(lat)
            if success:
                ok += 1
            else:
                fail += 1

    print(f"\n{'='*60}")
    print(f"  {name}: {total} requests, concurrency={concurrency}")
    print(f"{'='*60}")

    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        # Warmup: 1 request
        print("  Warmup...")
        await send_request(client, TEST_IMAGE)

        # Run
        print(f"  Running {total} requests...")
        tasks = [bounded_request(client) for _ in range(total)]
        await asyncio.gather(*tasks)

    wall_time = time.perf_counter() - start
    latencies.sort()

    def percentile(data, p):
        k = (len(data) - 1) * p / 100
        f = int(k)
        c = f + 1
        if c >= len(data):
            return data[f]
        return data[f] + (k - f) * (data[c] - data[f])

    result = {
        "name": name,
        "requests": total,
        "concurrency": concurrency,
        "ok": ok,
        "fail": fail,
        "mean": statistics.mean(latencies),
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "rps": ok / wall_time if wall_time > 0 else 0,
    }

    print(f"  OK: {ok} | Fail: {fail}")
    print(f"  Mean:  {result['mean']:,.0f} ms")
    print(f"  P50:   {result['p50']:,.0f} ms")
    print(f"  P95:   {result['p95']:,.0f} ms")
    print(f"  P99:   {result['p99']:,.0f} ms")
    print(f"  RPS:   {result['rps']:.2f}")

    return result


async def main():
    # Check API health
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            resp = await client.get("http://localhost:9000/health")
            print(f"API health: {resp.status_code}")
        except Exception as e:
            print(f"API not reachable: {e}")
            sys.exit(1)

        try:
            resp = await client.get("http://localhost:9020/health")
            health = resp.json()
            print(f"Inference: vlm_backend={health.get('vlm_backend')}")
        except Exception as e:
            print(f"Inference server not reachable: {e}")
            sys.exit(1)

    results = []
    for scenario in SCENARIOS:
        r = await run_scenario(scenario["name"], scenario["requests"], scenario["concurrency"])
        results.append(r)
        # Cool down between scenarios
        await asyncio.sleep(3)

    # Summary table
    print(f"\n{'='*60}")
    print("  SUMMARY — vLLM Embedded (H200)")
    print(f"{'='*60}")
    print(f"{'Scenario':<10} {'Conc':>5} {'P50':>10} {'P95':>10} {'P99':>10} {'RPS':>8} {'OK/Fail':>10}")
    print("-" * 60)
    for r in results:
        print(
            f"{r['name']:<10} {r['concurrency']:>5} "
            f"{r['p50']/1000:>9.1f}s {r['p95']/1000:>9.1f}s {r['p99']/1000:>9.1f}s "
            f"{r['rps']:>7.2f} {r['ok']:>4}/{r['fail']:<4}"
        )


if __name__ == "__main__":
    asyncio.run(main())
