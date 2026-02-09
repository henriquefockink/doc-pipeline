#!/usr/bin/env python3
"""
Benchmark script for doc-pipeline.

Runs the same 5 scenarios from BENCHMARKS.md and collects:
- Latency percentiles (P50, P95, P99, mean)
- Throughput (rps)
- Error rate
- Resource usage: VRAM, Redis memory, container memory
"""

import asyncio
import json
import statistics
import subprocess
import sys
import time

import httpx

API_URL = "http://localhost:9000/process"
TEST_FILE = "test_files/cnh_noventa_graus.jpg"

SCENARIOS = [
    {"name": "Single Request", "requests": 10, "concurrency": 1, "rps": 1},
    {"name": "Low Concurrency", "requests": 30, "concurrency": 5, "rps": 2},
    {"name": "Medium Concurrency", "requests": 50, "concurrency": 10, "rps": 5},
    {"name": "High Concurrency", "requests": 50, "concurrency": 20, "rps": 10},
    {"name": "Stress Test", "requests": 60, "concurrency": 30, "rps": 15},
]


def get_resource_usage() -> dict:
    """Collect resource usage from GPU, Redis, and Docker containers."""
    info = {}

    # GPU VRAM
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        used, free, util = out.split(", ")
        info["vram_used_mb"] = int(used)
        info["vram_free_mb"] = int(free)
        info["gpu_util_pct"] = int(util)
    except Exception as e:
        info["vram_error"] = str(e)

    # Redis memory
    try:
        out = subprocess.check_output(
            ["docker", "exec", "doc-pipeline-redis-1", "redis-cli", "INFO", "memory"],
            text=True,
        )
        for line in out.splitlines():
            if line.startswith("used_memory_human:"):
                info["redis_memory"] = line.split(":")[1].strip()
            elif line.startswith("used_memory:"):
                info["redis_memory_bytes"] = int(line.split(":")[1].strip())
    except Exception as e:
        info["redis_error"] = str(e)

    # Redis connected clients
    try:
        out = subprocess.check_output(
            ["docker", "exec", "doc-pipeline-redis-1", "redis-cli", "INFO", "clients"],
            text=True,
        )
        for line in out.splitlines():
            if line.startswith("connected_clients:"):
                info["redis_clients"] = int(line.split(":")[1].strip())
    except Exception as e:
        pass

    # Inference queue depth
    try:
        out = subprocess.check_output(
            ["docker", "exec", "doc-pipeline-redis-1", "redis-cli", "LLEN", "queue:doc:inference"],
            text=True,
        ).strip()
        info["inference_queue_depth"] = int(out)
    except Exception:
        pass

    # Container memory: vllm
    try:
        out = subprocess.check_output(
            ["docker", "stats", "doc-pipeline-vllm", "--no-stream", "--format",
             "{{.MemUsage}}"],
            text=True,
        ).strip()
        info["vllm_container_mem"] = out
    except Exception:
        pass

    # Container memory: inference-server
    try:
        out = subprocess.check_output(
            ["docker", "stats", "doc-pipeline-inference-server", "--no-stream", "--format",
             "{{.MemUsage}}"],
            text=True,
        ).strip()
        info["inference_server_mem"] = out
    except Exception:
        pass

    return info


def percentile(data: list[float], p: float) -> float:
    """Calculate percentile."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


async def send_request(client: httpx.AsyncClient, file_path: str) -> tuple[bool, float]:
    """Send a single /process request. Returns (success, latency_ms)."""
    start = time.perf_counter()
    try:
        with open(file_path, "rb") as f:
            files = {"arquivo": ("test.jpg", f, "image/jpeg")}
            resp = await client.post(API_URL, files=files)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if resp.status_code == 200:
            data = resp.json()
            return data.get("success", False), elapsed_ms
        return False, elapsed_ms
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return False, elapsed_ms


async def run_scenario(scenario: dict) -> dict:
    """Run a single benchmark scenario."""
    name = scenario["name"]
    total = scenario["requests"]
    concurrency = scenario["concurrency"]
    rps_limit = scenario["rps"]

    print(f"\n{'='*60}")
    print(f"  {name}: {total} requests, concurrency={concurrency}, rps={rps_limit}")
    print(f"{'='*60}")

    # Pre-test resources
    pre_resources = get_resource_usage()

    latencies: list[float] = []
    ok_count = 0
    fail_count = 0
    sem = asyncio.Semaphore(concurrency)
    interval = 1.0 / rps_limit if rps_limit > 0 else 0

    # Peak resource tracking
    peak_vram = pre_resources.get("vram_used_mb", 0)
    peak_queue = 0

    async with httpx.AsyncClient(timeout=httpx.Timeout(180.0, connect=10.0)) as client:

        async def worker(idx: int):
            nonlocal ok_count, fail_count, peak_vram, peak_queue
            async with sem:
                success, latency = await send_request(client, TEST_FILE)
                latencies.append(latency)
                if success:
                    ok_count += 1
                else:
                    fail_count += 1
                # Progress
                done = ok_count + fail_count
                if done % 5 == 0 or done == total:
                    print(f"  Progress: {done}/{total}  (last: {latency:.0f}ms)")

        start_time = time.perf_counter()

        tasks = []
        for i in range(total):
            if i > 0 and interval > 0:
                await asyncio.sleep(interval)
            tasks.append(asyncio.create_task(worker(i)))

        await asyncio.gather(*tasks)

        total_time = time.perf_counter() - start_time

    # Post-test resources
    post_resources = get_resource_usage()

    # Calculate stats
    result = {
        "name": name,
        "requests": total,
        "concurrency": concurrency,
        "rps_limit": rps_limit,
        "ok": ok_count,
        "fail": fail_count,
        "latency_mean_ms": round(statistics.mean(latencies), 0) if latencies else 0,
        "latency_p50_ms": round(percentile(latencies, 50), 0),
        "latency_p95_ms": round(percentile(latencies, 95), 0),
        "latency_p99_ms": round(percentile(latencies, 99), 0),
        "latency_min_ms": round(min(latencies), 0) if latencies else 0,
        "latency_max_ms": round(max(latencies), 0) if latencies else 0,
        "throughput_rps": round(ok_count / total_time, 2) if total_time > 0 else 0,
        "total_time_s": round(total_time, 1),
        "resources_pre": pre_resources,
        "resources_post": post_resources,
    }

    # Print results
    print(f"\n  Results:")
    print(f"    OK/Fail: {ok_count}/{fail_count}")
    print(f"    Latency mean: {result['latency_mean_ms']:.0f} ms")
    print(f"    P50: {result['latency_p50_ms']:.0f} ms")
    print(f"    P95: {result['latency_p95_ms']:.0f} ms")
    print(f"    P99: {result['latency_p99_ms']:.0f} ms")
    print(f"    Throughput: {result['throughput_rps']:.2f} rps")
    print(f"    Total time: {result['total_time_s']:.1f}s")
    print(f"    VRAM: {post_resources.get('vram_used_mb', '?')} MB")
    print(f"    Redis: {post_resources.get('redis_memory', '?')}")
    print(f"    vLLM container: {post_resources.get('vllm_container_mem', '?')}")
    print(f"    Inference server: {post_resources.get('inference_server_mem', '?')}")
    print(f"    Queue depth: {post_resources.get('inference_queue_depth', '?')}")

    return result


async def main():
    print("Doc-Pipeline Benchmark — vLLM (continuous batching)")
    print(f"API: {API_URL}")
    print(f"Test file: {TEST_FILE}")

    # Check API health
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get("http://localhost:9000/health")
            print(f"API health: {resp.status_code}")
        except Exception as e:
            print(f"API not reachable: {e}")
            sys.exit(1)

        try:
            resp = await client.get("http://localhost:9020/health")
            health = resp.json()
            print(f"Inference server: {health.get('vlm_backend', '?')} backend")
        except Exception as e:
            print(f"Inference server not reachable: {e}")
            sys.exit(1)

    # Idle resources
    print("\n--- Idle Resource Usage ---")
    idle = get_resource_usage()
    print(f"  VRAM: {idle.get('vram_used_mb', '?')} MB")
    print(f"  Redis: {idle.get('redis_memory', '?')}")
    print(f"  Redis clients: {idle.get('redis_clients', '?')}")
    print(f"  vLLM container: {idle.get('vllm_container_mem', '?')}")
    print(f"  Inference server: {idle.get('inference_server_mem', '?')}")

    results = []
    for scenario in SCENARIOS:
        result = await run_scenario(scenario)
        results.append(result)
        # Cool down between scenarios
        print("\n  Cooling down 5s...")
        await asyncio.sleep(5)

    # Final summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Scenario':<20} {'Conc':>5} {'P50':>8} {'P95':>8} {'RPS':>7} {'Err':>5}")
    print(f"{'-'*20} {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*5}")
    for r in results:
        print(
            f"{r['name']:<20} {r['concurrency']:>5} "
            f"{r['latency_p50_ms']:>7.0f}ms {r['latency_p95_ms']:>7.0f}ms "
            f"{r['throughput_rps']:>6.2f} {r['fail']:>5}"
        )

    # Save JSON
    with open("benchmark_results_vllm.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to benchmark_results_vllm.json")


if __name__ == "__main__":
    asyncio.run(main())
