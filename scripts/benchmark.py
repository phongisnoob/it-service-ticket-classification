#!/usr/bin/env python3
"""Reproducible benchmark for the IT ticket classification API.

Usage:
    python scripts/benchmark.py --url http://localhost:8000 --concurrency 4 --requests 200

Measures warm-up, throughput, error count, p50/p95/p99 latency, and process RSS.
Output is saved to reports/private/benchmark_<backend>_<timestamp>.json
(excluded from Git).
"""

import argparse
import concurrent.futures
import datetime
import json
import os
import statistics
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    print("httpx not installed. Run: pip install httpx", file=sys.stderr)
    sys.exit(1)

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

SAMPLE_TEXTS = [
    "I cannot access the shared network drive",
    "My laptop keyboard is not working properly",
    "Password reset request for user jsmith",
    "VPN connection drops after 10 minutes",
    "Need to install software on workstation",
    "Printer in room 3B is out of paper",
    "Email is not syncing on mobile device",
    "Request for new laptop for new hire",
    "Database server is running slow",
    "Cannot log in to the HR portal",
]


def single_request(client: httpx.Client, url: str, text: str) -> dict[str, object]:
    start = time.perf_counter()
    try:
        resp = client.post(f"{url}/predict", json={"text": text}, timeout=10.0)
        elapsed = time.perf_counter() - start
        return {"ok": resp.status_code == 200, "latency": elapsed, "status": resp.status_code}
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"ok": False, "latency": elapsed, "error": str(e), "status": 0}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * p / 100)
    return sorted_vals[min(idx, len(sorted_vals) - 1)]


def run_benchmark(
    url: str,
    concurrency: int,
    n_requests: int,
    warmup: int,
) -> dict[str, object]:
    # Warm-up
    print(f"Warming up with {warmup} requests...")
    with httpx.Client() as client:
        for i in range(warmup):
            text = SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)]
            single_request(client, url, text)

    # /health to get backend metadata
    with httpx.Client() as client:
        health = client.get(f"{url}/health", timeout=5.0).json()
    backend = health.get("model_backend", "unknown")
    model_sha256 = health.get("model_sha256", "unknown")

    print(f"Backend: {backend}, SHA256: {str(model_sha256)[:12]}...")
    print(f"Running {n_requests} requests with concurrency={concurrency}...")

    latencies: list[float] = []
    errors = 0
    start_total = time.perf_counter()

    with httpx.Client() as client:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    single_request,
                    client,
                    url,
                    SAMPLE_TEXTS[i % len(SAMPLE_TEXTS)],
                )
                for i in range(n_requests)
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                latencies.append(float(result["latency"]))
                if not result["ok"]:
                    errors += 1

    total_time = time.perf_counter() - start_total
    throughput = n_requests / total_time if total_time > 0 else 0

    # Process RSS
    rss_mb: float | None = None
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        rss_mb = process.memory_info().rss / (1024 * 1024)

    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "url": url,
        "backend": backend,
        "model_sha256": model_sha256,
        "n_requests": n_requests,
        "concurrency": concurrency,
        "warmup_requests": warmup,
        "errors": errors,
        "total_time_s": round(total_time, 3),
        "throughput_rps": round(throughput, 2),
        "latency_p50_ms": round(percentile(latencies, 50) * 1000, 2),
        "latency_p95_ms": round(percentile(latencies, 95) * 1000, 2),
        "latency_p99_ms": round(percentile(latencies, 99) * 1000, 2),
        "latency_mean_ms": round(statistics.mean(latencies) * 1000, 2) if latencies else 0,
        "process_rss_mb": round(rss_mb, 2) if rss_mb is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the IT ticket classification API")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent workers")
    parser.add_argument("--requests", type=int, default=200, help="Total number of requests")
    parser.add_argument("--warmup", type=int, default=20, help="Number of warm-up requests")
    parser.add_argument("--output", help="JSON file to save results (default: reports/private/)")
    args = parser.parse_args()

    results = run_benchmark(
        url=args.url,
        concurrency=args.concurrency,
        n_requests=args.requests,
        warmup=args.warmup,
    )

    print("\n=== Benchmark Results ===")
    for k, v in results.items():
        print(f"  {k:30s}: {v}")

    if args.output:
        output_path = Path(args.output)
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backend = str(results.get("backend", "unknown"))
        output_path = Path("reports/private") / f"benchmark_{backend}_{ts}.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
