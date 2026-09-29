"""MachineEcho — NPU / CPU Benchmark.

Measures YAMNet inference latency on different ONNX Runtime
execution providers to quantify the Snapdragon NPU advantage.

Usage:
    python benchmark.py
    python benchmark.py --runs 200
"""
from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from config import YAMNET_ONNX_PATH, MEL_BANDS


def benchmark_provider(
    model_path: Path, provider: str, n_runs: int = 100
) -> dict | None:
    """Run inference N times with a given provider and collect timing."""
    if ort is None:
        print("  onnxruntime not installed.")
        return None

    try:
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(model_path), sess_options=sess_opts, providers=[provider]
        )
    except Exception as exc:
        print(f"  {provider}: unavailable ({exc})")
        return None

    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    # Create dummy input matching expected shape
    # Handle dynamic dimensions
    resolved_shape = []
    for dim in input_shape:
        if isinstance(dim, int) and dim > 0:
            resolved_shape.append(dim)
        else:
            resolved_shape.append(96 if len(resolved_shape) == 1 else 64 if len(resolved_shape) == 2 else 1)
    dummy_input = np.random.randn(*resolved_shape).astype(np.float32)

    # Warm up
    for _ in range(5):
        session.run(None, {input_name: dummy_input})

    # Timed runs
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_input})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    latencies_arr = np.array(latencies)
    return {
        "provider": provider,
        "runs": n_runs,
        "mean_ms": float(np.mean(latencies_arr)),
        "std_ms": float(np.std(latencies_arr)),
        "min_ms": float(np.min(latencies_arr)),
        "max_ms": float(np.max(latencies_arr)),
        "p50_ms": float(np.percentile(latencies_arr, 50)),
        "p95_ms": float(np.percentile(latencies_arr, 95)),
        "p99_ms": float(np.percentile(latencies_arr, 99)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark YAMNet inference.")
    parser.add_argument("--runs", type=int, default=100, help="Number of inference runs.")
    parser.add_argument("--model", type=str, default=str(YAMNET_ONNX_PATH), help="ONNX model path.")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\n[ERROR] Model not found: {model_path}")
        print("  Download yamnet.onnx from Qualcomm AI Hub and place it in models/")
        sys.exit(1)

    print(f"\n{'='*65}")
    print("  MachineEcho -- Inference Benchmark")
    print(f"{'='*65}")
    print(f"  Model : {model_path.name}")
    print(f"  Runs  : {args.runs}")

    if ort:
        print(f"  ONNX Runtime version: {ort.__version__}")
        print(f"  Available providers : {ort.get_available_providers()}")
    print()

    providers_to_test = [
        "QNNExecutionProvider",
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]

    results = []
    for provider in providers_to_test:
        print(f"  Testing {provider} ...")
        result = benchmark_provider(model_path, provider, args.runs)
        if result:
            results.append(result)
            print(f"    Mean: {result['mean_ms']:.3f} ms")
            print(f"    Std:  {result['std_ms']:.3f} ms")
            print(f"    P50:  {result['p50_ms']:.3f} ms")
            print(f"    P95:  {result['p95_ms']:.3f} ms")
            print(f"    Min:  {result['min_ms']:.3f} ms")
            print(f"    Max:  {result['max_ms']:.3f} ms")
        print()

    if not results:
        print("  No providers could run the model.")
        return

    # Summary table
    print(f"  {'Provider':<30} {'Mean (ms)':>10} {'P95 (ms)':>10} {'Speedup':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10}")
    baseline = results[-1]["mean_ms"]  # CPU as baseline
    for r in results:
        speedup = baseline / r["mean_ms"] if r["mean_ms"] > 0 else 0
        print(
            f"  {r['provider']:<30} {r['mean_ms']:>10.3f} {r['p95_ms']:>10.3f} {speedup:>9.1f}x"
        )

    print(f"\n{'='*65}\n")


if __name__ == "__main__":
    main()
