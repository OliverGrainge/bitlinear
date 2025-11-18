import argparse
import json
import time
from pathlib import Path

import torch

import bitlinear as bitlinear_module
from bitlinear import BitLinear


IN_FEATURES = 1024
OUT_FEATURES = 1024
WARMUP = 5
ITERS = 20


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def get_model_memory_mb(model: torch.nn.Module) -> float:
    """Calculate static memory consumption of model weights and buffers."""
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.nelement() * param.element_size()
    for buffer in model.buffers():
        total_bytes += buffer.nelement() * buffer.element_size()
    return total_bytes / 1024**2


def get_memory_stats(device: torch.device) -> dict:
    """Get current memory usage statistics."""
    if device.type == "cuda":
        return {
            "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
            "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        }
    else:
        # For CPU, we can't easily track memory in the same way
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
        }


def benchmark(layer: BitLinear, x: torch.Tensor, label: str, mode: str) -> dict:
    """
    Benchmark a layer and return performance metrics.
    
    Returns:
        dict with keys: latency_ms, throughput_samples_per_sec, gflops, memory_stats, model_memory_mb
    """
    device = x.device
    batch_size = x.shape[0]
    in_features = getattr(layer, "in_features", IN_FEATURES)
    out_features = getattr(layer, "out_features", OUT_FEATURES)
    ops_per_forward = 2 * batch_size * in_features * out_features

    # Calculate static model memory
    model_memory_mb = get_model_memory_mb(layer)

    # Reset memory stats if CUDA
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    # Record memory before
    mem_before = get_memory_stats(device)

    # Warmup to avoid cold-start effects.
    with torch.inference_mode():
        for _ in range(WARMUP):
            layer(x)
        _synchronize_if_needed(device)

    # Record memory after warmup
    mem_after_warmup = get_memory_stats(device)

    # Timed runs.
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(ITERS):
            layer(x)
        _synchronize_if_needed(device)
    duration = time.perf_counter() - start

    # Record memory after inference
    mem_after = get_memory_stats(device)

    avg_ms = (duration / ITERS) * 1e3
    throughput = (batch_size * ITERS) / duration
    gflops = (ops_per_forward * ITERS) / duration / 1e9
    
    print(
        f"{label}: {avg_ms:.3f} ms/run, {throughput:.2f} samples/s, {gflops:.2f} GFLOP/s"
    )
    print(f"  Model memory: {model_memory_mb:.2f} MB")
    
    if device.type == "cuda":
        print(f"  Runtime memory: {mem_after['allocated_mb']:.2f} MB allocated, "
              f"{mem_after['max_allocated_mb']:.2f} MB peak")

    return {
        "mode": mode,
        "label": label,
        "batch_size": batch_size,
        "latency_ms": avg_ms,
        "throughput_samples_per_sec": throughput,
        "gflops": gflops,
        "model_memory_mb": model_memory_mb,
        "memory_before_mb": mem_before,
        "memory_after_warmup_mb": mem_after_warmup,
        "memory_after_mb": mem_after,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile BitLinear latency and memory.")
    parser.add_argument(
        "--device",
        default=None,
        help=(
            "Training device (e.g. cpu, cuda, cuda:1, mps). Defaults to cpu to match "
            "the optimized inference path."
        ),
    )
    parser.add_argument(
        "--deploy-device",
        default=None,
        help=(
            "Device to benchmark the deployed inference path on. Defaults to --device."
        ),
    )
    parser.add_argument(
        "--output",
        default="profile_results.json",
        help="Output JSON file to save profiling results (default: profile_results.json)",
    )
    return parser.parse_args()


def resolve_device(
    device_arg: str | None, *, default: torch.device | None = None
) -> torch.device:
    """
    Resolve the device to run on.

    NOTE: The optimized C++ BitLinear kernel in this repo is CPU-only. While
    training mode can run on CUDA via the pure-Python implementation, the
    deployed inference path will error out if you try to use CUDA tensors.
    To keep the perf script robust across machines with/without GPUs, we
    default to CPU here.
    """
    if device_arg is None:
        if default is not None:
            return default
        return torch.device("cpu")
    if device_arg == "cpu":
        return torch.device("cpu")

    try:
        device = torch.device(device_arg)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(f"Invalid device specification: {device_arg}") from exc

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")

    return device


def main() -> None:
    args = parse_args()
    train_device = resolve_device(args.device)
    deploy_device = resolve_device(args.deploy_device, default=train_device)

    if (
        deploy_device.type != "cpu"
        and bitlinear_module.HAS_BITLINEAR
        and not getattr(bitlinear_module, "_has_warned_fallback", False)
    ):
        print(
            "NOTE: Forcing fallback Python implementation for deployment benchmarks "
            f"to honor deploy device '{deploy_device}'."
        )
        bitlinear_module.HAS_BITLINEAR = False
        bitlinear_module._has_warned_fallback = True

    results = []

    for batch_size in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024):
        print(f"\n=== batch_size={batch_size} ===")
        layer = BitLinear(IN_FEATURES, OUT_FEATURES).to(train_device)
        x = torch.randn(
            batch_size, IN_FEATURES, device=train_device, dtype=torch.float32
        )

        # Benchmark eval mode (training mode before deployment)
        eval_result = benchmark(
            layer, x, f"Eval mode, batch={batch_size}", mode="eval"
        )
        results.append(eval_result)

        # Move to deploy device if different
        if train_device != deploy_device:
            _synchronize_if_needed(train_device)
            layer = layer.to(deploy_device)
            x = x.to(deploy_device)

        # Deploy and benchmark
        layer.deploy()
        deploy_result = benchmark(
            layer, x, f"Deploy mode, batch={batch_size}", mode="deploy"
        )
        results.append(deploy_result)

    # Save results to JSON
    output_path = Path(args.output)
    output_data = {
        "train_device": str(train_device),
        "deploy_device": str(deploy_device),
        "in_features": IN_FEATURES,
        "out_features": OUT_FEATURES,
        "warmup_iters": WARMUP,
        "benchmark_iters": ITERS,
        "results": results,
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()