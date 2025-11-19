import argparse
import time

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


def benchmark(layer: BitLinear, x: torch.Tensor, label: str) -> None:
    device = x.device
    batch_size = x.shape[0]
    in_features = getattr(layer, "in_features", IN_FEATURES)
    out_features = getattr(layer, "out_features", OUT_FEATURES)
    ops_per_forward = 2 * batch_size * in_features * out_features

    # Warmup to avoid cold-start effects.
    with torch.inference_mode():
        for _ in range(WARMUP):
            layer(x)
        _synchronize_if_needed(device)

    # Timed runs.
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(ITERS):
            layer(x)
        _synchronize_if_needed(device)
    duration = time.perf_counter() - start

    avg_ms = (duration / ITERS) * 1e3
    throughput = (batch_size * ITERS) / duration
    gflops = (ops_per_forward * ITERS) / duration / 1e9
    print(
        f"{label}: {avg_ms:.3f} ms/run, {throughput:.2f} samples/s, {gflops:.2f} GFLOP/s"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark BitLinear throughput.")
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

    for batch_size in (1, 1024):
        print(f"\n=== batch_size={batch_size} ===")
        layer = BitLinear(IN_FEATURES, OUT_FEATURES).to(train_device)
        x = torch.randn(
            batch_size, IN_FEATURES, device=train_device, dtype=torch.float32
        )

        benchmark(
            layer, x, f"Training-mode (before deployment), batch={batch_size}"
        )

        if train_device != deploy_device:
            _synchronize_if_needed(train_device)
            layer = layer.to(deploy_device)
            x = x.to(deploy_device)

        layer.deploy()
        benchmark(layer, x, f"Deployment-mode (after deployment), batch={batch_size}")


if __name__ == "__main__":
    main()
