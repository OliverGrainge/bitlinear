import argparse
import time

import torch

from main import BitLinear


IN_FEATURES = 1024
OUT_FEATURES = 1024
WARMUP = 5
ITERS = 20


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark(layer: BitLinear, x: torch.Tensor, label: str) -> None:
    device = x.device
    layer.to(device)

    batch_size = x.shape[0]
    ops_per_forward = 2 * batch_size * IN_FEATURES * OUT_FEATURES

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
    print(f"{label}: {avg_ms:.3f} ms/run, {throughput:.2f} samples/s, {gflops:.2f} GFLOP/s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark BitLinear throughput.")
    parser.add_argument(
        "--device",
        default=None,
        choices=("cpu", "cuda"),
        help="Device to run on. Defaults to cuda if available, else cpu.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    for batch_size in (1, 1024):
        print(f"\n=== batch_size={batch_size} ===")
        layer = BitLinear(IN_FEATURES, OUT_FEATURES).to(device)
        x = torch.randn(batch_size, IN_FEATURES, device=device, dtype=torch.float32)

        benchmark(layer, x, f"Training-mode (before deployment), batch={batch_size}")

        layer.deploy()
        benchmark(layer, x, f"Deployment-mode (after deployment), batch={batch_size}")


if __name__ == "__main__":
    main()


