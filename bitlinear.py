"""Binary linear layer implementations with shared quantization utilities."""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

try:
    import _bitlinear as bnn

    HAS_BITLINEAR = True
except ImportError:
    HAS_BITLINEAR = False


def quantize_act(
    x: torch.Tensor, eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize activations to int8 range [-127, 127].

    Returns:
        x_scale: Scale factor for dequantization (NOT inverse scale!)
        x_quant: Quantized activations
    """
    orig_dtype = x.dtype
    x_fp32 = x.float()
    x_scale = x_fp32.abs().amax(dim=-1, keepdim=True).clamp(min=eps) / 127.0
    x_quant = (x_fp32 / x_scale).round().clamp(-128, 127)
    return x_scale.to(orig_dtype), x_quant.to(orig_dtype)


def quantize_weight(
    w: torch.Tensor, eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weights to ternary values {-1, 0, 1}.

    Returns:
        w_scale: Scale factor for dequantization (NOT inverse scale!)
        w_quant: Quantized weights
    """
    orig_dtype = w.dtype
    w_fp32 = w.float()
    w_scale = w_fp32.abs().mean().clamp(min=eps)
    w_quant = (w_fp32 / w_scale).round().clamp(-1, 1)
    return w_scale.to(dtype=orig_dtype), w_quant.to(dtype=orig_dtype)


def _fallback_bitlinear(
    x: torch.Tensor,
    w_scale: torch.Tensor,
    w_packed: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fallback Python implementation of bitlinear forward pass.

    Args:
        x: Input activations
        w_scale: Weight scale factor
        w_packed: Quantized weights (not actually packed in fallback)
        bias: Bias term
        eps: Epsilon for numerical stability
    """
    x_scale, x_quant = quantize_act(x, eps)
    # Straight-through estimator: forward uses quantized, backward uses original
    x_dequant = x + ((x_scale * x_quant) - x).detach()

    # In fallback, w_packed contains the quantized weights directly
    w_dequant = w_scale * w_packed

    return F.linear(x_dequant, w_dequant, bias)


def _fallback_prepare_weights(
    weight: torch.Tensor,
    eps: float,
    quant_type: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fallback Python implementation of weight preparation.

    Returns:
        w_scale: Weight scale factor
        w_quant: Quantized weights (not packed in fallback)
    """
    return quantize_weight(weight, eps)


class BitLinear(nn.Module):
    """
    A binary neural network linear layer that quantizes weights to {-1, 0, 1}.

    This layer supports two execution modes:
    - Training mode: Applies quantized weights while preserving gradient flow.
    - Deployment mode: Swaps parameters for packed buffers and a lightweight
      inference-only computation path.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        bias: Whether to include a bias term.
        eps: Small epsilon for numerical stability in quantization.
        quant_type: Identifier for activation/weight quantization pair to use.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        eps: float = 1e-6,
        quant_type: str = "ai8pc_wpt",
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.eps = eps
        self.quant_type = quant_type

        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The variable names here are misleading in the original code!
        # quantize_weight and quantize_act return (scale, quant), not (inv_scale, quant)
        w_scale, w_quant = quantize_weight(self.weight, self.eps)
        x_scale, x_quant = quantize_act(x, self.eps)

        # Apply straight-through estimator for training
        w_dequant = self.weight + ((w_scale * w_quant) - self.weight).detach()
        x_dequant = x + ((x_scale * x_quant) - x).detach()

        return F.linear(x_dequant, w_dequant, self.bias)

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, quant_type: str, eps: float = 1e-6
    ) -> "BitLinear":
        """Create a BitLinear layer from an existing nn.Linear layer."""
        qt = getattr(linear, "quant_type", quant_type)
        layer = cls(
            linear.in_features, linear.out_features, linear.bias is not None, eps, qt
        )
        layer.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            layer.bias.data.copy_(linear.bias.data)
        return layer

    def _init_weights(self) -> None:
        """Initialize weights using Xavier uniform initialization."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def deploy(self) -> None:
        """
        Deploy the layer for efficient inference by:
        1. Quantizing and packing weights (and bias)
        2. Removing original parameters
        3. Switching to optimized forward pass
        """
        # Save bias data before modifying parameters
        bias_data = self.bias.data.clone() if self.bias is not None else None

        # Use fallback if bitlinear is not available
        if HAS_BITLINEAR:
            # The C++ extension is CPU-only. Guard explicitly against CUDA
            # tensors so we fail loudly instead of segfaulting when trying to
            # dereference device pointers from C++.
            if self.weight.is_cuda:
                raise RuntimeError(
                    "BitLinear C++ kernel currently supports only CPU tensors. "
                    "Move the module to CPU (e.g. layer.to('cpu')) before calling "
                    "deploy(), or run inference with device='cpu'."
                )

            w_scale, w_packed = bnn.prepare_weights(
                self.weight,
                self.eps,
                self.quant_type,
            )
        else:
            w_scale, w_packed = _fallback_prepare_weights(
                self.weight,
                self.eps,
                self.quant_type,
            )

        # Delete original weight parameter
        del self.weight

        # Delete original bias parameter if it exists
        if self.bias is not None:
            del self.bias

        # Register packed weights and scale as buffers
        self.register_buffer("w_scale", w_scale)
        self.register_buffer("w_packed", w_packed)

        # Register bias as buffer (not bias_buffer!) if it exists
        if bias_data is not None:
            self.register_buffer("bias", bias_data)

        # Switch to optimized forward pass
        self.forward = self._deploy_forward

    def _deploy_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the quantized inference pathway after `deploy` has packed the weights."""
        # Get bias - it's stored as self.bias (a buffer), not self.bias_buffer
        bias = self.bias if hasattr(self, "bias") else None

        # Use fallback if bitlinear is not available
        if HAS_BITLINEAR:
            # Guard against accidentally passing CUDA tensors into the CPU-only
            # kernel. This used to cause hard segfaults on x86 when CUDA was
            # available; now we raise a clear error instead.
            if x.is_cuda:
                raise RuntimeError(
                    "BitLinear deployed C++ kernel is CPU-only. "
                    "Call layer.to('cpu') and move inputs to CPU before "
                    "running inference, or use the Python fallback path."
                )

            if x.ndim == 3:
                B, T, K = x.shape
                assert K == self.in_features  # sanity check (optional)

                # Flatten (B, T, K) -> (B*T, K)
                x_2d = x.reshape(B * T, K)

                # Call C++ kernel
                y_2d = bnn.bitlinear(x_2d, self.w_scale, self.w_packed, bias, self.eps)

                # Restore to (B, T, out_features)
                y = y_2d.reshape(B, T, self.out_features)
                return y
            else:
                return bnn.bitlinear(x, self.w_scale, self.w_packed, bias, self.eps)

        else:
            return _fallback_bitlinear(x, self.w_scale, self.w_packed, bias, self.eps)
