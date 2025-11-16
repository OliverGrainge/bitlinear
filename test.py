import pytest 
import torch 


# Assuming main.py contains the BitLinear implementation
from main import BitLinear, HAS_BITLINEAR


@pytest.fixture 
def bitlinear(): 
    return BitLinear(12, 24)


def test_bitlinear_forward(bitlinear):
    x = torch.randn(10, 12)
    out = bitlinear(x)
    assert out.shape == (10, 24)


def test_bitlinear_backward(bitlinear):
    x = torch.randn(10, 12, requires_grad=True)
    out = bitlinear(x)
    grad_output = torch.randn_like(out)
    out.backward(grad_output)

    assert x.grad is not None
    assert x.grad.shape == x.shape

    grads = [p.grad for p in bitlinear.parameters() if p.requires_grad]
    assert grads and all(g is not None for g in grads)


def test_bitlinear_deploy(bitlinear):
    x = torch.randn(10, 12)
    out = bitlinear(x)
    bitlinear.deploy()
    out = bitlinear(x)
    assert out.shape == (10, 24)


def test_has_kernel(): 
    assert HAS_BITLINEAR == True


def test_bitlinear_eval_deploy_equivalence(bitlinear):
    """Test that eval mode and deploy mode produce similar outputs."""
    bitlinear.eval()
    x = torch.randn(10, 12)
    
    # Get output in eval mode (using training forward pass)
    with torch.no_grad():
        out_eval = bitlinear(x)
    
    # Deploy and get output
    bitlinear.deploy()
    with torch.no_grad():
        out_deploy = bitlinear(x)
    
    assert out_deploy.shape == (10, 24)
    
    # The outputs should be close but not identical due to:
    # 1. Floating point rounding differences
    # 2. Different code paths (Python vs C++)
    # We use a reasonable tolerance
    assert torch.allclose(out_eval, out_deploy, rtol=1e-4, atol=1e-5), \
        f"Max diff: {(out_eval - out_deploy).abs().max():.6f}"


def test_weight_packing(bitlinear): 
    assert bitlinear.weight.shape == (24, 12)
    assert bitlinear.weight.dtype == torch.float32 
    
    bitlinear.deploy() 
    
    # After deployment, weight parameter should be gone
    assert not hasattr(bitlinear, "weight")
    
    # Should have scale and packed weights as buffers
    assert hasattr(bitlinear, "w_scale")
    assert hasattr(bitlinear, "w_packed")
    
    # Scale should be a single value
    assert bitlinear.w_scale.numel() == 1
    
    # Packed weights should have correct shape
    assert bitlinear.w_packed.shape == (24, 12 // 4)
    
    # Check dtypes
    assert bitlinear.w_scale.dtype == torch.float32
    
    # The packed weights dtype depends on whether using C++ kernel or fallback
    if HAS_BITLINEAR:
        # C++ kernel packs into uint8
        assert bitlinear.w_packed.dtype == torch.uint8
    else:
        # Fallback doesn't actually pack, keeps as float32
        assert bitlinear.w_packed.dtype == torch.float32


def test_bias_handling(bitlinear):
    """Test that bias is properly converted from parameter to buffer."""
    assert hasattr(bitlinear, 'bias')
    assert isinstance(bitlinear.bias, torch.nn.Parameter)
    
    bias_values = bitlinear.bias.data.clone()
    
    bitlinear.deploy()
    
    # After deployment, bias should be a buffer, not a parameter
    assert hasattr(bitlinear, 'bias')
    assert not isinstance(bitlinear.bias, torch.nn.Parameter)
    assert torch.allclose(bitlinear.bias, bias_values)
    
    # Check it's in buffers
    assert 'bias' in dict(bitlinear.named_buffers())
    assert 'bias' not in dict(bitlinear.named_parameters())


def test_bitlinear_bias_forward():
    """Bias should be added in training forward path."""
    layer = BitLinear(4, 3, bias=True)
    bias = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32)
    
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.copy_(bias)
    
    x = torch.randn(2, 4)
    out = layer(x)
    
    expected = bias.unsqueeze(0).expand_as(out)
    assert torch.allclose(out, expected, atol=1e-6)


def test_bitlinear_bias_forward_deploy():
    """Bias buffer should still be applied after deploy."""
    layer = BitLinear(4, 3, bias=True)
    bias = torch.tensor([0.5, 0.0, -0.5], dtype=torch.float32)
    
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.copy_(bias)
    
    x = torch.randn(2, 4)
    
    layer.deploy()
    out = layer(x)
    
    expected = bias.unsqueeze(0).expand_as(out)
    assert torch.allclose(out, expected, atol=1e-6)


def test_bitlinear_bias_forward_batch1_divisible_by4():
    """Bias should be added correctly when batch_size=1 and dims divisible by 4."""
    layer = BitLinear(4, 4, bias=True)
    bias = torch.tensor([0.1, -0.2, 0.3, 0.4], dtype=torch.float32)

    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.copy_(bias)

    x = torch.randn(1, 4)

    # Get output in eval mode
    layer.eval()
    with torch.no_grad():
        out_eval = layer(x)

    # Deploy and get output
    layer.deploy()
    with torch.no_grad():
        out_deploy = layer(x)

    # Both should effectively be equal to the broadcast bias, and to each other
    expected = bias.unsqueeze(0).expand_as(out_eval)
    assert torch.allclose(out_eval, expected, atol=1e-6)
    assert torch.allclose(out_deploy, expected, atol=1e-6)
    assert torch.allclose(out_eval, out_deploy, rtol=1e-4, atol=1e-5)


def test_bitlinear_bias_forward_deploy_batch1_divisible_by4():
    """Bias buffer should still be applied after deploy when batch_size=1 and dims divisible by 4."""
    layer = BitLinear(4, 4, bias=True)
    bias = torch.tensor([0.5, 0.0, -0.5, 0.25], dtype=torch.float32)

    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.copy_(bias)

    x = torch.randn(1, 4)

    # Get output in eval mode
    layer.eval()
    with torch.no_grad():
        out_eval = layer(x)

    # Deploy and get output
    layer.deploy()
    with torch.no_grad():
        out_deploy = layer(x)

    # Both should effectively be equal to the broadcast bias, and to each other
    expected = bias.unsqueeze(0).expand_as(out_eval)
    assert torch.allclose(out_eval, expected, atol=1e-6)
    assert torch.allclose(out_deploy, expected, atol=1e-6)
    assert torch.allclose(out_eval, out_deploy, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "batch_size,in_features,out_features",
    [
        (1, 4, 4),
        (2, 4, 8),
        (3, 8, 4),
        (5, 8, 8),
        (7, 12, 4),
        (4, 12, 8),
    ],
)
def test_bitlinear_eval_deploy_equivalence_various_shapes(
    batch_size, in_features, out_features
):
    """Eval and deploy outputs should be similar for various shapes (dims divisible by 4)."""
    layer = BitLinear(in_features, out_features, bias=True)

    x = torch.randn(batch_size, in_features)

    # Eval path
    layer.eval()
    with torch.no_grad():
        out_eval = layer(x)

    # Deploy path
    layer.deploy()
    with torch.no_grad():
        out_deploy = layer(x)

    assert out_eval.shape == (batch_size, out_features)
    assert out_deploy.shape == (batch_size, out_features)

    # Allow small numerical differences between eval (Python) and deploy (C++)
    assert torch.allclose(out_eval, out_deploy, rtol=1e-4, atol=1e-5), (
        f"Shape (B={batch_size}, in={in_features}, out={out_features}) "
        f"max diff: {(out_eval - out_deploy).abs().max():.6f}"
    )


@pytest.mark.parametrize(
    "batch_size,seq_len,in_features,out_features",
    [
        (1, 2, 4, 4),
        (2, 3, 4, 8),
        (3, 4, 8, 4),
        (5, 5, 8, 8),
        (7, 2, 12, 4),
        (4, 6, 12, 8),
    ],
)
def test_bitlinear_eval_deploy_equivalence_various_shapes_3d(
    batch_size, seq_len, in_features, out_features
):
    """Eval and deploy outputs should be similar for 3D inputs (dims divisible by 4)."""
    layer = BitLinear(in_features, out_features, bias=True)

    x = torch.randn(batch_size, seq_len, in_features)

    # Eval path
    layer.eval()
    with torch.no_grad():
        out_eval = layer(x)

    # Deploy path
    layer.deploy()
    with torch.no_grad():
        out_deploy = layer(x)

    assert out_eval.shape == (batch_size, seq_len, out_features)
    assert out_deploy.shape == (batch_size, seq_len, out_features)

    # Allow small numerical differences between eval (Python) and deploy (C++)
    assert torch.allclose(out_eval, out_deploy, rtol=1e-4, atol=1e-5), (
        f"Shape (B={batch_size}, T={seq_len}, in={in_features}, out={out_features}) "
        f"max diff: {(out_eval - out_deploy).abs().max():.6f}"
    )