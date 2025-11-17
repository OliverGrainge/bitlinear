try:
    import _bitlinear as bnn

    HAS_BITLINEAR = True
except ImportError:
    HAS_BITLINEAR = False


import torch


x = torch.randn(10, 12).cuda()
w = torch.randn(24, 12).cpu()

print(w.shape, w.dtype)

w_scale_cpu, w_packed_cpu = bnn.prepare_weights(w)

w_scale_cuda, w_packed_cuda = bnn.prepare_weights(w.cuda())

print(w_scale_cuda.shape, w_scale_cuda.dtype)
print(w_packed_cuda.shape, w_packed_cuda.dtype)

print(w_scale_cpu.shape, w_scale_cpu.dtype)
print(w_packed_cpu.shape, w_packed_cpu.dtype)

print(w_packed_cpu.flatten()[:10])
print(w_packed_cuda.flatten()[:10])

y_cuda = bnn.bitlinear(x.cuda(), w_scale_cuda, w_packed_cuda)

y_cpu = bnn.bitlinear(x.cpu(), w_scale_cpu, w_packed_cpu)
print("cuda")
print(y_cuda.shape, y_cuda.dtype)
print(y_cpu.shape, y_cpu.dtype)

print(y_cuda.flatten()[:10])
print(y_cpu.flatten()[:10])
