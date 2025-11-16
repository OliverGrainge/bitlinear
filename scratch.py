from main import BitLinear  
import torch 


layer = BitLinear(1024, 1024)
layer.eval()

x = torch.randn(1024, 1024)
out = layer(x)
layer.deploy()
out2 = layer(x)