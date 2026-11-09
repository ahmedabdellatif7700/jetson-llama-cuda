"""Fail unless CUDA supports both a forward and backward operation."""
import torch

assert torch.cuda.is_available(), 'CUDA unavailable; CPU training is not enabled'
x = torch.randn(128, 128, device='cuda', requires_grad=True)
(x @ x.T).square().mean().backward()
torch.cuda.synchronize()
assert x.grad is not None and torch.isfinite(x.grad).all()
print('CUDA forward/backward passed:', torch.__version__, torch.version.cuda,
      torch.cuda.get_device_name(), torch.cuda.get_device_capability())
