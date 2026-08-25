"""Causality self-test for the TTT layer:
changing FUTURE tokens must NEVER affect PAST outputs."""
import sys
import torch
sys.path.insert(0, '.')
from ttt_layers import TTTLinear

torch.manual_seed(0)

layer = TTTLinear(d_model=128, d_head=64, chunk_size=64)
layer.eval()

B, T, D = 2, 512, 128
x = torch.randn(B, T, D)

with torch.no_grad():
    y1 = layer(x)

    x2 = x.clone()
    x2[:, 384:] = torch.randn_like(x2[:, 384:])      # mutate far future
    y2 = layer(x2)

    x3 = x.clone()
    x3[:, 100:] += 0.5 * torch.randn_like(x3[:, 100:])  # mutate near future
    y3 = layer(x3)

# outputs at positions < 64 (before any affected chunk boundary) must match
safe = 64
d_far = (y1[:, :safe] - y2[:, :safe]).abs().max().item()
d_near = (y1[:, :safe] - y3[:, :safe]).abs().max().item()
print(f"max |Δy| at positions [{0}:{safe}] when mutating t>=384: {d_far:.3e}")
print(f"max |Δy| at positions [{0}:{safe}] when mutating t>=100: {d_near:.3e}")

ok = d_far < 1e-5 and d_near < 1e-5
print("CAUSALITY:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
