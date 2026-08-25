"""
Test-Time Training (TTT) layers -- reference PyTorch implementation.

Based on "Learning to (Learn at Test Time): RNNs with Expressive Hidden States"
(Yu Sun et al., 2024, arXiv:2407.04620).

Core idea
---------
The hidden state of the layer IS a machine learning model f(x; W), and the
state update rule is one step of self-supervised gradient descent:

    W_t = W_{t-1} - eta * grad_W || f(k_t; W_{t-1}) - v_t ||^2      (inner loop)
    y_t = f(q_t; W_t)

where k/v/q are learned "views" (projections) of the input token, exactly
analogous to attention's K/V/Q. Only the OUTER parameters (theta_K/V/Q,
gates, norms, W0, lr scale) are trained by the outer loop (your optimizer);
W is a running state, recomputed every forward pass.

This implementation
-------------------
* Causality-safe two-pass chunked online gradient:
    Pass 1 (sequential over chunks): carry state W, snapshot before each
            chunk's averaged gradient step -> W_starts.
    Pass 2 (fully parallel): output for token t uses its chunk-start state
            plus a first-order correction for earlier tokens in the chunk,
                y_t = W_c q_t + eta * sum_{j in c, j<t} v_j (k_j . q_t)
          which is the 1st-order expansion of exact online GD around W_c.
* Stability kit (each piece earned the hard way -- see RESULTS.md):
    - learnable initial state W0
    - LayerNorm on K/V/Q views (bounds inner-loop magnitudes)
    - clamped learnable log-lr (prevents outer-loop explosion)
    - mean (not summed) gradient per chunk

Known gaps vs the official JAX/CUDA implementation:
    - zeroth/first-order chunk approximation instead of exact closed-form
      per-token dual form
    - eager PyTorch loop over chunks (no fused kernel)
    - no per-token learnable lr eta(x) = base * sigmoid(theta_lr . x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTTLinear(nn.Module):
    """
    TTT layer whose inner model is linear: f(x; W) = W @ x,  W in R^{d x d}.

    Fastest variant; matches Transformer perplexity at 8k context in the
    original paper and keeps improving beyond it.
    """

    def __init__(self, d_model, d_head=None, lr_base=1.0, chunk_size=64):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head if d_head is not None else d_model
        self.lr_base = lr_base
        self.chunk_size = chunk_size

        # outer-loop projections ("views", analogous to attention K/V/Q)
        self.theta_K = nn.Linear(d_model, self.d_head, bias=False)
        self.theta_V = nn.Linear(d_model, self.d_head, bias=False)
        self.theta_Q = nn.Linear(d_model, self.d_head, bias=False)

        # SwiGLU-style output gate
        self.W_gate = nn.Linear(self.d_head, d_model, bias=False)
        self.W_out = nn.Linear(self.d_head, d_model, bias=False)

        # stability kit
        self.view_norm = nn.LayerNorm(self.d_head)
        self.norm = nn.LayerNorm(self.d_head)
        self.log_lr_scale = nn.Parameter(torch.zeros(1))
        self.W0 = nn.Parameter(torch.eye(self.d_head))

        self._init_views()

    def _init_views(self):
        for lin in (self.theta_K, self.theta_V, self.theta_Q):
            nn.init.normal_(lin.weight, std=0.02)

    def _eta(self, d):
        # clamped so the outer loop cannot blow up the inner learning rate
        return self.lr_base * self.log_lr_scale.clamp(-6.0, 2.0).exp() / d

    def _finish(self, y):
        y = self.norm(y)
        gate = torch.sigmoid(self.W_gate(y))
        return gate * self.W_out(y)

    def forward(self, x):
        B, T, D = x.shape
        d = self.d_head
        device = x.device

        k = self.view_norm(self.theta_K(x))   # (B, T, d)
        v = self.view_norm(self.theta_V(x))   # (B, T, d)
        q = self.view_norm(self.theta_Q(x))   # (B, T, d)
        eta = self._eta(d)

        if T % self.chunk_size != 0:
            return self._forward_ragged(k, v, q, eta)

        b = self.chunk_size
        nc = T // b

        # ---- pass 1: sequential state snapshots (output-before-update) ----
        W_starts = []
        W = self.W0.unsqueeze(0).expand(B, -1, -1).clone()
        for c in range(nc):
            W_starts.append(W)
            kc = k[:, c * b:(c + 1) * b]
            vc = v[:, c * b:(c + 1) * b]
            pred = torch.bmm(W, kc.transpose(1, 2))       # (B, d, b)
            err = pred - vc.transpose(1, 2)               # (B, d, b)
            grad = torch.bmm(err, kc) / b                 # (B, d, d), MEAN grad
            W = W - eta * grad
        W_starts = torch.stack(W_starts, dim=1)           # (B, nc, d, d)

        # ---- pass 2: parallel outputs + intra-chunk online correction ----
        Kc = k.view(B, nc, b, d)
        Vc = v.view(B, nc, b, d)
        Qc = q.view(B, nc, b, d)

        base = torch.einsum('bnij,bncj->bnci', W_starts, Qc)  # (B,nc,b,d)

        qk = torch.matmul(Qc, Kc.transpose(-1, -2))           # (B,nc,b,b)
        causal = torch.tril(torch.ones(b, b, device=device), diagonal=-1)
        corr = torch.matmul(qk * causal, Vc) * eta            # (B,nc,b,d)

        y = (base + corr).reshape(B, T, d)
        return self._finish(y)

    def _forward_ragged(self, k, v, q, eta):
        """Zeroth-order fallback when T is not a multiple of chunk_size."""
        B, T, _ = k.shape
        outputs = []
        W = self.W0.unsqueeze(0).expand(B, -1, -1).clone()
        for t in range(0, T, self.chunk_size):
            end = min(t + self.chunk_size, T)
            n = end - t
            kc, vc, qc = k[:, t:end], v[:, t:end], q[:, t:end]
            out = torch.bmm(W, qc.transpose(1, 2))
            outputs.append(out.transpose(1, 2))
            pred = torch.bmm(W, kc.transpose(1, 2))
            err = pred - vc.transpose(1, 2)
            W = W - eta * torch.bmm(err, kc) / n
        return self._finish(torch.cat(outputs, dim=1))


class TTTMLP(nn.Module):
    """
    TTT layer whose inner model is a 2-layer MLP: f(x) = W2 GELU(W1 x).

    More expressive hidden state (best long-context potential in the paper),
    but the inner loop is inherently sequential here -> SLOW in pure PyTorch.
    Treat as experimental/reference; use TTTLinear for anything serious.
    Shares the stability kit (view norm, clamped lr) with TTTLinear.
    """

    def __init__(self, d_model, d_head=None, lr_base=0.1, chunk_size=16):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head if d_head is not None else d_model
        self.inner_dim = 4 * self.d_head
        self.lr_base = lr_base
        self.chunk_size = chunk_size

        self.theta_K = nn.Linear(d_model, self.d_head, bias=False)
        self.theta_V = nn.Linear(d_model, self.d_head, bias=False)
        self.theta_Q = nn.Linear(d_model, self.d_head, bias=False)

        self.W_gate = nn.Linear(self.d_head, d_model, bias=False)
        self.W_out = nn.Linear(self.d_head, d_model, bias=False)

        self.view_norm = nn.LayerNorm(self.d_head)
        self.norm = nn.LayerNorm(self.d_head)
        self.log_lr_scale = nn.Parameter(torch.zeros(1))

        self._init_views()

    def _init_views(self):
        for lin in (self.theta_K, self.theta_V, self.theta_Q):
            nn.init.normal_(lin.weight, std=0.02)

    def _finish(self, y):
        y = self.norm(y)
        gate = torch.sigmoid(self.W_gate(y))
        return gate * self.W_out(y)

    def forward(self, x):
        B, T, D = x.shape
        d, inner = self.d_head, self.inner_dim
        device = x.device

        k = self.view_norm(self.theta_K(x))
        v = self.view_norm(self.theta_V(x))
        q = self.view_norm(self.theta_Q(x))

        # clamped lr, scaled like TTTLinear
        eta = 0.1 * self.log_lr_scale.clamp(-6.0, 2.0).exp() / d

        # per-sequence inner-model state (recomputed each forward)
        W1 = torch.randn(B, inner, d, device=device) * 0.02
        W2 = torch.randn(B, d, inner, device=device) * 0.02

        outputs = []
        for t in range(0, T, self.chunk_size):
            end = min(t + self.chunk_size, T)
            kc, vc, qc = k[:, t:end], v[:, t:end], q[:, t:end]

            for i in range(end - t):                       # sequential inner GD
                k_i = kc[:, i].unsqueeze(-1)               # (B, d, 1)
                v_i = vc[:, i]                             # (B, d)

                h = F.gelu(torch.bmm(W1, k_i))             # (B, inner, 1)
                err = torch.bmm(W2, h).squeeze(-1) - v_i   # (B, d)

                W2_grad = torch.bmm(err.unsqueeze(-1), h.transpose(1, 2))
                dh = torch.bmm(W2.transpose(1, 2), err.unsqueeze(-1)) \
                     * (h > 0).float()
                W1_grad = torch.bmm(dh, k_i.transpose(1, 2))

                W1 = W1 - eta * W1_grad
                W2 = W2 - eta * W2_grad

            out = torch.bmm(W2, F.gelu(torch.bmm(W1, qc.transpose(1, 2))))
            outputs.append(out.transpose(1, 2))

        return self._finish(torch.cat(outputs, dim=1))
