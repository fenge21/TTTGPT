"""
tttGPT: a GPT-style language model whose attention is replaced by
Test-Time Training (TTT) layers.

Structure follows nanoGPT (Karpathy) so the two codebases can be compared
line by line; the sequence-modeling sublayer is the only fundamental change:

    nanoGPT:  x = x + Attention(LN(x));  x = x + MLP(LN(x))
    tttGPT:   x = x + TTT(LN(x));        x = x + SwiGLU(LN(x))

with an optional causal depth-wise temporal convolution in front of the TTT
layer (Mamba/Griffin-style backbone, recommended by the TTT paper).
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from ttt_layers import TTTLinear, TTTMLP


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class SwiGLU(nn.Module):
    """SwiGLU FFN (LLaMA/PaLM style)."""

    def __init__(self, config):
        super().__init__()
        hidden = int(config.n_embd * config.ffn_multiplier)
        self.w1 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.w2 = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.w3 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TemporalConv(nn.Module):
    """Causal depth-wise temporal convolution (Mamba/Griffin backbone piece).

    Collects local information before the recurrent-style TTT layer;
    the TTT paper found this improves perplexity.
    """

    def __init__(self, d_model, kernel_size=4):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size,
                              padding=kernel_size - 1, groups=d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.conv(x.transpose(1, 2))
        x = x[:, :, :residual.size(1)]          # causal crop
        return self.norm(x.transpose(1, 2) + residual)


class TTTBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.pre_conv = TemporalConv(config.n_embd) if config.use_conv else None

        d_head = config.n_embd // config.n_head if config.n_head else config.n_embd
        layer_cls = TTTLinear if config.ttt_type == 'linear' else TTTMLP
        self.ttt = layer_cls(
            d_model=config.n_embd,
            d_head=d_head,
            lr_base=config.ttt_lr,
            chunk_size=config.chunk_size,
        )

        self.mlp = SwiGLU(config)
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        if self.pre_conv is not None:
            x = self.pre_conv(x)
        x = x + self.dropout(self.ttt(self.ln_1(x)))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class TTTGPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12          # heads only set d_head = n_embd // n_head
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False
    # TTT-specific
    ttt_type: str = 'linear'     # 'linear' | 'mlp'
    ttt_lr: float = 1.0          # inner-loop base lr (0.1 for mlp)
    chunk_size: int = 64         # tokens per online-gradient chunk
    use_conv: bool = True        # causal depth-wise conv before TTT
    ffn_multiplier: float = 4.0  # SwiGLU hidden width


class TTTGPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList(TTTBlock(config) for _ in range(config.n_layer)),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        # scaled init for output projections (GPT-2 recipe)
        for pn, p in self.named_parameters():
            if pn.endswith('w2.weight') or pn.endswith('W_out.weight'):
                torch.nn.init.normal_(p, mean=0.0,
                                      std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"sequence length {t} exceeds block size {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)
        tok_emb = self.transformer.wte(idx)
        x = self.transformer.drop(tok_emb + self.transformer.wpe(pos))

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])   # last position only
            loss = None
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas,
            **({'fused': True} if use_fused else {}))
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """Rough MFU vs A100 bf16 peak (attention-free flops are approximate)."""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 3 * L * Q * Q * T      # no O(T^2) term
        flops_per_iter = flops_per_token * T * fwdbwd_per_iter
        return flops_per_iter / dt / 312e12

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        NOTE: this naive loop re-runs the full prefix each step (like nanoGPT).
        A TTT-native decoder would cache W per layer and advance O(1)/token --
        see README roadmap.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
