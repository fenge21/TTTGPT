# tttGPT

**A nanoGPT-style reference implementation of Test-Time Training (TTT) language models.**

The hidden state of a TTT layer is not a vector (RNN) and not a sequence (Transformer) —
**it is a neural network that trains itself on every token it reads.**

```
Transformer :  y_t = Attention(q_t, K, V)              # look everything up
RNN         :  h_t = tanh(W_h h_{t-1} + W_x x_t)        # squeeze into a vector
TTT         :  W_t = W_{t-1} - η ∇_W ||f(k_t;W_{t-1}) - v_t||²
               y_t = f(q_t; W_t)                        # LEARN from every token
```

Built to make [arXiv:2407.04620](https://arxiv.org/abs/2407.04620)
(Sun et al., 2024 — the architecture behind
[SSI's post-scaling bet](#why-this-matters)) reproducible at nanoGPT scale,
with an honest accounting of what works, what doesn't, and why.

---

## Why this matters

|  | Transformer | RNN / Mamba | **TTT** |
|---|---|---|---|
| FLOPs per sequence | O(n²) | O(n) | **O(n)** |
| Streaming state size | grows with n (KV cache) | constant but tiny | **constant AND expressive** |
| Uses longer context? | yes | plateaus (~16k) | **yes, keeps improving** |

A fixed-size hidden state that is itself *trainable* can keep compressing more
context into better predictions — the property RNNs lack and Transformers pay
quadratically for. This is the first architecture class where "the model
learns at inference time" is the mechanism, not a bolt-on.

## What's verified in this repo

Run `python experiments/verify_claims.py` yourself. Measured on one RTX 5060 Ti:

**1. Linear time scaling — confirmed**
Fitted exponent of latency vs sequence length: TTT-Linear **p = 0.90 ≈ 1**
(attention p → 2; Flash Attention constants hide it below ~8k).

**2. Constant streaming state — confirmed**
State needed to continue decoding:

| context length | attention KV cache | TTT state W | ratio |
|---|---|---|---|
| 1K | 4 MB | 64 KB | 64× |
| 64K | 256 MB | 64 KB | 4096× |
| 1M | 4 GB | 64 KB | **65536×** |

**3. Long-context memory utilization — partially reproduced**
On a motif-recall benchmark (recurring random patterns at controlled distances),
attention gains ~+3 nats by remembering, an LSTM baseline gains ~+0.1
(fixed-vector bottleneck confirmed), while our simplified TTT layer did **not**
form transferable copy circuits within budget. See the honest write-up in
[RESULTS.md](RESULTS.md) — including the four failure modes we had to fix
(causality leaks in naive dual form, train/eval mask skew, memorization
shortcuts, NaN-prone inner lr). Reproducing the paper's quality claims
clearly requires their systems work, which is exactly what this repo documents.

## Quick start

```bash
pip install -r requirements.txt

# 1) data (tiny Shakespeare, char-level)
python data/shakespeare_char/prepare.py

# 2) train (~10 min on one consumer GPU)
python train.py --max_iters=3000 --eval_interval=200

# 3) sample
python sample.py --out_dir=out --start="First Citizen:" --num_samples=2
```

Larger config: `python train.py config/train_ttt_linear_124m.py`

### Verify the claims

```bash
python causality_test.py          # future-token mutation must not affect past outputs
python experiments/verify_claims.py p1   # time-complexity scaling + fitted exponents
python experiments/verify_claims.py p2   # state-size analysis (KV cache vs W)
python experiments/verify_claims.py p3   # long-context memory vs LSTM baseline
```

## Repository layout

```
ttt_layers.py        TTTLinear / TTTMLP layers (the core contribution)
model.py             TTTGPT: nanoGPT skeleton with TTT blocks + SwiGLU + temporal conv
train.py             single-file trainer (DDP, cosine LR, checkpointing)
sample.py            generation from a checkpoint
causality_test.py    automated causality regression test for the layer
experiments/
  verify_claims.py   the three-part empirical verification suite
config/              ready-to-run configs (124M linear / mlp)
data/                dataset prep (shakespeare_char bundled)
RESULTS.md           full experimental log, numbers, failure modes
```

## Implementation notes (the part most repos skip)

The TTT inner loop is a *recurrent optimization*, and naive implementations
break in interesting ways. Things we learned by breaking them:

1. **Causality**: computing a chunk's gradient before its output lets early
   positions peek at later tokens inside the chunk. Our two-pass scheme
   snapshots state first (`output-before-update`) and adds a parallel
   first-order correction for earlier tokens within the chunk.
2. **Inner-lr explosion**: the learnable `log_lr_scale` runs away without a
   clamp → NaN within 100 steps.
3. **Mask skew**: any corruption applied only during training shifts the
   input distribution at eval and destroys the model.
4. **Memorization shortcuts**: with few unique documents, unrolled-inner-loop
   models learn to *identify the document* rather than copy content;
   validation loss diverges monotonically while training loss hits ~0.
5. **Mean, don't sum**, the accumulated chunk gradient — otherwise update
   magnitude scales with chunk length and W explodes over long sequences.

`ttt_layers.py` docstrings point at each fix.

## Roadmap

- [ ] exact closed-form per-token dual form (official JAX equivalence)
- [ ] fused Triton kernel for pass 1+2 (paper shows >Transformer speed at 8k+)
- [ ] per-token learnable inner lr η(x) = base·sigmoid(θ_lr·x)
- [ ] LN + residual inside the inner model f
- [ ] O(1)/token cached decoding (carry W across generate() steps)
- [ ] torch.compile support for the chunked scan

PRs welcome — especially kernel work.

## Citation

```bibtex
@article{sun2024learning,
  title={Learning to (Learn at Test Time): RNNs with Expressive Hidden States},
  author={Sun, Yu and Li, Xinhao and Dalal, Karan and Xu, Jiarui and Vikram, Arjun and
          Zhang, Hanlin and Dubois, Yann and Lu, Xin and Al-Shaibi, Sami and
          Longpre, Simon and others},
  journal={arXiv preprint arXiv:2407.04620},
  year={2024}
}
```

Architecture skeleton inherits from nanoGPT (Karpathy, MIT).
This repo is an independent educational implementation, not affiliated with
the paper authors or SSI.

## License

MIT — see [LICENSE](LICENSE).
