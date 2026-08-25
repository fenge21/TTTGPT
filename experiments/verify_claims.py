"""
Verify TTT theoretical advantages empirically.

Claim 1: O(n) time complexity (vs O(n^2) attention)
         -> latency scaling test with log-log exponent fitting
Claim 2: Constant inference state (vs growing KV cache)
         -> state size analysis + peak memory scaling
Claim 3: Keeps exploiting longer context (vs RNN plateau)
         -> positional NLL on data with long-range motifs,
            compared against an LSTM baseline (fixed-size vector state)
"""

import os
import sys
import time
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import TTTGPTConfig, TTTGPT

import importlib.util
spec = importlib.util.spec_from_file_location("nanogpt_model", "../nanoGPT/model.py")
nanogpt_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nanogpt_module)
GPTConfig = nanogpt_module.GPTConfig
GPT = nanogpt_module.GPT


# ----------------------------------------------------------------------
# LSTM baseline: hidden state is a FIXED-SIZE VECTOR (classic RNN)
# ----------------------------------------------------------------------
class LSTMLM(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_layer=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, n_embd)
        self.lstm = nn.LSTM(n_embd, n_embd, num_layers=n_layer, batch_first=True)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.n_embd = n_embd

    def logits(self, idx):
        x = self.emb(idx)
        x, _ = self.lstm(x)
        return self.head(x)

    def forward(self, idx, targets=None):
        logits = self.logits(idx)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1), ignore_index=-1)
        return logits, loss


def full_logits_generic(model, idx):
    """Full-sequence logits for GPT / TTTGPT (identical submodule layout)."""
    t = idx.shape[1]
    pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
    x = model.transformer.wte(idx) + model.transformer.wpe(pos)
    x = model.transformer.drop(x)
    for blk in model.transformer.h:
        x = blk(x)
    x = model.transformer.ln_f(x)
    return model.lm_head(x)


def model_logits(model, idx):
    if isinstance(model, LSTMLM):
        return model.logits(idx)
    return full_logits_generic(model, idx)


# ----------------------------------------------------------------------
# Data with LONG-RANGE structure: recurring random motifs
# A motif (48 tokens) recurs throughout each document with random gaps
# in between. Predicting a later motif occurrence REQUIRES remembering
# content from far back -> probes long-context exploitation.
# ----------------------------------------------------------------------
def gen_motif_data(total_tokens, seed=0, motif_len=24,
                   dist_choices=(100, 150, 200, 300, 400)):
    """Documents of EXACTLY 1024 tokens: one motif recurs with controlled
    gaps, and every repeat's predecessor is FULLY VISIBLE inside the doc
    (docs == attention/TTT/LSTM processing windows -> no anti-learning
    from half-visible pairs). Returns (data, events)."""
    rng = np.random.default_rng(seed)
    DOC = 1024
    n_docs = total_tokens // DOC
    out, events = [], []
    for i in range(n_docs):
        base = i * DOC
        motif = rng.integers(0, 35, motif_len)
        toks = [motif]; ev = [(base, None)]; pos = motif_len
        while True:
            d = int(rng.choice(dist_choices))
            if pos + d + motif_len > DOC - 8:      # leave padding room
                break
            toks.append(rng.integers(35, 61, d))
            toks.append(motif)
            ev.append((base + pos + d, d)); pos += d + motif_len
        pad = rng.integers(35, 61, DOC - pos)
        toks.append(pad)
        out.append(np.concatenate(toks)); events.extend(ev)
    return np.concatenate(out)[:n_docs*DOC].astype(np.uint16), events


def get_batch(data, bs, T, device):
    ix = torch.randint(len(data) - T - 1, (bs,))
    x = torch.stack([torch.from_numpy(data[i:i+T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+T].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def get_batch_docaligned(data, bs, T, device):
    """Windows start exactly at document boundaries so every motif repeat's
    predecessor is visible -> clean copy-signal (no anti-learning)."""
    n_docs = max(len(data) // T - 1, 1)   # keep i+1+T in bounds
    ix = torch.randint(n_docs, (bs,)) * T
    x = torch.stack([torch.from_numpy(data[i:i+T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+T].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


# ======================================================================
# PART 1: TIME COMPLEXITY  (latency vs sequence length)
# ======================================================================
def part1_latency(models, device, seq_lens=(128, 256, 512, 1024, 2048, 4096, 8192)):
    print("\n" + "=" * 78)
    print("PART 1  TIME COMPLEXITY  (forward latency vs sequence length, B=1)")
    print("  O(n^2) predicts: time doubles->4x when n doubles")
    print("  O(n)   predicts: time doubles->2x when n doubles")
    print("=" * 78)

    results = {}
    for name, model in models.items():
        model.eval()
        row = {}
        for T in seq_lens:
            x = torch.randint(0, 65, (1, T), device=device)
            with torch.no_grad():
                for _ in range(2):
                    model_logits(model, x)
            torch.cuda.synchronize()
            ts = []
            for _ in range(5):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    model_logits(model, x)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1000)
            row[T] = np.mean(ts)
        results[name] = row

    base = seq_lens[0]
    hdr = f"{'seq_len':>8} | {'n/base':>7} | "
    hdr += " | ".join(f"{n:>14}" for n in models)
    print(hdr)
    print("-" * len(hdr))
    for T in seq_lens:
        line = f"{T:>8} | {T/base:>6.0f}x | "
        line += " | ".join(
            f"{results[n][T]:>9.2f}ms ({results[n][T]/results[n][base]:>4.1f}x)" for n in models
        )
        print(line)

    # fitted exponent:  time ~ c * n^p
    print("\nFitted scaling exponent p (time ~ n^p):   [target: attn~2, ttt~1]")
    logsT = np.log(np.array(seq_lens, dtype=float))
    for name, model in models.items():
        # use points from 1024 upward where real compute dominates overhead
        pts = [T for T in seq_lens if T >= 1024]
        y = np.log([results[name][T] for T in pts])
        p, c = np.polyfit(np.log(np.array(pts, dtype=float)), y, 1)
        print(f"  {name:<18}: p = {p:.2f}")
    return results


# ======================================================================
# PART 2: INFERENCE STATE SIZE  (what must you carry to keep generating?)
# ======================================================================
def part2_state(models, device, cfg_common, seq_lens=(1024, 8192, 65536, 1048576)):
    print("\n" + "=" * 78)
    print("PART 2  STREAMING INFERENCE STATE  (bytes required to continue decoding)")
    print("  Attention: must KEEP K,V of all past tokens -> grows linearly with n")
    print("  TTT      : only carries weight matrix W       -> CONSTANT")
    print("=" * 78)

    L = cfg_common['n_layer']
    d = cfg_common['n_embd']

    print(f"\n{'context n':>12} | {'Attn KV cache (fp16)':>22} | {'TTT state W (fp32)':>20} | {'ratio':>8}")
    print("-" * 72)
    for T in seq_lens:
        kv = 2 * L * T * d * 2          # K and V, fp16
        w  = L * (d // 4) ** 2 * 4      # per-layer d_head x d_head W, fp32
        print(f"{T:>12,} | {kv:>19,.0f} B | {w:>17,.0f} B | {kv/w:>7.0f}x")

    # empirical: peak forward-pass memory delta vs n
    print("\nEmpirical peak memory DELTA of one forward pass (no_grad):")
    print(f"{'seq_len':>8} | " + " | ".join(f"{n:>18}" for n in models))
    print("-" * 60)
    for T in (1024, 4096, 8192):
        row = f"{T:>8} | "
        vals = []
        for name, model in models.items():
            model.eval()
            x = torch.randint(0, 65, (1, T), device=device)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base_mem = torch.cuda.memory_allocated()
            with torch.no_grad():
                model_logits(model, x)
            torch.cuda.synchronize()
            delta = (torch.cuda.max_memory_allocated() - base_mem) / 1024 / 1024
            vals.append(delta)
        print(row + " | ".join(f"{v:>15.1f} MB" for v in vals))


# ======================================================================
# PART 3: LONG-CONTEXT UTILIZATION  (positional NLL on motif data)
# ======================================================================
def part3_longctx(device, n_iters=5000):
    print("\n" + "=" * 78)
    print("PART 3  LONG-CONTEXT UTILIZATION")
    print("  Data: each doc has ONE symbol-motif (24 tok) recurring every")
    print("        100-400 letter-class tokens. Repeats ARE predictable.")
    print("  Metric: motif-token NLL bucketed by DISTANCE to previous occurrence.")
    print("  Positive gain over 'first' baseline = model actually uses memory.")
    print("=" * 78)

    T, bs = 1024, 12
    vocab = 65
    # 2500 unique docs: enough repetition for the copy-circuit phase
    # transition to occur within budget (verified: GPT learns genuine,
    # transferable induction on val here); best-val checkpoint selection
    # guards the late-stage memorization regime.
    train, _ = gen_motif_data(2_560_000, seed=1,
                              dist_choices=(80, 120, 180, 240))
    val, val_events = gen_motif_data(409_600, seed=2,
                                     dist_choices=(80, 120, 180, 240))
    print(f"motif data: train={len(train):,} tok ({len(train)//1024} unique docs), "
          f"val={len(val):,} tok ({len(val)//1024} docs)")

    n_layer, n_head, n_embd = 4, 4, 256
    cfgs = {
        'GPT (attention)': GPTConfig(block_size=T, vocab_size=vocab, n_layer=n_layer,
                                     n_head=n_head, n_embd=n_embd, bias=False),
        'TTT-Linear':      TTTGPTConfig(block_size=T, vocab_size=vocab, n_layer=n_layer,
                                        n_head=n_head, n_embd=n_embd, bias=False,
                                        ttt_type='linear', ttt_lr=1.0,
                                        chunk_size=64, use_conv=True),
    }
    models = {n: cls(c).to(device) for n, (c, cls) in
              [('GPT (attention)', (cfgs['GPT (attention)'], GPT)),
               ('TTT-Linear',      (cfgs['TTT-Linear'], TTTGPT))]}
    models['LSTM (vector state)'] = LSTMLM(vocab, n_embd, 2).to(device)

    # EQUAL compute for all models; early-stop style checkpoint selection
    # guards against the late-stage memorization regime
    schedules = {
        'GPT (attention)':      8000,
        'TTT-Linear':           8000,
        'LSTM (vector state)':  8000,
    }
    lrs = {'GPT (attention)': 1e-3, 'TTT-Linear': 1e-3, 'LSTM (vector state)': 2e-3}

    # fp32 everywhere: TTT inner loop accumulates W over chunks (bf16 -> NaN)
    ctx = nullcontext()

    @torch.no_grad()
    def quick_val(model, n=10):
        model.eval()
        ls = []
        for _ in range(n):
            Xv, Yv = get_batch_docaligned(val, bs, T, device)
            _, l = model(Xv, Yv)
            ls.append(l.item())
        model.train()
        return float(np.mean(ls))

    # checkpoints live in ./checkpoints (gitignored); existing files are
    # loaded to skip retraining
    ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    for name, model in models.items():
        n_it = schedules[name]
        ck = os.path.join(ckpt_dir, f"p3_{name.split()[0].lower()}.pt")
        if os.path.exists(ck):
            print(f"  [{name:<20}] loading existing checkpoint -> skip training")
            model.load_state_dict(torch.load(ck, map_location=device))
            continue
        optm = torch.optim.AdamW(model.parameters(), lr=lrs[name], betas=(0.9, 0.95), weight_decay=0.1)
        model.train()
        t0 = time.time()
        best_val, best_sd = float('inf'), None
        for i in range(n_it):
            X, Y = get_batch_docaligned(train, bs, T, device)   # doc-aligned windows
            with ctx:
                _, loss = model(X, Y)
            optm.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optm.step()
            if i % 500 == 0 or i == n_it - 1:
                vl = quick_val(model)
                if vl < best_val:
                    best_val = vl
                    best_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
                print(f"  [{name:<20}] iter {i:>5}  loss {loss.item():.4f}  val {vl:.4f}  ({time.time()-t0:.0f}s)")
        if best_sd is not None:                       # restore BEST-val weights
            model.load_state_dict(best_sd)
            print(f"  [{name:<20}] restored best checkpoint (val {best_val:.4f})")
        torch.save(model.state_dict(), ck)   # save immediately per model

    # disable TTT's random input-mask for deterministic evaluation
    # (transformer.h is an nn.ModuleList -> plain iteration yields modules)
    for model in models.values():
        if hasattr(model, 'transformer'):
            for blk in model.transformer.h:
                if hasattr(blk, 'ttt') and hasattr(blk.ttt, 'mask_ratio'):
                    blk.ttt.mask_ratio = 0.0

    # transparency: standard random-window val loss vs doc-aligned val loss,
    # plus an IID-noise probe: on pure random tokens NO model can beat
    # ln(65)=4.17; a value far below that indicates leakage/degeneracy.
    print("\nPost-training diagnostics:")
    for name, model in models.items():
        model.eval()
        rnd, ali = [], []
        with torch.no_grad():
            for _ in range(10):
                _, l = model(*get_batch(val, bs, T, device)); rnd.append(l.item())
                _, l = model(*get_batch_docaligned(val, bs, T, device)); ali.append(l.item())
            noise_nll = []
            for _ in range(10):
                Xn = torch.randint(0, vocab, (bs, T), device=device)
                lg = model_logits(model, Xn)
                noise_nll.append(F.cross_entropy(
                    lg.reshape(-1, vocab), Xn.reshape(-1)).item())
        print(f"  {name:<24} val(rand-win)={np.mean(rnd):.4f}  "
              f"val(doc-aligned)={np.mean(ali):.4f}  IID-noise-NLL={np.mean(noise_nll):.4f}  (chance=4.174)")

    # ---- distance-bucketed evaluation ----
    # Next-token alignment: logits at pos t predict token t+1.
    # A repeat occurrence is COUNTED only when its predecessor occurrence
    # lies FULLY INSIDE the same 1024-window (all models see windows
    # independently; otherwise the "memory" test would be unfair).
    dist_choices = [80, 120, 180, 240]
    IGNORE = 2 + len(dist_choices)
    n_win = (len(val) - 1) // T          # leave room for the +1 target shift

    # class map: 0=filler, 1=first occurrence, 2..2+nd-1=distance buckets
    # (doc-aligned windows guarantee every repeat's predecessor is visible,
    #  so no IGNORE class is needed)
    tag = torch.zeros(len(val), dtype=torch.long)
    for start, d in val_events:
        c = 1 if d is None else 2 + dist_choices.index(int(d))
        tag[start:start+24] = c
    tag_nll = tag[1 : n_win*T + 1].view(n_win, T)   # align: target t+1 <- logits t

    val_x   = torch.from_numpy(val[:n_win*T].astype(np.int64)).view(n_win, T)
    val_y   = torch.from_numpy(val[1 : n_win*T + 1].astype(np.int64)).view(n_win, T)

    @torch.no_grad()
    def bucket_nll(model):
        model.eval()
        sums = torch.zeros(IGNORE + 1)
        cnts = torch.zeros(IGNORE + 1)
        for i0 in range(0, n_win, bs):
            Xw = val_x[i0:i0+bs].to(device)
            Yw = val_y[i0:i0+bs].to(device)
            logits = model_logits(model, Xw)                     # (b,T,V)
            nll = F.cross_entropy(logits.reshape(-1, vocab),
                                  Yw.reshape(-1), reduction='none').view(Xw.shape).cpu()
            tw = tag_nll[i0:i0+bs]
            for c in range(IGNORE + 1):
                m = tw == c
                sums[c] += (nll * m).sum()
                cnts[c] += m.sum()
        return sums / cnts.clamp(min=1)

    buckets = {name: bucket_nll(model) for name, model in models.items()}

    # correct explicit mapping: idx -> label
    idx_label = [(0, "filler"), (1, "first")] + \
                [(2 + i, f"d={d}") for i, d in enumerate(dist_choices)]
    print("\nMotif-token NLL by DISTANCE to previous occurrence:")
    hdr = f"{'model':<24}" + "".join(f"{lab:>8}" for _, lab in idx_label)
    print(hdr)
    print("-" * len(hdr))
    for name, row in buckets.items():
        print(f"{name:<24}" + "".join(f"{row[i].item():>8.3f}" for i, _ in idx_label))

    # memory usage per model: repeat gain vs first at each distance
    print("\nMemory analysis (gain = first_NLL - repeat_NLL):")
    print(f"  {'model':<24}" + "".join(f"{'gain@'+str(d):>10}" for d in dist_choices))
    for name, row in buckets.items():
        first = row[1].item()
        gains = [first - row[2+i].item() for i in range(len(dist_choices))]
        print(f"  {name:<24}" + "".join(f"{g:>10.3f}" for g in gains))

    print("""
Interpretation:
  - 'first': motif never seen -> unpredictable baseline (memory useless)
  - 'd=80..240': recall of a PREVIOUSLY SEEN motif at that token distance.
    Positive gain over 'first' => the model actually USES its context.
  - expected ordering: attention keeps high gain at ALL distances (exact
    retrieval); TTT decays gracefully; LSTM's fixed vector forgets fastest,
    so its gain shrinks toward zero as d grows.
""")


# ======================================================================
def main():
    parts = set(sys.argv[1:]) or {'p1', 'p2', 'p3'}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  ({torch.cuda.get_device_name(0)})"
          if torch.cuda.is_available() else f"Device: {device}")

    torch.manual_seed(1337)

    n_layer, n_head, n_embd, V = 4, 4, 256, 65
    cfg_common = dict(n_layer=n_layer, n_embd=n_embd)

    if 'p1' in parts or 'p2' in parts:
        nano = GPT(GPTConfig(block_size=8192, vocab_size=V, n_layer=n_layer,
                             n_head=n_head, n_embd=n_embd, bias=False)).to(device)
        ttt = TTTGPT(TTTGPTConfig(block_size=8192, vocab_size=V, n_layer=n_layer,
                                  n_head=n_head, n_embd=n_embd, bias=False,
                                  ttt_type='linear', ttt_lr=1.0, chunk_size=64,
                                  use_conv=True)).to(device)
        models = {'GPT (attention)': nano, 'TTT-Linear': ttt}
        for m in models.values():
            m.eval()
        if 'p1' in parts:
            part1_latency(models, device)
        if 'p2' in parts:
            part2_state(models, device, cfg_common)
    if 'p3' in parts:
        part3_longctx(device, n_iters=5000)


if __name__ == '__main__':
    main()
