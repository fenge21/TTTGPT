"""
tttGPT training script -- nanoGPT-style, single file.

Single GPU:  python train.py --dataset=shakespeare_char --max_iters=5000
Multi GPU :  torchrun --standalone --nproc_per_node=4 train.py
Config file: python train.py config/train_ttt_linear_124m.py
"""

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import TTTGPTConfig, TTTGPT

# -----------------------------------------------------------------------------
# I/O
out_dir = 'out'
eval_interval = 500
log_interval = 10
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'            # 'scratch' | 'resume'
wandb_log = False
wandb_project = 'tttgpt'
wandb_run_name = 'run'
# data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 12
block_size = 1024
# model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.0
bias = False
ttt_type = 'linear'              # 'linear' | 'mlp'
ttt_lr = 1.0                     # inner-loop base lr (0.1 for mlp)
chunk_size = 64                  # tokens per TTT online-gradient chunk
use_conv = True                  # causal depth-wise conv before TTT
# optimizer
learning_rate = 6e-4
max_iters = 10000
weight_decay = 0.1
beta1, beta2 = 0.9, 0.95
grad_clip = 1.0
# lr schedule (cosine with warmup)
decay_lr = True
warmup_iters = 200
lr_decay_iters = 10000
min_lr = 6e-5
# system
backend = 'nccl'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
compile_model = False            # see README: torch.compile currently untested on the chunked layer
# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items()
               if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
    seed_offset = ddp_rank
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

print(f"tokens per iteration: {gradient_accumulation_steps * ddp_world_size * batch_size * block_size:,}")
if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
           'float16': torch.float16}[dtype]
# NOTE: autocast is OFF by default for TTT -- the inner loop accumulates state
# across chunks and low precision there caused NaNs in our experiments.
ctx = (torch.amp.autocast(device_type=device_type, dtype=ptdtype)
       if device_type == 'cuda' and dtype != 'float32' else nullcontext())

data_dir = os.path.join('data', dataset)

def get_batch(split):
    path = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

iter_num, best_val_loss = 0, 1e9

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta_vocab_size = pickle.load(f)['vocab_size']
    print(f"found vocab_size = {meta_vocab_size}")

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                  block_size=block_size, bias=bias, vocab_size=None,
                  dropout=dropout, ttt_type=ttt_type, ttt_lr=ttt_lr,
                  chunk_size=chunk_size, use_conv=use_conv)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    model_args['vocab_size'] = meta_vocab_size or 50304
    model = TTTGPT(TTTGPTConfig(**model_args))
elif init_from == 'resume':
    print(f"Resuming from {out_dir}")
    checkpoint = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
    for k in model_args:
        if k in checkpoint['model_args']:
            model_args[k] = checkpoint['model_args'][k]
    model = TTTGPT(TTTGPTConfig(**model_args))
    state_dict = checkpoint['model']
    for k in list(state_dict):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

model.to(device)
scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

if compile_model:
    print("compiling model (torch.compile)...")
    unoptimized_model = model
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    return min_lr + 0.5 * (learning_rate - min_lr) * (1 + math.cos(math.pi * ratio))

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train {losses['train']:.4f}  val {losses['val']:.4f}")
        if wandb_log:
            wandb.log({"iter": iter_num, "train/loss": losses['train'],
                       "val/loss": losses['val'], "lr": lr, "mfu": running_mfu * 100})
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                torch.save({'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'model_args': model_args,
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': config},
                           os.path.join(out_dir, 'ckpt.pt'))
                print(f"saved checkpoint to {out_dir}")
    if iter_num == 0 and eval_only:
        break

    for micro in range(gradient_accumulation_steps):
        if ddp:
            model.require_backward_grad_sync = (micro == gradient_accumulation_steps - 1)
        with ctx:
            _, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
        scaler.scale(loss).backward()

    if grad_clip:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(f"iter {iter_num}: loss {lossf:.4f}  {dt*1000:.1f}ms  mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
