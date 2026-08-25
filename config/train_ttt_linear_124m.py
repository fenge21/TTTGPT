# tttGPT-Linear, ~124M params -- flagship config
# comparable scale to GPT-2 small / nanoGPT default

n_layer = 12
n_head = 12
n_embd = 768
block_size = 1024
dropout = 0.0
bias = False

# TTT
ttt_type = 'linear'
ttt_lr = 1.0
chunk_size = 64
use_conv = True

# data (prepare with nanoGPT's data/shakespeare_char/prepare.py,
# or point dataset to any folder containing train.bin/val.bin/meta.pkl)
dataset = 'shakespeare_char'
batch_size = 8
block_size = 512

# optimization
learning_rate = 6e-4
max_iters = 20000
lr_decay_iters = 20000
warmup_iters = 500
min_lr = 6e-5
weight_decay = 0.1
grad_clip = 1.0

out_dir = 'out-ttt-linear'
wandb_run_name = 'ttt-linear-124m'
