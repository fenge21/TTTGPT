# tttGPT-MLP variant -- more expressive hidden state, much slower in eager
# PyTorch (sequential inner loop). Experimental; see README.

n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# TTT
ttt_type = 'mlp'
ttt_lr = 0.1                    # paper: lower base lr for the MLP inner loop
chunk_size = 16                 # smaller chunks for the sequential inner loop
use_conv = True

dataset = 'shakespeare_char'
batch_size = 4
block_size = 256

learning_rate = 6e-4
max_iters = 20000
lr_decay_iters = 20000
warmup_iters = 500
min_lr = 6e-5
weight_decay = 0.1
grad_clip = 1.0

out_dir = 'out-ttt-mlp'
wandb_run_name = 'ttt-mlp-124m'
