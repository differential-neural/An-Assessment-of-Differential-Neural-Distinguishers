from train_nets import train_distinguisher
from cipher.chacha import ChaCha

# Script for training a ChaCha distinguisher using the same hyper-parameter as in the paper

chacha = ChaCha(n_rounds=3)
train_distinguisher(
    chacha, [0, 0, 0, 0x8000], calc_back=1, lr_high=0.002, lr_low=0.00014, kernel_size=7, reg_param=0.000014,
    n_train_samples=5*10**6  # We use halve the samples only (due to RAM limitations)
)
