from train_nets import train_distinguisher
from cipher.simon import Simon

# Script for training a Simon distinguisher using the same hyper-parameter as in the paper

simon = Simon(n_rounds=9)
train_distinguisher(
    simon, [0, 0x40], lr_high=0.0027, lr_low=0.0002, kernel_size=7, reg_param=0.0000022, cconv=True
)
