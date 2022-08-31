from train_nets import train_distinguisher
from cipher.skinny import Skinny

# Script for training a Skinny distinguisher using the same hyper-parameter as in the paper

skinny = Skinny(n_rounds=7)
train_distinguisher(
    skinny, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0], calc_back=2, lr_high=0.0011, lr_low=0.000045,
    kernel_size=3, reg_param=0.000000043
)
