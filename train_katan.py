from train_nets import train_distinguisher
from cipher.katan import Katan

# Script for training a Katan distinguisher using the same hyper-parameter as in the paper

katan = Katan(n_rounds=66)
train_distinguisher(
    katan, [0x2000], calc_back=2, lr_high=0.0017, lr_low=0.00006, kernel_size=7, reg_param=0.000006
)
