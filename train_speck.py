from train_nets import train_distinguisher
from cipher.speck import Speck

# Script for training a Speck distinguisher using the same hyper-parameter as in the paper

speck = Speck(n_rounds=7)
train_distinguisher(
    speck, [0x40, 0], lr_high=0.0035, lr_low=0.00022, reg_param=0.000000849, n_neurons=80, n_filters=16
)
