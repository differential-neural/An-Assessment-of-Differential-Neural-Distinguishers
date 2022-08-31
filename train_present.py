from train_nets import train_distinguisher
from cipher.present import Present

# Script for training a Present distinguisher using the same hyper-parameter as in the paper

present = Present(n_rounds=7)
train_distinguisher(
    present, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xd, 0, 0, 0, 0, 0], lr_high=0.003, lr_low=0.00028, kernel_size=1,
    reg_param=0.00000062
)
