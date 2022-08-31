from scipy.stats import binom
import numpy as np
import matplotlib.pyplot as plt

from cipher.chacha import ChaCha
from cipher.katan import Katan
from cipher.present import Present
from cipher.simon import Simon
from cipher.skinny import Skinny
from cipher.speck import Speck
from make_train_data import make_train_data

n_samples = 10**7
significance_level = 0.01

# Initialize ciphers
chacha = ChaCha(n_rounds=3)
katan = Katan(n_rounds=66)
present = Present(n_rounds=7)
simon = Simon(n_rounds=9)
skinny = Skinny(n_rounds=7)
speck = Speck(n_rounds=7)


def critical_values(bonferroni_correction=1):
    """
    :param bonferroni_correction: For how many experiments we want to correct for
    :return: Critical values for the significance level and number of samples defined above
    """
    half_mass = significance_level/(2*bonferroni_correction)
    critical_low = binom.ppf(half_mass, n_samples, 0.5) / n_samples - 0.5
    critical_high = binom.ppf(1 - half_mass, n_samples, 0.5) / n_samples - 0.5
    return critical_low, critical_high


def eval_biases(x, save_as):
    """
    :param x: Ciphertext-pairs in binary form
    :param save_as: Where to save the plot
    :return: List of biases for all possible one bit masks
    """
    # Calculate frequency of one-bit differences for the identical bit position of the two states
    ct_size = len(x[0]) // 2
    biases = np.sum(x[:, :ct_size] ^ x[:, ct_size:], axis=0) / len(x) - 0.5
    crit = critical_values(bonferroni_correction=ct_size)
    filtered = np.arange(ct_size)[(biases > crit[1]) + (biases < crit[0])]

    fig, ax = plt.subplots()
    ax.plot(filtered, biases[filtered], "x", color="#17365c")
    ax.axhline(y=crit[0], color="#8dae10", linestyle="-")
    ax.axhline(y=crit[1], color="#8dae10", linestyle="-")
    ax.set_ylabel("Bias")
    ax.set_xlabel("Bit")

    plt.tight_layout()

    plt.savefig(save_as)
    plt.clf()


if __name__ == "__main__":
    # Generate encryption data and plot biases
    x, _ = make_train_data(n_samples, chacha, [0, 0, 0, 0x8000], calc_back=1, y=1)
    eval_biases(x, "biases_chacha.svg")

    x, _ = make_train_data(n_samples, katan, [0x2000], calc_back=2, y=1)
    eval_biases(x, "biases_katan.svg")

    x, _ = make_train_data(n_samples, present, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0], y=1)
    eval_biases(x, "biases_present.svg")

    x, _ = make_train_data(n_samples, simon, [0, 0x40], calc_back=1, y=1)
    eval_biases(x, "biases_simon.svg")

    x, _ = make_train_data(n_samples, skinny, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0], calc_back=2, y=1)
    eval_biases(x, "biases_skinny.svg")

    x, _ = make_train_data(n_samples, speck, [0x40, 0], calc_back=1, y=1)
    eval_biases(x, "biases_speck.svg")


