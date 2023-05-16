from tensorflow.keras.models import load_model

from cipher.katan import Katan
from make_data import make_train_data, make_real_differences_data, make_mult_pairs_data
from eval import evaluate, evaluate_mult_pairs

n_samples = 10**7
n_samples_mult_pairs = 10**6

nets = {
    60: load_model('nets/katan_60_rounds.h5'),
    61: load_model('nets/katan_61_rounds.h5'),
    62: load_model('nets/katan_62_rounds.h5'),
    63: load_model('nets/katan_63_rounds.h5'),
    64: load_model('nets/katan_64_rounds.h5'),
    65: load_model('nets/katan_65_rounds.h5'),
    66: load_model('nets/katan_66_rounds.h5'),
}

in_diff = [0x2000]

if __name__ == "__main__":
    print("### Evaluating Katan neural distinguishers on the usual data distribution ###")
    for r, net in nets.items():
        print(f'{r} rounds:')
        katan = Katan(n_rounds=r)
        x, y = make_train_data(n_samples, katan, in_diff, calc_back=2)
        evaluate(net, x, y)

    print("### Evaluating Katan neural distinguishers in the real difference experiment setting ###")
    for r, net in nets.items():
        print(f'{r} rounds:')
        katan = Katan(n_rounds=r)
        x, y = make_real_differences_data(n_samples, katan, in_diff, calc_back=2)
        evaluate(net, x, y)

    print("### Evaluating 66-round Katan neural distinguisher using the free key schedule ###")
    katan_free_key = Katan(n_rounds=66, use_key_schedule=False)
    x, y = make_train_data(n_samples, katan_free_key, in_diff, calc_back=2)
    evaluate(nets[66], x, y)

    print("### Combining scores of Katan neural distinguishers under independence assumption ###")
    for r, net in nets.items():
        print(f'{r} rounds:')
        katan = Katan(n_rounds=r)
        for pairs in [1, 2, 4, 8, 16]:
            print(f'{pairs} pairs:')
            x, y = make_mult_pairs_data(n_samples_mult_pairs, katan, in_diff, calc_back=2, n_pairs=pairs)
            evaluate_mult_pairs(net, katan, x, y, n_pairs=pairs)
