from tensorflow.keras.models import load_model

from cipher.chacha import ChaCha
from make_train_data import make_train_data, make_real_differences_data, make_mult_pairs_data
from eval import evaluate, evaluate_mult_pairs

n_samples = 5*10**6
n_samples_mult_pairs = 10**6

net = load_model('nets/chacha_3_rounds.h5')
chacha = ChaCha(n_rounds=3)

in_diff = [0, 0, 0, 0x8000]

if __name__ == "__main__":
    print("### Evaluating 3-round ChaCha neural distinguisher on the usual data distribution ###")
    x, y = make_train_data(n_samples, chacha, in_diff, calc_back=1)
    evaluate(net, x, y)

    print("### Evaluating 3-round ChaCha neural distinguisher in the real difference experiment setting ###")
    x, y = make_real_differences_data(n_samples, chacha, in_diff, calc_back=1)
    evaluate(net, x, y)

    print("### Combining scores of 3-round ChaCha neural distinguisher under independence assumption ###")
    for pairs in [1, 2, 4]:
        print(f'{pairs} pairs:')
        x, y = make_mult_pairs_data(n_samples_mult_pairs, chacha, in_diff, calc_back=1, n_pairs=pairs)
        evaluate_mult_pairs(net, chacha, x, y, n_pairs=pairs)
