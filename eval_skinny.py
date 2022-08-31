from tensorflow.keras.models import load_model

from cipher.skinny import Skinny
from make_train_data import make_train_data, make_real_differences_data, make_mult_pairs_data
from eval import evaluate, evaluate_mult_pairs

n_samples = 10**7
n_samples_mult_pairs = 10**6

net = load_model('nets/skinny_7_rounds.h5')

skinny = Skinny(n_rounds=7)
skinny_free_key = Skinny(n_rounds=7, use_key_schedule=False)

in_diff = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0]

if __name__ == "__main__":
    print("### Evaluating 7-round Skinny neural distinguisher on the usual data distribution ###")
    x, y = make_train_data(n_samples, skinny, in_diff, calc_back=2)
    evaluate(net, x, y)

    print("### Evaluating 7-round Skinny neural distinguisher in the real difference experiment setting ###")
    x, y = make_real_differences_data(n_samples, skinny, in_diff, calc_back=2)
    evaluate(net, x, y)

    print("### Evaluating 7-round Skinny neural distinguisher using the free key schedule ###")
    x, y = make_train_data(n_samples, skinny_free_key, in_diff, calc_back=2)
    evaluate(net, x, y)

    print("### Combining scores of 7-round Skinny neural distinguisher under independence assumption ###")
    for pairs in [1, 2, 4, 8, 16]:
        print(f'{pairs} pairs:')
        x, y = make_mult_pairs_data(n_samples_mult_pairs, skinny, in_diff, calc_back=2, n_pairs=pairs)
        evaluate_mult_pairs(net, skinny, x, y, n_pairs=pairs)
