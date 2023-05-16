from tensorflow.keras.models import load_model

from cipher.simon import Simon
from make_data import make_train_data, make_real_differences_data, make_mult_pairs_data
from eval import evaluate, evaluate_mult_pairs

n_samples = 10**7
n_samples_mult_pairs = 10**6

net = load_model('nets/simon_9_rounds.h5')
net_calc_back = load_model('nets/simon_9_rounds_calc_back.h5')
net10 = load_model('nets/simon_10_rounds.h5')
net11 = load_model('nets/simon_11_rounds.h5')

simon = Simon(n_rounds=9)
simon_free_key = Simon(n_rounds=9, use_key_schedule=False)
simon10 = Simon(n_rounds=10)
simon11 = Simon(n_rounds=11)

in_diff = [0, 0x40]

if __name__ == "__main__":
    print("### Evaluating 9-round Simon neural distinguishers on the usual data distribution ###")
    print("Without backwards calculation:")
    x, y = make_train_data(n_samples, simon, in_diff)
    evaluate(net, x, y)
    print("With backwards calculation:")
    x, y = make_train_data(n_samples, simon, in_diff, calc_back=1)
    evaluate(net_calc_back, x, y)

    print("### Evaluating 10-round Simon neural distinguisher on the usual data distribution ###")
    x, y = make_train_data(n_samples, simon10, in_diff)
    evaluate(net10, x, y)

    print("### Evaluating 11-round Simon neural distinguisher on the usual data distribution ###")
    x, y = make_train_data(n_samples, simon11, in_diff)
    evaluate(net11, x, y)

    print("### Evaluating 9-round Simon neural distinguishers in the real difference experiment setting ###")
    print("Without backwards calculation:")
    x, y = make_real_differences_data(n_samples, simon, in_diff)
    evaluate(net, x, y)
    print("With backwards calculation:")
    x, y = make_real_differences_data(n_samples, simon, in_diff, calc_back=1)
    evaluate(net_calc_back, x, y)

    print("### Evaluating 9-round Simon neural distinguisher using the free key schedule ###")
    x, y = make_train_data(n_samples, simon_free_key, in_diff)
    evaluate(net, x, y)

    print("### Combining scores of 9-round Simon neural distinguishers under independence assumption ###")
    print("Without backwards calculation:")
    for pairs in [1, 2, 4, 8, 16, 32, 64]:
        print(f'{pairs} pairs:')
        x, y = make_mult_pairs_data(n_samples_mult_pairs, simon, in_diff, n_pairs=pairs)
        evaluate_mult_pairs(net, simon, x, y, n_pairs=pairs)
    print("With backwards calculation:")
    for pairs in [1, 2, 4, 8, 16, 32, 64]:
        print(f'{pairs} pairs:')
        x, y = make_mult_pairs_data(n_samples_mult_pairs, simon, in_diff, calc_back=1, n_pairs=pairs)
        evaluate_mult_pairs(net_calc_back, simon, x, y, n_pairs=pairs)

    print("### Combining scores for 10- and 11-round Simon ###")
    print("10 rounds:")
    for pairs in [1, 2, 4, 8, 16, 32, 64]:
        print(f'{pairs} pairs:')
        x, y = make_mult_pairs_data(n_samples_mult_pairs, simon10, in_diff, n_pairs=pairs)
        evaluate_mult_pairs(net10, simon10, x, y, n_pairs=pairs)
    print("11 rounds:")
    for pairs in [1, 2, 4, 8, 16, 32, 64]:
        print(f'{pairs} pairs:')
        x, y = make_mult_pairs_data(n_samples_mult_pairs, simon11, in_diff, n_pairs=pairs)
        evaluate_mult_pairs(net11, simon11, x, y, n_pairs=pairs)
