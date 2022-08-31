from tensorflow.keras.models import load_model

from cipher.present import Present
from make_train_data import make_train_data, make_real_differences_data, make_mult_pairs_data
from eval import evaluate, evaluate_mult_pairs

n_samples = 10**7
n_samples_mult_pairs = 10**6

net6 = load_model('nets/present_6_rounds.h5')
net6_csyy22 = load_model('CSYY22/deep_present/saved_model/6_distinguisher.h5')
net7 = load_model('nets/present_7_rounds.h5')
net7_csyy22 = load_model('CSYY22/deep_present/saved_model/7_distinguisher.h5')

present6 = Present(n_rounds=6)
present7 = Present(n_rounds=7)
present_free_key = Present(n_rounds=7, use_key_schedule=False)

in_diff = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xd, 0, 0, 0, 0, 0]

if __name__ == "__main__":
    print("### Evaluating Present neural distinguishers on the usual data distribution ###")
    print("6 rounds:")
    x, y = make_train_data(n_samples, present6, in_diff)
    evaluate(net6, x, y)
    print("7 rounds:")
    x, y = make_train_data(n_samples, present7, in_diff)
    evaluate(net7, x, y)

    print("### Evaluating Present neural distinguishers in the real difference experiment setting ###")
    print("6 rounds:")
    x, y = make_real_differences_data(n_samples, present6, in_diff)
    evaluate(net6, x, y)
    print("7 rounds:")
    x, y = make_real_differences_data(n_samples, present7, in_diff)
    evaluate(net7, x, y)

    print("### Evaluating 7-round Present neural distinguisher using the free key schedule ###")
    x, y = make_train_data(n_samples, present_free_key, in_diff)
    evaluate(net7, x, y)

    print("### Combining scores of Present neural distinguishers under independence assumption ###")
    print("6 rounds:")
    for pairs in [1, 2, 4, 8, 16]:
        print(f'{pairs} pairs:')
        x, y = make_mult_pairs_data(n_samples_mult_pairs, present6, in_diff, n_pairs=pairs)
        evaluate_mult_pairs(net6, present6, x, y, n_pairs=pairs)
    print("7 rounds:")
    for pairs in [1, 2, 4, 8, 16]:
        print(f'{pairs} pairs:')
        x, y = make_mult_pairs_data(n_samples_mult_pairs, present7, in_diff, n_pairs=pairs)
        evaluate_mult_pairs(net7, present7, x, y, n_pairs=pairs)

    print("### Combining scores of singele pair Present neural distinguishers from CSYY22 under independence assumption ###")
    print("6 rounds:")
    for pairs in [1, 2, 4, 8, 16]:
        print(f'{pairs} pairs:')
        x, y = make_mult_pairs_data(n_samples_mult_pairs, present6, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9], n_pairs=pairs)
        evaluate_mult_pairs(net6_csyy22, present6, x, y, n_pairs=pairs)
    print("7 rounds:")
    for pairs in [1, 2, 4, 8, 16]:
        print(f'{pairs} pairs:')
        x, y = make_mult_pairs_data(n_samples_mult_pairs, present7, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9], n_pairs=pairs)
        evaluate_mult_pairs(net7_csyy22, present7, x, y, n_pairs=pairs)

    print("### Evaluating Present neural distinguishers on more rounds than they were trained on ###")
    print("Trained for 6 rounds:")
    for r in range(7, 11):
        print(f'Evaluation on {r}-round Present:')
        present = Present(n_rounds=r)
        x, y = make_train_data(n_samples, present, in_diff)
        evaluate(net6, x, y)
    print("Trained for 7 rounds:")
    for r in range(8, 11):
        print(f'Evaluation on {r}-round Present:')
        present = Present(n_rounds=r)
        x, y = make_train_data(n_samples, present, in_diff)
        evaluate(net7, x, y)
