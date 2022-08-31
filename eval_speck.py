from tensorflow.keras.models import model_from_json, load_model

from cipher.speck import Speck
from make_train_data import make_train_data, make_real_differences_data, make_mult_pairs_data
from eval import evaluate, evaluate_mult_pairs

n_samples = 10**7
n_samples_mult_pairs = 10**6

# load Gohr's distinguishers
json_file = open('Goh19/single_block_resnet.json', 'r')
json_model = json_file.read()

net5 = model_from_json(json_model)
net6 = model_from_json(json_model)
net7 = model_from_json(json_model)
net8 = model_from_json(json_model)

net5.load_weights('Goh19/net5_small.h5')
net6.load_weights('Goh19/net6_small.h5')
net7.load_weights('Goh19/net7_small.h5')
net8.load_weights('Goh19/net8_small.h5')

# load distinguishers from CSYY22
net5_csyy22 = load_model('CSYY22/deep_speck/saved_model/5_distinguisher.h5')
net6_csyy22 = load_model('CSYY22/deep_speck/saved_model/6_distinguisher.h5')
net7_csyy22 = load_model('CSYY22/deep_speck/saved_model/7_distinguisher.h5')

# load our distinguisher
net = load_model('nets/speck_7_rounds.h5')
net_small = load_model('nets/speck_7_rounds_small.h5')

# initialize ciphers
speck5 = Speck(n_rounds=5)
speck6 = Speck(n_rounds=6)
speck7 = Speck(n_rounds=7)
speck_free_key = Speck(n_rounds=7, use_key_schedule=False)
speck8 = Speck(n_rounds=8)

# define input difference
in_diff = [0x40, 0]

if __name__ == "__main__":
    print("### Evaluating Speck neural distinguishers on the usual data distribution ###")
    print('5 rounds (Gohr):')
    x, y = make_train_data(n_samples, speck5, in_diff)
    evaluate(net5, x, y)
    print('6 rounds (Gohr):')
    x, y = make_train_data(n_samples, speck6, in_diff)
    evaluate(net6, x, y)
    print('7 rounds (Gohr):')
    x, y = make_train_data(n_samples, speck7, in_diff)
    evaluate(net7, x, y)
    print('7 rounds (our):')
    x, y = make_train_data(n_samples, speck7, in_diff)
    evaluate(net, x, y)
    print('7 rounds (our - small):')
    x, y = make_train_data(n_samples, speck7, in_diff)
    evaluate(net_small, x, y)
    print('8 rounds (Gohr):')
    x, y = make_train_data(n_samples, speck8, in_diff)
    evaluate(net8, x, y)

    print("### Evaluating Speck neural distinguishers in the real difference experiment setting ###")
    print('5 rounds (Gohr):')
    x, y = make_real_differences_data(n_samples, speck5, in_diff)
    evaluate(net5, x, y)
    print('6 rounds (Gohr):')
    x, y = make_real_differences_data(n_samples, speck6, in_diff)
    evaluate(net6, x, y)
    print('7 rounds (Gohr):')
    x, y = make_real_differences_data(n_samples, speck7, in_diff)
    evaluate(net7, x, y)
    print('7 rounds (our):')
    x, y = make_real_differences_data(n_samples, speck7, in_diff)
    evaluate(net, x, y)
    print('8 rounds (Gohr):')
    x, y = make_real_differences_data(n_samples, speck8, in_diff)
    evaluate(net8, x, y)

    print("### Evaluating 7-round Speck neural distinguisher using the free key schedule ###")
    x, y = make_train_data(n_samples, speck_free_key, in_diff)
    evaluate(net, x, y)

    print("### Combining scores of Speck neural distinguisher under independence assumption ###")
    print('5 rounds:')
    for pairs in [1, 2, 4, 8, 16, 32, 64]:
        print(f'{pairs} pairs (Gohr):')
        x, y = make_mult_pairs_data(n_samples_mult_pairs, speck5, in_diff, n_pairs=pairs)
        evaluate_mult_pairs(net5, speck5, x, y, n_pairs=pairs)
    for pairs in [1, 2, 4, 8, 16, 32, 64]:
        print(f'{pairs} pairs (CSYY22):')
        x, y = make_mult_pairs_data(n_samples_mult_pairs, speck5, in_diff, n_pairs=pairs)
        evaluate_mult_pairs(net5_csyy22, speck5, x, y, n_pairs=pairs)
    print('6 rounds:')
    for pairs in [1, 2, 4, 8, 16, 32, 64]:
        print(f'{pairs} pairs (Gohr):')
        x, y = make_mult_pairs_data(n_samples_mult_pairs, speck6, in_diff, n_pairs=pairs)
        evaluate_mult_pairs(net6, speck6, x, y, n_pairs=pairs)
    for pairs in [1, 2, 4, 8, 16, 32, 64]:
        print(f'{pairs} pairs (CSYY22):')
        x, y = make_mult_pairs_data(n_samples_mult_pairs, speck6, in_diff, n_pairs=pairs)
        evaluate_mult_pairs(net6_csyy22, speck6, x, y, n_pairs=pairs)
    print('7 rounds:')
    for pairs in [1, 2, 4, 8, 16, 32, 64]:
        print(f'{pairs} pairs (Gohr):')
        x, y = make_mult_pairs_data(n_samples_mult_pairs, speck7, in_diff, n_pairs=pairs)
        evaluate_mult_pairs(net7, speck7, x, y, n_pairs=pairs)
    print('7 rounds:')
    for pairs in [1, 2, 4, 8, 16, 32, 64]:
        print(f'{pairs} pairs (CSYY22):')
        x, y = make_mult_pairs_data(n_samples_mult_pairs, speck7, in_diff, n_pairs=pairs)
        evaluate_mult_pairs(net7_csyy22, speck7, x, y, n_pairs=pairs)
    print('7 rounds:')
    for pairs in [1, 2, 4, 8, 16, 32, 64]:
        print(f'{pairs} pairs (our):')
        x, y = make_mult_pairs_data(n_samples_mult_pairs, speck7, in_diff, n_pairs=pairs)
        evaluate_mult_pairs(net, speck7, x, y, n_pairs=pairs)
    print('8 rounds:')
    for pairs in [1, 2, 4, 8, 16, 32, 64]:
        print(f'{pairs} pairs (Gohr):')
        x, y = make_mult_pairs_data(n_samples_mult_pairs, speck8, in_diff, n_pairs=pairs)
        evaluate_mult_pairs(net8, speck8, x, y, n_pairs=pairs)
