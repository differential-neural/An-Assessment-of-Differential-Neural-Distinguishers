from tensorflow.keras.models import load_model
import gc

from cipher.speck import Speck
from cipher.simon import Simon
from make_data import make_mult_pairs_advantage_data
from eval import evaluate, evaluate_mult_pairs

n_samples = 10**7

# define input difference
in_diff_speck = [0x40, 0]
in_diff_simon = [0, 0x40]


def adv(accuracy):
    return 2*(accuracy - 0.5)


if __name__ == "__main__":
    print("### Evaluating multi-pair advantage for Speck neural distinguishers from ZWWW22 ###")
    for r in [7, 8]:
        print(f"{r} rounds")
        speck = Speck(n_rounds=r)
        for pairs in [2, 4, 8, 16]:
            print(f"{pairs} pairs")
            net = load_model(
                f'ZWWW22/Speck3264/key_recoveray_attack/our_train_net/model_{r}r_depth5_num_epochs20_pairs{pairs}.h5'
            )
            x, y = make_mult_pairs_advantage_data(
                n_samples, speck, in_diff_speck, n_pairs=pairs, data_format="ZWWW22", combine=True
            )
            acc = evaluate(net, x, y)
            del x, y
            gc.collect()
            print(f"Multi-pair advantage: {adv(acc)}")

    print("### Evaluating multi-pair advantage for our 7-round Speck combined-response distinguisher ###")
    speck = Speck(n_rounds=7)
    net_small = load_model('nets/speck_7_rounds_small.h5')
    for pairs in [2, 4, 8, 16]:
        print(f"{pairs} pairs")
        x, y = make_mult_pairs_advantage_data(
            n_samples, speck, in_diff_speck, n_pairs=pairs, data_format=None, combine=True
        )
        acc = evaluate_mult_pairs(net_small, speck, x, y, n_pairs=pairs)
        del x, y
        gc.collect()
        print(f"Multi-pair advantage: {adv(acc)}")

    print("### Evaluating multi-pair advantage for Simon neural distinguishers from LLS+23 ###")
    for r in [9, 10, 11, 12]:
        print(f"{r} rounds and 8 pairs")
        simon = Simon(n_rounds=r)
        net = load_model(f'LLS+23/ND_Simon32_{r}R{"" if r != 12 else "_stage"}.h5')
        x, y = make_mult_pairs_advantage_data(
            n_samples, simon, in_diff_simon, n_pairs=8, data_format="LLS+23",  combine=True
        )
        acc = evaluate(net, x, y)
        del x, y
        gc.collect()
        print(f"Difference in accuracy: {adv(acc)}")
