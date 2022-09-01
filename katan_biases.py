import numpy as np
import click

from tensorflow.keras.models import load_model, Model
from cipher.katan import Katan
from make_train_data import make_train_data
from scipy import stats
from os import urandom

from sklearn.linear_model import Ridge

from tqdm import tqdm


def evaluate_model(model, n_samples, nr, diff, verbose=False):
    """Evaluate the model on real and random ciphertext pairs"""
    katan = Katan(n_rounds=nr)
    X,Y = make_train_data(n_samples, katan, diff, calc_back=2)
    model_help = Model(inputs=model.input, outputs=model.layers[-2].output)
    Z = model_help.predict(X, batch_size=5000)
    clf = Ridge()
    clf.fit(Z, Y)
    X,Y = make_train_data(n_samples, katan, diff, calc_back=2)
    Z = model_help.predict(X, batch_size=5000)
    pred = clf.predict(Z)
    acc = np.mean((pred > 0.5) == Y)
    m1 = np.mean(pred[Y==1])
    m0 = np.mean(pred[Y==0])
    s0 = np.std(pred[Y==0])
    delta = np.sqrt(np.sum(Y==1)) * (m1 - m0) / s0
    p_value = 1 - stats.norm.cdf(delta)
    if verbose:
        print("Accuracy: ", acc)
        print(f'Results for {nr} rounds, input difference {diff}, model {model.name}:')
        print(f'Mean of random responses: {m0}')
        print(f'Mean of real responses: {m1}')
        print(f'Standard deviation of random responses: {s0}')
        print(f'For sample size {n_samples}, the observed value of the real-distribution mean is {delta:.4f} standard deviations above the random distribution mean response.')
        print(f'P-value: {p_value}')
        print('The p-value gives the estimated likelihood of obtaining a mean response at least as large as the observed value for the real distribution by sampling from the random distribution.')
    return p_value


@click.command()
@click.option('--n_samples', default=10**6, help='Number of samples to evaluate on')
@click.option('--nr', default=71, help='Number of rounds')
@click.option('--diff', multiple=True, default=[0x2000], help='Input difference')
@click.option('--model-path', default='./nets/katan_60_rounds.h5', help='Path to model')
def main(n_samples, nr, diff, model_path):
    """Evaluate the model on real and random ciphertext pairs"""
    print('Parameters:')
    print(f'n_samples: {n_samples}')
    print(f'Number of rounds: {nr}')
    print(f'Input difference: {[hex(x) for x in diff]}')
    print(f'Model path: {model_path}')
    print('Starting tests...')
    model = load_model(model_path)
    evaluate_model(model, n_samples, nr, diff, verbose=True)

if __name__ == '__main__':
    main()




