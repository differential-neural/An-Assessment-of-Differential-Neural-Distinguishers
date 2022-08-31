import numpy as np
import click

from tensorflow.keras.models import load_model
from cipher.present import Present
from make_train_data import make_train_data
from scipy import stats
from os import urandom

from tqdm import tqdm

def gen_random(n):
    """Quickly generate random ciphertext pairs"""
    X = np.frombuffer(urandom(128 * n), dtype=np.uint8) & 1
    X = X.reshape((n, 128))
    return X

def responses_random(n, model, step_size=10**6, verbose=False):
    """Generate random ciphertext pairs and use the model to predict the response"""	
    pred = np.zeros(n, dtype=np.float32)
    for i in tqdm(range(0, n, step_size), disable=not verbose):
        steps_remaining = n - i
        batch = min(steps_remaining, step_size)
        T = gen_random(batch)
        pred[i:i+batch] = model.predict(T, batch_size=5000).flatten()
    return pred

def responses_real_present(n, nr, diff, model, step_size=10**6, verbose=False):
    """Generate real ciphertext pairs and use the model to predict the response"""
    pred = np.zeros(n, dtype=np.float32)
    cipher = Present(nr)
    for i in tqdm(range(0, n, step_size), disable=not verbose):
        steps_remaining = n - i
        batch = min(steps_remaining, step_size)
        T, _ = make_train_data(batch, cipher, diff, y=1)
        pred[i:i+batch] = model.predict(T, batch_size=5000).flatten()
    return pred

def evaluate_model(model, n_real, n_random, nr, diff, verbose=False, random_response_path=None):
    """Evaluate the model on real and random ciphertext pairs"""
    if verbose:
        print("Evaluating model on real ciphertext pairs...")
    pred_real = responses_real_present(n_real, nr, diff, model, verbose=verbose)
    if verbose:
        print("Evaluating model on random ciphertext pairs...")
    m0, s0 = None, None
    if random_response_path is not None:
        try:
            m0, s0 = np.load(random_response_path)
        except FileNotFoundError:
            pass
    if m0 is None or s0 is None:
        pred_random = responses_random(n_random, model, verbose=verbose)
        m0 = np.mean(pred_random)
        s0 = np.std(pred_random)
        if random_response_path is not None:
            np.save(random_response_path, (m0, s0))
    m1 = np.mean(pred_real)
    delta = np.sqrt(n_real) * (m1 - m0) / s0
    p_value = 1 - stats.norm.cdf(delta)
    if verbose: 
        print(f'Results for {nr} rounds, input difference {diff}, model {model.name}:')
        print(f'Mean of random responses: {m0}')
        print(f'Mean of real responses: {m1}')
        print(f'Standard deviation of random responses: {s0}')
        print(f'For sample size {n_real}, the observed value of the real-distribution mean is {delta:.4f} standard deviations above the random distribution mean response.')
        print(f'P-value: {p_value}')
        print('The p-value gives the estimated likelihood of obtaining a mean response at least as large as the observed value for the real distribution by sampling from the random distribution.')
    return p_value

@click.command()
@click.option('--n_real', default=10**7, help='Number of real ciphertext pairs to evaluate on')
@click.option('--n_random', default=10**8, help='Number of random ciphertext pairs to evaluate on')
@click.option('--nr', default=10, help='Number of rounds')
@click.option('--diff', multiple=True, default=[0,0,0,0,0,0,0,0,0,0,0xd,0,0,0,0,0], help='Input difference')
@click.option('--model-path', default='./nets/present_7_rounds.h5', help='Path to model')
@click.option('--random-response-path', default='random_response_present_7_rounds.npy', help='Path to random response file')
def main(n_real, n_random, nr, diff, model_path, random_response_path):
    """Evaluate the model on real and random ciphertext pairs"""
    print('Parameters:')
    print(f'n_real: {n_real}')
    print(f'n_random: {n_random}')
    print(f'Number of rounds: {nr}')
    print(f'Input difference: {[hex(x) for x in diff]}')
    print(f'Model path: {model_path}')
    print('Starting tests...')
    model = load_model(model_path)
    evaluate_model(model, n_real, n_random, nr, diff, verbose=True, random_response_path=random_response_path)

if __name__ == '__main__':
    main()




