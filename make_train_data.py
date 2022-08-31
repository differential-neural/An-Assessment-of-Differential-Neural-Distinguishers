from os import urandom
import numpy as np


def convert_to_binary(arr, n_words, word_size):
    """
    Converts a ciphertext pair to an array of bits
    :param arr: Ciphertext pair
    :param n_words: Number of word in each ciphertext
    :param word_size: Size of one word (in bits)
    :return: 
    """
    sample_len = 2 * n_words * word_size
    n_samples = len(arr[0])
    x = np.zeros((sample_len, n_samples), dtype=np.uint8)
    for i in range(sample_len):
        index = i // word_size
        offset = word_size - (i % word_size) - 1
        x[i] = (arr[index] >> offset) & 1
    x = x.transpose()
    return x


def make_train_data(n_samples, cipher, diff, calc_back=0, y=None):
    """
    Generates data for the differential scenario
    :param n_samples: The number of samples
    :param cipher: A cipher object used for encryption
    :param diff: The plaintext difference
    :param calc_back: The variant of calculating back (0 means not calculating back)
    :param y: The label to use for all data. 'None' means random labels
    :return: Training/validation samples
    """
    # generate labels
    if y is None:
        y = np.frombuffer(urandom(n_samples), dtype=np.uint8) & 1
    elif y == 0 or y == 1:
        y = np.array([y for _ in range(n_samples)])
    # draw keys and plaintexts
    keys = cipher.draw_keys(n_samples)
    pt0 = cipher.draw_plaintexts(n_samples)
    pt1 = pt0 ^ np.array(diff, dtype=cipher.word_dtype)[:, np.newaxis]
    # replace plaintexts in pt1 with random ones if label is 0
    num_rand_samples = np.sum(y == 0)
    pt1[:, y == 0] = cipher.draw_plaintexts(num_rand_samples)
    # encrypt
    ct0 = cipher.encrypt(pt0, keys)
    ct1 = cipher.encrypt(pt1, keys)
    if calc_back != 0:
        # Note that the plaintext gets used for ChaCha only
        ct0 = cipher.calc_back(ct0, pt0, calc_back)
        ct1 = cipher.calc_back(ct1, pt1, calc_back)
    # convert to binary and return
    x = convert_to_binary(np.concatenate((ct0, ct1), axis=0), cipher.get_n_words(), cipher.get_word_size())
    return x, y


def make_mult_pairs_data(n_samples, cipher, diff, calc_back=0, n_pairs=1):
    """
    Generates data for the differential scenario using multiple pairs
    :param n_samples: The number of samples
    :param cipher: A cipher object used for encryption
    :param diff: The plaintext difference
    :param calc_back: The variant of calculating back (0 means not calculating back)
    :param n_pairs: The number of ciphertext pairs that should make up one sample
    :return: Training/validation samples
    """
    # generate labels
    y = np.frombuffer(urandom(n_samples), dtype=np.uint8) & 1
    # repeat labels for the pairs to combine
    y_atomic = np.repeat(y, n_pairs)
    # generate data
    keys = cipher.draw_keys(n_samples * n_pairs)
    pt0 = cipher.draw_plaintexts(n_samples * n_pairs)
    pt1 = pt0 ^ np.array(diff, dtype=cipher.word_dtype)[:, np.newaxis]
    num_rand_samples = np.sum(y_atomic == 0)
    pt1[:, y_atomic == 0] = cipher.draw_plaintexts(num_rand_samples)
    # encrypt
    ct0 = cipher.encrypt(pt0, keys)
    ct1 = cipher.encrypt(pt1, keys)
    if calc_back != 0:
        # Note that the plaintext gets used for ChaCha only
        ct0 = cipher.calc_back(ct0, pt0, calc_back)
        ct1 = cipher.calc_back(ct1, pt1, calc_back)
    # convert into an array of zero and ones
    x = convert_to_binary(np.concatenate((ct0, ct1), axis=0), cipher.get_n_words(), cipher.get_word_size())
    x = x.reshape((-1, n_pairs, n_samples))
    return x, y


def make_real_differences_data(n_samples, cipher, diff, calc_back=0):
    """
    Generates data for the real difference experiment
    :param n_samples: The number of samples
    :param cipher: A cipher object used for encryption
    :param diff: The plaintext difference
    :param calc_back: The variant of calculating back (0 means not calculating back)
    :return: Samples for the real difference experiment
    """
    # generate labels
    y = np.frombuffer(urandom(n_samples), dtype=np.uint8) & 1
    # draw keys and plaintexts
    keys = cipher.draw_keys(n_samples)
    pt0 = cipher.draw_plaintexts(n_samples)
    pt1 = pt0.copy() ^ np.array(diff, dtype=cipher.word_dtype)[:, np.newaxis]
    # encrypt
    ct0 = cipher.encrypt(pt0, keys)
    ct1 = cipher.encrypt(pt1, keys)
    if calc_back != 0:
        ct0 = cipher.calc_back(ct0, pt0, calc_back)
        ct1 = cipher.calc_back(ct1, pt1, calc_back)
    if type(ct0) is tuple or type(ct0) is list:  # Convert ciphertexts into numpy arrays in case they aren't already
        ct0 = np.array(ct0, dtype=cipher.word_dtype)
        ct1 = np.array(ct1, dtype=cipher.word_dtype)
    # blind samples with label 0
    num_rand_samples = np.sum(y == 0)
    blinding_val = cipher.draw_ciphertexts(num_rand_samples)
    ct0[:, y == 0] ^= blinding_val
    ct1[:, y == 0] ^= blinding_val
    # convert to binary and return
    x = convert_to_binary(np.concatenate((ct0, ct1), axis=0), cipher.get_n_words(), cipher.get_word_size())
    return x, y
