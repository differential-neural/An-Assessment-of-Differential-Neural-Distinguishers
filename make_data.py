from os import urandom
import numpy as np

from cipher.simon import Simon
from cipher.speck import Speck


def convert_to_binary(arr, n_words, word_size) -> np.ndarray:
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


def preprocess_samples(ct0, ct1, pt0, pt1, cipher, calc_back=0, data_format=None) -> np.ndarray:
    """
    Preprocesses the samples and returns them as an array of bits
    :param ct0: First numpy array of ciphertexts
    :param ct1: Second numpy array of ciphertexts
    :param pt0: Plaintexts corresponding to ct0
    :param pt1: Plaintexts corresponding to ct1
    :param cipher: A cipher object used for backwards calculation
    :param calc_back: The variant of calculating back (0 means not calculating back)
    :param data_format: Allows to specify a special data format
    """
    if data_format:
        if calc_back != 0:
            raise Exception("Backwards calculation is not compatible with using a specific data format!")
        if data_format == "LLS+23" and type(cipher) is Simon:
            # calculate diff = (\Delta_L^r,\Delta_R^r)
            diff_l = ct0[0] ^ ct1[0]
            diff_r = ct0[1] ^ ct1[1]
            # calculate diff_r_previous = \Delta_R^{r-1}
            previous0 = cipher.calc_back(ct0, pt0, 1)
            previous1 = cipher.calc_back(ct1, pt1, 1)
            diff_r_previous = previous0[1] ^ previous1[1]
            # calculate diff_r_penultimate = p\Delta_R^{r-2}
            penultimate0 = cipher.calc_back(previous0, pt0, 1)
            penultimate1 = cipher.calc_back(previous1, pt1, 1)
            diff_r_penultimate = penultimate0[1] ^ penultimate1[1]
            # merge into (\Delta_L^r, \Delta_R^r, C_l, C_r, C_l', C_r', \Delta_R^{r-1}, p\Delta_R^{r-2})
            return convert_to_binary(
                np.concatenate(((diff_l, diff_r), ct0, ct1, (diff_r_previous, diff_r_penultimate)), axis=0),
                4, cipher.get_word_size()
            )
        elif data_format == "ZWWW22" and type(cipher) is Speck:
            previous0 = cipher.calc_back(ct0, pt0, 1)
            previous1 = cipher.calc_back(ct1, pt1, 1)
            return convert_to_binary(
                np.concatenate(((previous0[1], previous1[1]), ct0, ct1), axis=0), 3, cipher.get_word_size()
            )
        else:
            raise Exception(f"Unknown data format {data_format} for cipher {type(cipher)}")
    if calc_back != 0:
        # note that the plaintext gets used for ChaCha only
        ct0 = cipher.calc_back(ct0, pt0, calc_back)
        ct1 = cipher.calc_back(ct1, pt1, calc_back)
    # convert to binary and return
    return convert_to_binary(np.concatenate((ct0, ct1), axis=0), cipher.get_n_words(), cipher.get_word_size())


def make_train_data(
        n_samples, cipher, diff, calc_back=0, y=None, additional_conditions=None, data_format=None
) -> (np.ndarray, np.ndarray):
    """
    Generates data for the differential scenario
    :param n_samples: The number of samples
    :param cipher: A cipher object used for encryption
    :param diff: The plaintext difference
    :param calc_back: The variant of calculating back (0 means not calculating back)
    :param y: The label to use for all data. 'None' means random labels
    :param data_format: Allows to specify a special data format
    :return: Training/validation samples
    """
    # generate labels
    if y is None:
        y = np.frombuffer(urandom(n_samples), dtype=np.uint8) & 1
    elif y == 0 or y == 1:
        y = np.array([y for _ in range(n_samples)], dtype=np.uint8)
    # draw keys and plaintexts
    keys = cipher.draw_keys(n_samples)
    pt0 = cipher.draw_plaintexts(n_samples)
    if additional_conditions is not None:
        pt0 = additional_conditions(pt0)
    pt1 = pt0 ^ np.array(diff, dtype=cipher.word_dtype)[:, np.newaxis]
    # replace plaintexts in pt1 with random ones if label is 0
    num_rand_samples = np.sum(y == 0)
    pt1[:, y == 0] = cipher.draw_plaintexts(num_rand_samples)
    # encrypt
    ct0 = cipher.encrypt(pt0, keys)
    ct1 = cipher.encrypt(pt1, keys)
    # perform backwards calculation and other preprocessing
    x = preprocess_samples(ct0, ct1, pt0, pt1, cipher, calc_back, data_format)
    return x, y


def make_mult_pairs_data(
        n_samples, cipher, diff, calc_back=0, y=None, n_pairs=1, data_format=None, redraw_key="sample", combine=False
) -> (np.ndarray, np.ndarray):
    """
    Generates data for the differential scenario using multiple pairs
    :param n_samples: The number of samples
    :param cipher: A cipher object used for encryption
    :param diff: The plaintext difference
    :param calc_back: The variant of calculating back (0 means not calculating back)
    :param y: The label to use for all data. 'None' means random labels
    :param n_pairs: The number of ciphertext pairs that should make up one sample
    :param data_format: Allows to specify a special data format
    :param redraw_key: Whether to draw a new key for every pair (redraw_key="pair") or very sample (redraw_key="sample")
    :param combine: Whether to combine those multiple pairs into one sample
    :return: Training/validation samples
    """
    # generate labels
    if y is None:
        y = np.frombuffer(urandom(n_samples), dtype=np.uint8) & 1
    elif y == 0 or y == 1:
        y = np.array([y for _ in range(n_samples)], dtype=np.uint8)
    # repeat labels for the pairs to combine
    y_atomic = np.tile(y, n_pairs)
    # generate data
    if redraw_key == "pair":
        keys = cipher.draw_keys(n_samples * n_pairs)
    elif redraw_key == "sample":
        keys = cipher.draw_keys(n_samples)
        keys = np.tile(keys, n_pairs)
    else:
        raise Exception(f"Unknown option for redraw_key \"{redraw_key}\"")
    pt0 = cipher.draw_plaintexts(n_samples * n_pairs)
    pt1 = pt0 ^ np.array(diff, dtype=cipher.word_dtype)[:, np.newaxis]
    num_rand_samples = np.sum(y_atomic == 0)
    pt1[:, y_atomic == 0] = cipher.draw_plaintexts(num_rand_samples)
    # encrypt
    ct0 = cipher.encrypt(pt0, keys)
    ct1 = cipher.encrypt(pt1, keys)
    # perform backwards calculation and other preprocessing
    x = preprocess_samples(ct0, ct1, pt0, pt1, cipher, calc_back, data_format)
    x = x.reshape((n_pairs, n_samples, -1)).transpose((1, 0, 2))
    if combine:
        x = x.reshape(n_samples, 1, -1)
        x = np.squeeze(x)
    return x, y


def make_real_differences_data(n_samples, cipher, diff, calc_back=0) -> (np.ndarray, np.ndarray):
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


def make_mult_pairs_advantage_data(
        n_samples, cipher, diff, calc_back=0, n_pairs=1, data_format=None, combine=False
) -> (np.ndarray, np.ndarray):
    """
    Generates data to evaluate multi-pair advantage
    :param n_samples: The number of samples
    :param cipher: A cipher object used for encryption
    :param diff: The plaintext difference
    :param calc_back: The variant of calculating back (0 means not calculating back)
    :param n_pairs: The number of ciphertext pairs that should make up one sample
    :param data_format: Allows to specify a special data format
    :param combine: Whether to combine those multiple pairs into one sample
    :return: Training/validation samples
    """
    y = np.frombuffer(urandom(n_samples), dtype=np.uint8) & 1
    # generate default samples
    x, _ = make_mult_pairs_data(
        n_samples, cipher, diff, calc_back=calc_back, y=1, n_pairs=n_pairs,
        data_format=data_format, redraw_key="sample", combine=combine
    )
    # override with independent pairs for y==0
    x[y == 0], _ = make_mult_pairs_data(
        np.sum(y == 0), cipher, diff, calc_back=calc_back, y=1, n_pairs=n_pairs,
        data_format=data_format, redraw_key="pair", combine=combine
    )
    return x, y
