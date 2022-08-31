import numpy as np
from os import urandom

from cipher.abstract_cipher import AbstractCipher
from exception.NotImplementedException import NotImplementedException


class ChaCha(AbstractCipher):

    def __init__(self, n_rounds=20, use_key_schedule=True):
        """
        Initializes a ChaCha cipher object
        :param n_rounds: The number of rounds used for de-/encryption
        :param use_key_schedule: Whether to use the key schedule or independent round keys
        """
        if not use_key_schedule:
            raise Exception('ERROR: use_key_schedule=False not supported for ChaCha')
        super(ChaCha, self).__init__(
            n_rounds, word_size=32, n_words=16, n_main_key_words=8,
            n_round_key_words=None, use_key_schedule=use_key_schedule
        )

    def quarter_round(self, state, a, b, c, d):
        """
        Applies the ChaCha quarter round to state
        :param state: The state
        :param a: Index of input a
        :param b: Index of input b
        :param c: Index of input c
        :param d: Index of input d
        :return: The manipulated state
        """
        state[a] += state[b]
        state[d] ^= state[a]
        state[d] = self.rol(state[d], 16)
        state[c] += state[d]
        state[b] ^= state[c]
        state[b] = self.rol(state[b], 12)
        state[a] += state[b]
        state[d] ^= state[a]
        state[d] = self.rol(state[d], 8)
        state[c] += state[d]
        state[b] ^= state[c]
        state[b] = self.rol(state[b], 7)
        return state

    def encrypt_one_round(self, p, k, rc=None):
        """
        Round function of the cipher
        :param p: The plaintext
        :param k: The round key
        :param rc: The round constant
        :return: The one round encryption of p using key k
        """
        s = p  # p is the state which already contains the key. Hence, k is not used
        if rc == 0:  # This is a column round
            s = self.quarter_round(s, 0, 4, 8, 12)
            s = self.quarter_round(s, 1, 5, 9, 13)
            s = self.quarter_round(s, 2, 6, 10, 14)
            s = self.quarter_round(s, 3, 7, 11, 15)
        else:  # This is a diagonal round
            s = self.quarter_round(s, 0, 5, 10, 15)
            s = self.quarter_round(s, 1, 6, 11, 12)
            s = self.quarter_round(s, 2, 7, 8, 13)
            s = self.quarter_round(s, 3, 4, 9, 14)
        return s

    @staticmethod
    def build_initial_state(nonce_and_counter, key):
        """
        :param nonce_and_counter: Nonce and Counter
        :param key: The key
        :return: The initial state
        """
        n_samples = len(nonce_and_counter[0])
        constants = np.array([0x61707865, 0x3320646e, 0x79622d32, 0x6b206574], dtype=np.uint32)
        constants = np.repeat(np.expand_dims(constants, axis=1), n_samples, axis=1)  # Repeat for every sample
        return np.concatenate((constants, key, nonce_and_counter), axis=0)

    def encrypt(self, p, keys):
        """
        Generate the key stream. Note that, since this framework was designed with block ciphers in mind,
        the usage of the parameters is slightly different.
        :param p: The data an attacker is able to control, i.e. the counter and the nonce
        :param keys: The key that is used to generate the key stream block
        :return: The key stream block
        """
        state = self.build_initial_state(p, keys)
        working_state = state.copy()

        # Iterate rounds
        working_state = super(ChaCha, self).encrypt(working_state, [None for i in range(self.n_rounds)])

        # Calculate final key stream block and return
        return state + working_state

    def decrypt_one_round(self, c, k, rc=None):
        raise NotImplementedException("decrypt", "ChaCha")

    def calc_back(self, c, p=None, variant=1):
        """
        Revert deterministic parts of the round function
        :param c: The ciphertext
        :param p: The initial plaintext
        :param variant: Select the variant of how to calculate back (default is 1; 0 means not calculating back)
        :return: The inner state after reverting the deterministic transformation at the end of the encryption process
        """
        if variant == 0:
            return c
        if variant != 1:
            raise Exception(f'ERROR: Variant {variant} of calculating back is not implemented')
        if p is None:
            raise Exception("ERROR: Nonce and counter are needed in order to calculate state back")
        # Build known initial state
        zero_key = np.zeros((8, len(p[0])), dtype=np.uint32)
        known_initial_state = self.build_initial_state(p, zero_key)
        # Revert addition of initial state to working state for counter, constant and nonce
        return c - known_initial_state

    def get_rc(self, r):
        """
        :param r: The round
        :return: The round constant for round r
        """
        return r % 2  # To distinguish between column and diagonal rounds

    def draw_plaintexts(self, n_samples):
        """
        Draw nonce and counter, i.e. 4 words
        :param n_samples: How many plaintexts to draw
        :return: An array of plaintexts
        """
        return np.reshape(
            np.frombuffer(urandom(4 * 4 * n_samples), dtype=self.word_dtype),
            (4, n_samples)
        )

    def key_schedule(self, key):
        """
        Applies the key schedule
        :param key: The key
        :return: A list of round keys
        """
        return key  # There exist no round keys fo ChaCha and the main key has already the shape we need

    @staticmethod
    def get_test_vectors():
        """
        Test vectors from https://tools.ietf.org/pdf/rfc7539.pdf, page 9
        :return: Returns the test vectors used for verifying the correct implementation of the cipher as a list of
            tuples of the form (cipher, plaintext, key, ciphertext), where cipher is an instance of self that will
            be used to verify the test vector
        """
        chacha = ChaCha()
        key = np.array([[0x03020100], [0x07060504], [0x0b0a0908], [0x0f0e0d0c],
                        [0x13121110], [0x17161514], [0x1b1a1918], [0x1f1e1d1c]], dtype=np.uint32)
        nonce_and_counter = np.array([[0x00000001], [0x09000000], [0x4a000000], [0x00000000]], dtype=np.uint32)
        stream = np.array([
            [0xe4e7f110], [0x15593bd1], [0x1fdd0f50], [0xc47120a3],
            [0xc7f4d1c7], [0x0368c033], [0x9aaa2204], [0x4e6cd4c3],
            [0x466482d2], [0x09aa9f07], [0x05d7c214], [0xa2028bd9],
            [0xd19c12b5], [0xb94e16de], [0xe883d0cb], [0x4e3c50a2]
        ], dtype=np.uint32)
        return [(chacha, nonce_and_counter, key, stream)]

