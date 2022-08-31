from abc import ABC, abstractmethod
import numpy as np
from os import urandom

from exception.NotImplementedException import NotImplementedException


class AbstractCipher(ABC):
    """ Abstract cipher class containing all methods a cipher class should implement """

    """ Data types for all the supported word sizes """
    DTYPES = {
        2: np.uint8,
        4: np.uint8,
        8: np.uint8,
        16: np.uint16,
        32: np.uint32
    }

    def __init__(
            self, n_rounds, word_size, n_words, n_main_key_words, n_round_key_words,
            use_key_schedule=True, main_key_word_size=None, round_key_word_size=None
    ):
        """
        Initializes a cipher object
        :param n_rounds: The number of rounds used for de-/encryption
        :param word_size: The size (in bits) of a ciphertext word
        :param n_words: The number of words of one ciphertext
        :param n_main_key_words: The number of words in the main key
        :param n_round_key_words: The number of words in each round key
        :param use_key_schedule: Whether to use the key schedule or independent round keys
        :param main_key_word_size: The size (in bits) of a main key word ('None' means the same as word_size)
        :param round_key_word_size: The size (in bits) of a round key word ('None' means the same as word_size)
        """
        self.n_rounds = n_rounds
        self.word_size = word_size
        self.word_dtype = self.DTYPES.get(self.word_size, None)
        if self.word_dtype is None:
            raise Exception(f'Error: Unexpected word size {self.word_size}')
        self.mask_val = 2 ** self.word_size - 1
        self.n_words = n_words
        self.n_main_key_words = n_main_key_words
        self.n_round_key_words = n_round_key_words
        self.use_key_schedule = use_key_schedule
        self.main_key_word_size = main_key_word_size if main_key_word_size is not None else word_size
        self.main_key_word_dtype = self.DTYPES.get(self.main_key_word_size, None)
        if self.main_key_word_dtype is None:
            raise Exception(f'Error: Unexpected word size {self.main_key_word_size}')
        self.round_key_word_size = round_key_word_size if round_key_word_size is not None else word_size
        self.round_key_word_dtype = self.DTYPES.get(self.round_key_word_size, None)
        if self.round_key_word_dtype is None:
            raise Exception(f'Error: Unexpected word size {self.round_key_word_size}')

    def get_word_size(self):
        """
        :return: The size (in bits) of one word (which could be the size of an s-box or of the right/left side)
        """
        return self.word_size

    def get_n_words(self):
        """
        :return: The number of words in one ciphertext
        """
        return self.n_words

    def get_block_size(self):
        """
        :return: The size (in bits) of one ciphertext
        """
        return self.word_size * self.n_words

    def get_n_rounds(self):
        """
        :return: The number of rounds
        """
        return self.n_rounds

    def set_n_rounds(self, new_n_rounds):
        """
        Sets the number of rounds
        :param new_n_rounds: The new number of rounds
        """
        self.n_rounds = new_n_rounds

    @staticmethod
    def bytes_per_word(word_size):
        """
        :param word_size: The word size (in bits)
        :return: Returns the number of bytes to represent a word of word_size bits
        """
        return word_size // 8 + (1 if (word_size % 8) else 0)

    @abstractmethod
    def encrypt_one_round(self, p, k, rc=None):
        """
        Round function of the cipher
        :param p: The plaintext
        :param k: The round key
        :param rc: The round constant
        :return: The one round encryption of p using key k
        """
        pass

    def encrypt(self, p, keys):
        """
        Encrypt by applying the round function for each given round key
        :param p: The plaintext
        :param keys: A list of round keys
        :return: The encryption of p under the round keys in keys
        """
        state = p
        for i in range(len(keys)):
            state = self.encrypt_one_round(state, keys[i], self.get_rc(i))
        return state

    @abstractmethod
    def decrypt_one_round(self, c, k, rc=None):
        """
        Inverse round function of the cipher
        :param c: The ciphertext
        :param k: The round key
        :param rc: The round constant
        :return: The one round decryption of c using key k
        """
        pass

    def decrypt(self, c, keys):
        """
        Decrypt by applying the inverse round function for each given key
        :param c: The ciphertext
        :param keys: A list of round keys
        :return: The decryption of c under the round keys in keys
        """
        state = c
        for i in range(len(keys) - 1, -1, -1):
            state = self.decrypt_one_round(state, keys[i], self.get_rc(i))
        return state

    @abstractmethod
    def calc_back(self, c, p=None, variant=1):
        """
        Revert deterministic parts of the round function
        :param c: The ciphertext
        :param p: The initial plaintext
        :param variant: Select the variant of how to calculate back (default is 1; 0 means not calculating back)
        :return: The inner state after reverting the deterministic transformation at the end of the encryption process
        """
        pass

    def get_rc(self, r):
        """
        :param r: The round
        :return: The round constant for round r
        """
        return None

    def draw_keys(self, n_samples):
        """
        :param n_samples: How many keys to draw
        :return: An array of keys
        """
        if self.use_key_schedule:
            bytes_per_word = self.bytes_per_word(self.main_key_word_size)
            main_key = np.frombuffer(
                urandom(self.n_main_key_words * bytes_per_word * n_samples), dtype=self.main_key_word_dtype
            ).reshape(self.n_main_key_words, n_samples)
            if self.main_key_word_size < 8:
                # Note: If the word size is greater than 8, it will always fit the dtype for the ciphers we use
                main_key = np.right_shift(main_key, 8 - self.main_key_word_size)
            return self.key_schedule(main_key)
        else:
            bytes_per_word = self.bytes_per_word(self.round_key_word_size)
            round_keys = np.frombuffer(
                urandom(self.n_rounds * self.n_round_key_words * bytes_per_word * n_samples),
                dtype=self.round_key_word_dtype
            ).reshape(self.n_rounds, self.n_round_key_words, n_samples)
            if self.round_key_word_size < 8:
                # Note: If the word size is greater than 8, it will always fit the dtype for the ciphers we use
                round_keys = np.right_shift(round_keys, 8 - self.round_key_word_size)
            return round_keys

    def draw_plaintexts(self, n_samples):
        """
        :param n_samples: How many plaintexts to draw
        :return: An array of plaintexts
        """
        # In most cases the format of the plain- and ciphertexts are the same,
        # so we can return random ciphertexts at this point
        return self.draw_ciphertexts(n_samples)

    def draw_ciphertexts(self, n_samples):
        """
        :param n_samples: How many ciphertexts to draw
        :return: An array of ciphertexts
        """
        bytes_per_word = self.bytes_per_word(self.word_size)
        ct = np.reshape(
            np.frombuffer(urandom(bytes_per_word * self.n_words * n_samples), dtype=self.word_dtype),
            (self.n_words, n_samples)
        )
        if self.word_size < 8:
            # Note: If the word size is greater than 8, it will always fit the dtype for the ciphers we use
            ct = np.right_shift(ct, 8 - self.word_size)
        return ct

    @abstractmethod
    def key_schedule(self, key):
        """
        Applies the key schedule
        :param key: The key
        :return: A list of round keys
        """
        pass

    def rol(self, x, k):
        """
        :param x: What to rotate
        :param k: How to rotate
        :return: x rotated by k bits to the left
        """
        return ((x << k) & self.mask_val) | (x >> (self.word_size - k))

    def ror(self, x, k):
        """
        :param x: What to rotate
        :param k: How to rotate
        :return: x rotated by k bits to the right
        """
        return (x >> k) | ((x << (self.word_size - k)) & self.mask_val)

    @staticmethod
    @abstractmethod
    def get_test_vectors():
        """
        :return: Returns the test vectors used for verifying the correct implementation of the cipher as a list of
            tuples of the form (cipher, plaintext, key, ciphertext), where cipher is an instance of self that will
            be used to verify the test vector
        """
        pass

    @classmethod
    def verify_test_vectors(cls):
        """
        Verifies the test vectors given by the designers
        :return: Result of the test
        """
        for cipher, pt, key, ct in cls.get_test_vectors():
            if not np.array_equal(ct, cipher.encrypt(pt, key)):
                print(f"ERROR: Test vector for {cls.__name__} not verified (encryption did not match)")
                return False
            try:  # This will use decrypt, which may not be implemented
                if not np.array_equal(pt, cipher.decrypt(ct, key)):
                    print(f"ERROR: Test vector for {cls.__name__} not verified (decryption did not match)")
                    return False
            except NotImplementedException as e:
                print(f"Info: Decryption not implemented for {cls.__name__}. (Original message: {e})")

        print(f"Info: All test vectors for {cls.__name__} verified")
        return True

