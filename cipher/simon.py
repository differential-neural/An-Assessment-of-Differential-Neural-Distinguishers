import numpy as np

from cipher.abstract_cipher import AbstractCipher


class Simon(AbstractCipher):

    def __init__(self, n_rounds=32, word_size=16, m=4, alpha=1, beta=8, gamma=2, const_seq=0, use_key_schedule=True):
        """
        Initializes a Simon cipher object
        :param n_rounds: The number of rounds used for de-/encryption
        :param word_size:
        :param m:
        :param alpha:
        :param beta:
        :param gamma:
        :param const_seq:
        :param use_key_schedule:
        """
        super(Simon, self).__init__(
            n_rounds, word_size, n_words=2, n_main_key_words=m, n_round_key_words=1, use_key_schedule=use_key_schedule
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.const_seq = const_seq

    def encrypt_one_round(self, p, k, rc=None):
        """
        Round function of the cipher
        :param p: The plaintext
        :param k: The round key
        :param rc: The round constant
        :return: The one round encryption of p using key k
        """
        c0 = (((self.rol(p[0], self.alpha) & self.rol(p[0], self.beta)) ^ self.rol(p[0], self.gamma)) ^ p[1]) ^ k[0]
        c1 = p[0]
        return c0, c1

    def decrypt_one_round(self, c, k, rc=None):
        """
        Inverse round function of the cipher
        :param c: The ciphertext
        :param k: The round key
        :param rc: The round constant
        :return: The one round decryption of c using key k
        """
        c0, c1 = c[0], c[1]
        c0 = ((c0 ^ k[0]) ^ self.rol(c1, self.gamma)) ^ (self.rol(c1, self.alpha) & self.rol(c1, self.beta))
        return c1, c0

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
            raise Exception("ERROR: Only one variant of calculating back is known")
        c0, c1 = c[0], c[1]
        c0 = (c0 ^ self.rol(c1, self.gamma)) ^ (self.rol(c1, self.alpha) & self.rol(c1, self.beta))
        return c1, c0

    def key_schedule(self, key):
        """
        Applies the key schedule
        :param key: The key
        :return: A list of round keys
        """
        z = [[1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0,1,1,1,1,1,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,1,1,0],
             [1,0,0,0,1,1,1,0,1,1,1,1,1,0,0,1,0,0,1,1,0,0,0,0,1,0,1,1,0,1,0,1,0,0,0,1,1,1,0,1,1,1,1,1,0,0,1,0,0,1,1,0,0,0,0,1,0,1,1,0,1,0],
             [1,0,1,0,1,1,1,1,0,1,1,1,0,0,0,0,0,0,1,1,0,1,0,0,1,0,0,1,1,0,0,0,1,0,1,0,0,0,0,1,0,0,0,1,1,1,1,1,1,0,0,1,0,1,1,0,1,1,0,0,1,1],
             [1,1,0,1,1,0,1,1,1,0,1,0,1,1,0,0,0,1,1,0,0,1,0,1,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,1,0,1,0,0,1,1,1,0,0,1,1,0,1,0,0,0,0,1,1,1,1],
             [1,1,0,1,0,0,0,1,1,1,1,0,0,1,1,0,1,0,1,1,0,1,1,0,0,0,1,0,0,0,0,0,0,1,0,1,1,1,0,0,0,0,1,1,0,0,1,0,1,0,0,1,0,0,1,1,1,0,1,1,1,1]]
        ks = [0 for i in range(self.n_rounds)]
        ks[:self.n_main_key_words] = list(reversed(key))
        for i in range(self.n_main_key_words, self.n_rounds):
            tmp = self.ror(ks[i-1], 3)
            if self.n_main_key_words == 4:
                tmp ^= ks[i-3]
            tmp ^= self.ror(tmp, 1)
            ks[i] = (ks[i-self.n_main_key_words] ^ self.mask_val) ^ tmp \
                    ^ z[self.const_seq][(i-self.n_main_key_words) % 62] ^ 3
        return np.array(ks, dtype=self.main_key_word_dtype)[:, np.newaxis]

    @staticmethod
    def get_test_vectors():
        """
        Test vectors from https://eprint.iacr.org/2013/404.pdf, page 41
        :return: Returns the test vectors used for verifying the correct implementation of the cipher as a list of
            tuples of the form (cipher, plaintext, key, ciphertext), where cipher is an instance of self that will
            be used to verify the test vector
        """
        # Initialize Simon32/64 according to specification
        simon32_64 = Simon()
        key = np.array([[0x1918], [0x1110], [0x0908], [0x0100]], dtype=np.uint16)
        ks = simon32_64.key_schedule(key)
        pt = np.array([[0x6565], [0x6877]], dtype=np.uint16)
        ct = np.array([[0xc69b], [0xe9bb]], dtype=np.uint16)

        # Initialize Simon64/128 according to specification
        simon64_128 = Simon(n_rounds=44, word_size=32, alpha=1, beta=8, gamma=2, const_seq=3)
        key = np.array([[0x1b1a1918], [0x13121110], [0x0b0a0908], [0x03020100]], dtype=np.uint32)
        ks2 = simon64_128.key_schedule(key)
        pt2 = np.array([[0x656b696c], [0x20646e75]], dtype=np.uint32)
        ct2 = np.array([[0x44c8fc20], [0xb9dfa07a]], dtype=np.uint32)

        return [(simon32_64, pt, ks, ct), (simon64_128, pt2, ks2, ct2)]

