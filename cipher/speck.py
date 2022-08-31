import numpy as np

from cipher.abstract_cipher import AbstractCipher


class Speck(AbstractCipher):
    
    def __init__(self, n_rounds=22, word_size=16, use_key_schedule=True, alpha=7, beta=2, m=4):
        """
        Initializes a Speck cipher object
        :param n_rounds: The number of rounds
        :param word_size: The size (in bits) of the right/left side
        :param use_key_schedule: Whether to use the key schedule or independent round keys
        :param alpha: The rotational constant alpha
        :param beta: The rotational constant beta
        :param m: Number of words of the key
        """
        super(Speck, self).__init__(
            n_rounds, word_size, n_words=2, n_main_key_words=m, n_round_key_words=1, use_key_schedule=use_key_schedule
        )
        self.alpha = alpha
        self.beta = beta
    
    def encrypt_one_round(self, p, k, rc=None):
        """
        Round function of the cipher
        :param p: The plaintext
        :param k: The round key
        :param rc: The round constant
        :return: The one round encryption of p using key k
        """
        c0, c1 = p[0], p[1]
        c0 = self.ror(c0, self.alpha)
        c0 = (c0 + c1) & self.mask_val
        c0 = c0 ^ k[0]  # round key consists of one word only
        c1 = self.rol(c1, self.beta)
        c1 = c1 ^ c0
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
        c1 = c1 ^ c0
        c1 = self.ror(c1, self.beta)
        c0 = c0 ^ k[0]  # round key consists of one word only
        c0 = (c0 - c1) & self.mask_val
        c0 = self.rol(c0, self.alpha)
        return c0, c1

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
        c0, c1 = c[0], c[1]
        c1 = c1 ^ c0
        c1 = self.ror(c1, self.beta)
        return c0, c1

    def key_schedule(self, key):
        """
        Applies the key schedule
        :param key: The key
        :return: A list of round keys
        """
        ks = [0 for i in range(self.n_rounds)]
        ks[0] = key[len(key)-1]
        l = list(reversed(key[:len(key)-1]))
        
        for i in range(self.n_rounds-1):
            l[i % 3], ks[i+1] = self.encrypt_one_round((l[i % 3], ks[i]), [i])
    
        return np.array(ks, dtype=self.main_key_word_dtype)[:, np.newaxis]

    @staticmethod
    def get_test_vectors():
        """
        Test vectors from https://eprint.iacr.org/2013/404.pdf, page 42
        :return: Returns the test vectors used for verifying the correct implementation of the cipher as a list of
            tuples of the form (cipher, plaintext, key, ciphertext), where cipher is an instance of self that will
            be used to verify the test vector
        """
        # Initialize Speck32/64 according to specification
        speck32_64 = Speck()
        key = np.array([[0x1918], [0x1110], [0x0908], [0x0100]], dtype=np.uint16)
        ks = speck32_64.key_schedule(key)
        pt = np.array([[0x6574], [0x694c]], dtype=np.uint16)
        ct = np.array([[0xa868], [0x42f2]], dtype=np.uint16)
        return [(speck32_64, pt, ks, ct)]

