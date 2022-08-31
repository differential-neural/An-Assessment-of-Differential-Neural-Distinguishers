import numpy as np

from cipher.abstract_cipher import AbstractCipher


class Present(AbstractCipher):

    def __init__(self, n_rounds=31, use_key_schedule=True):
        """
        Initializes a Present cipher object
        :param n_rounds: The number of round used for de-/encryption
        :param use_key_schedule: Whether to use the key schedule or independent round keys
        """
        super(Present, self).__init__(
            n_rounds + 1,  # Internally, we will use the number of rounds as the number of round key additions
            word_size=4, n_words=16, n_main_key_words=20, n_round_key_words=16, use_key_schedule=use_key_schedule
        )

    # Index:             0    1    2    3    4    5    6    7    8    9    A    B    C    D    E    F
    SBOX    = np.array([0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2])
    SBOXINV = np.array([0x5, 0xE, 0xF, 0x8, 0xC, 0x1, 0x2, 0xD, 0xB, 0x4, 0x6, 0x3, 0x0, 0x7, 0x9, 0xA])

    def get_n_rounds(self, key_additions=True):
        """
        :param key_additions: Whether to return the number of round key additions or the number of rounds
        :return: The number of rounds or the number of round key additions
        """
        if key_additions:
            return super(Present, self).get_n_rounds()
        return super(Present, self).get_n_rounds() - 1

    @staticmethod
    def bit_at_pos(x, i):
        """
        :param x: Array
        :param i: Bit position to return
        :return: The i-th bit of all elements in x
        """
        return np.right_shift(x, i) & 0b1

    @staticmethod
    def bit_to_pos(x, i):
        """
        :param x: Array
        :param i: Bit position
        :return: All elements of x shifted to the left by i bits
        """
        return np.left_shift(x, i)

    def encrypt_one_round(self, p, k, rc=None):
        """
        Round function of the cipher
        :param p: The plaintext
        :param k: The round key
        :param rc: The round constant
        :return: The one round encryption of p using key k
        """
        s = [np.zeros(p[i].shape, dtype=self.word_dtype) for i in range(16)]
        for i in range(16):
            # add round key
            s[i] = p[i] ^ k[i]
            # apply s-box layer
            s[i] = self.SBOX[s[i]]
        c = [np.zeros(s[i].shape, dtype=self.word_dtype) for i in range(16)]
        # permutation layer
        for i in range(4):
            q = 3-i
            c[4*i+0] = self.bit_to_pos(self.bit_at_pos(s[0], q), 3) + self.bit_to_pos(self.bit_at_pos(s[1], q), 2) + \
                       self.bit_to_pos(self.bit_at_pos(s[2], q), 1) + self.bit_to_pos(self.bit_at_pos(s[3], q), 0)

            c[4*i+1] = self.bit_to_pos(self.bit_at_pos(s[4], q), 3) + self.bit_to_pos(self.bit_at_pos(s[5], q), 2) + \
                       self.bit_to_pos(self.bit_at_pos(s[6], q), 1) + self.bit_to_pos(self.bit_at_pos(s[7], q), 0)

            c[4*i+2] = self.bit_to_pos(self.bit_at_pos(s[8], q), 3) + self.bit_to_pos(self.bit_at_pos(s[9], q), 2) + \
                       self.bit_to_pos(self.bit_at_pos(s[10], q), 1) + self.bit_to_pos(self.bit_at_pos(s[11], q), 0)

            c[4*i+3] = self.bit_to_pos(self.bit_at_pos(s[12], q), 3) + self.bit_to_pos(self.bit_at_pos(s[13], q), 2) + \
                       self.bit_to_pos(self.bit_at_pos(s[14], q), 1) + self.bit_to_pos(self.bit_at_pos(s[15], q), 0)
        return c

    def encrypt(self, p, keys):
        """
        Encrypt by applying the round function for each given round key
        :param p: The plaintext
        :param keys: A list of round keys
        :return: The encryption of p under the round keys in keys
        """
        # For Present, the number of round keys is the number of rounds + 1
        c = super(Present, self).encrypt(p, keys[:-1])
        return c ^ keys[-1]

    def decrypt_one_round(self, c, k, rc=None):
        """
        Inverse round function of the cipher
        :param c: The ciphertext
        :param k: The round key
        :param rc: The round constant
        :return: The one round decryption of c using key k
        """
        p = [np.zeros(c[i].shape, dtype=self.word_dtype) for i in range(16)]
        # inverse permutation layer
        for i in range(4):
            p[4*i] = self.bit_to_pos(self.bit_at_pos(c[i], 3), 3) + self.bit_to_pos(self.bit_at_pos(c[4+i], 3), 2) + \
                     self.bit_to_pos(self.bit_at_pos(c[8+i], 3), 1) + self.bit_to_pos(self.bit_at_pos(c[12+i], 3), 0)

            p[4*i+1] = self.bit_to_pos(self.bit_at_pos(c[i], 2), 3) + self.bit_to_pos(self.bit_at_pos(c[4+i], 2), 2) + \
                       self.bit_to_pos(self.bit_at_pos(c[8+i], 2), 1) + self.bit_to_pos(self.bit_at_pos(c[12+i], 2), 0)

            p[4*i+2] = self.bit_to_pos(self.bit_at_pos(c[i], 1), 3) + self.bit_to_pos(self.bit_at_pos(c[4+i], 1), 2) + \
                       self.bit_to_pos(self.bit_at_pos(c[8+i], 1), 1) + self.bit_to_pos(self.bit_at_pos(c[12+i], 1), 0)

            p[4*i+3] = self.bit_to_pos(self.bit_at_pos(c[i], 0), 3) + self.bit_to_pos(self.bit_at_pos(c[4+i], 0), 2) + \
                       self.bit_to_pos(self.bit_at_pos(c[8+i], 0), 1) + self.bit_to_pos(self.bit_at_pos(c[12+i], 0), 0)
        for i in range(16):
            # inverse s-box layer
            p[i] = self.SBOXINV[p[i]]
            # add round key
            p[i] = p[i] ^ k[i]
        return p

    def decrypt(self, c, keys):
        """
        Decrypt by applying the inverse round function for each given key
        :param c: The ciphertext
        :param keys: A list of round keys
        :return: The decryption of c under the round keys in keys
        """
        c = c ^ keys[-1]
        return super(Present, self).decrypt(c, keys[:-1])

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
        raise Exception("ERROR: No variant of calculating back is implemented")

    def key_schedule(self, key):
        """
        Applies the key schedule
        :param key: The key
        :return: A list of round keys
        """
        samples = len(key[0])
        ks = np.zeros((self.n_rounds, 20, samples), dtype=self.word_dtype)
        ks[0] = key.copy()
        for r in range(self.n_rounds - 1):
            lmk = ks[r].copy()
            for i in range(20):
                ks[r+1, i] = np.left_shift(lmk[(i+15) % 20] & 0b0111, 1) ^ np.right_shift(lmk[(i+16) % 20], 3)
            ks[r+1, 0] = self.SBOX[ks[r+1, 0]]
            ks[r+1, 15] = ks[r+1, 15] ^ ((r+1) >> 1)
            ks[r+1, 16] = ks[r+1, 16] ^ (((r+1) & 0b1) << 3)
        return ks[:, 0:16]

    @staticmethod
    def get_test_vectors():
        """
        Test vectors from https://link.springer.com/chapter/10.1007/978-3-540-74735-2_31, page 464 (15th page)
        :return: Returns the test vectors used for verifying the correct implementation of the cipher as a list of
            tuples of the form (cipher, plaintext, key, ciphertext), where cipher is an instance of self that will
            be used to verify the test vector
        """
        present_80 = Present()
        key = np.array([[0, 0xf, 0, 0xf], [0, 0xf, 0, 0xf], [0, 0xf, 0, 0xf], [0, 0xf, 0, 0xf],
                        [0, 0xf, 0, 0xf], [0, 0xf, 0, 0xf], [0, 0xf, 0, 0xf], [0, 0xf, 0, 0xf],
                        [0, 0xf, 0, 0xf], [0, 0xf, 0, 0xf], [0, 0xf, 0, 0xf], [0, 0xf, 0, 0xf],
                        [0, 0xf, 0, 0xf], [0, 0xf, 0, 0xf], [0, 0xf, 0, 0xf], [0, 0xf, 0, 0xf],
                        [0, 0xf, 0, 0xf], [0, 0xf, 0, 0xf], [0, 0xf, 0, 0xf], [0, 0xf, 0, 0xf]], dtype=np.uint8)
        ks = present_80.key_schedule(key)
        pt = np.array([[0, 0, 0xf, 0xf], [0, 0, 0xf, 0xf], [0, 0, 0xf, 0xf], [0, 0, 0xf, 0xf],
                       [0, 0, 0xf, 0xf], [0, 0, 0xf, 0xf], [0, 0, 0xf, 0xf], [0, 0, 0xf, 0xf],
                       [0, 0, 0xf, 0xf], [0, 0, 0xf, 0xf], [0, 0, 0xf, 0xf], [0, 0, 0xf, 0xf],
                       [0, 0, 0xf, 0xf], [0, 0, 0xf, 0xf], [0, 0, 0xf, 0xf], [0, 0, 0xf, 0xf]], dtype=np.uint8)
        ct = np.array([
            [0x5, 0xE, 0xA, 0x3], [0x5, 0x7, 0x1, 0x3], [0x7, 0x2, 0x1, 0x3], [0x9, 0xC, 0x2, 0x3],
            [0xC, 0x4, 0xF, 0xD], [0x1, 0x6, 0xF, 0xC], [0x3, 0xC, 0xC, 0xD], [0x8, 0x0, 0x7, 0x3],
            [0x7, 0xF, 0x2, 0x2], [0xB, 0x5, 0xF, 0x1], [0x2, 0x9, 0x6, 0x3], [0x2, 0x4, 0x8, 0x2],
            [0x8, 0x5, 0x4, 0x1], [0x4, 0x0, 0x1, 0x0], [0x4, 0x4, 0x7, 0xD], [0x5, 0x9, 0xB, 0x2]
        ], dtype=np.uint8)
        return [(present_80, pt, ks, ct)]
        


