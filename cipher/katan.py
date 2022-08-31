import numpy as np

from cipher.abstract_cipher import AbstractCipher


class Katan(AbstractCipher):

    def __init__(self, n_rounds=254, use_key_schedule=True):
        """
        Initializes a Katan cipher object
        :param n_rounds: The number of rounds used for de-/encryption
        :param use_key_schedule: Whether to use the key schedule or independent round keys
        """
        super(Katan, self).__init__(
            n_rounds, word_size=32, n_words=1,  # We see the state of katan32 as one word
            n_main_key_words=5, n_round_key_words=1, use_key_schedule=use_key_schedule,
            main_key_word_size=16, round_key_word_size=2
        )
        self.len_l1 = 13
        self.len_l2 = 19

    IR = (1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1,
          1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 
          0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 
          0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1,
          1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 
          0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 
          1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 
          1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1,
          1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 
          0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 
          1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 
          0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,
          1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0)

    @staticmethod
    def parse_as_bits(x, word_size=32):
        """
        Parses each element of p as a list of bits with the lsb of each element at position i = 0
        :param x: The current state
        :param word_size: The word size of x
        :return: List of bits representing p
        """
        bits = np.zeros(np.insert(x.shape, 0, word_size), dtype=np.uint8)
        for i in range(word_size):
            bits[i] = np.right_shift(x, i) & 0b1
        return bits

    def parse_as_int(self, bits):
        """
        Parses a list of bits to the corresponding integers
        :param bits: The bits
        :return: Integers representing bits
        """
        num = np.zeros(bits.shape[1:], dtype=self.word_dtype)
        for i in range(bits.shape[0]-1, -1, -1):
            num = np.left_shift(num, 1) ^ bits[i]
        return num

    def encrypt_one_round(self, p, k, rc=None):
        """
        Round function of the cipher
        :param p: The plaintext
        :param k: The round key
        :param rc: The round constant
        :return: The one round encryption of p using key k
        """
        k_a = k & 0x1
        k_b = (k >> 1) & 0x1

        f_a = p[self.len_l2 + 12] ^ p[self.len_l2 + 7] ^ (p[self.len_l2 + 8] & p[self.len_l2 + 5]) ^ k_a
        if rc:
            f_a = f_a ^ p[self.len_l2 + 3]

        f_b = p[18] ^ p[7] ^ (p[12] & p[10]) ^ (p[8] & p[3]) ^ k_b

        # Shift L1 and L2 by 1
        p = np.roll(p, 1, axis=0)
        # Set value of feedback function
        p[0] = f_a
        p[self.len_l2] = f_b
        return p

    def encrypt(self, p, keys):
        """
        Encrypt by applying the round function for each given round key
        :param p: The plaintext
        :param keys: A list of round keys
        :return: The encryption of p under the round keys in keys
        """
        state = self.parse_as_bits(p)
        c = super(Katan, self).encrypt(state, keys)
        return self.parse_as_int(c)

    def decrypt_one_round(self, c, k, rc=None):
        """
        Inverse round function of the cipher
        :param c: The ciphertext
        :param k: The round key
        :param rc: The round constant
        :return: The one round decryption of c using key k
        """
        k_a = k & 0x1
        k_b = (k >> 1) & 0x1
        
        f_a = c[0] ^ c[self.len_l2 + 8] ^ (c[self.len_l2 + 9] & c[self.len_l2 + 6]) ^ k_a
        if rc:
            f_a = f_a ^ c[self.len_l2 + 4]

        f_b = c[self.len_l2 + 0] ^ c[8] ^ (c[13] & c[11]) ^ (c[9] & c[4]) ^ k_b

        # Shift L1 and L2 by 1
        c = np.roll(c, -1, axis=0)
        # Set value of feedback function
        c[self.len_l2 - 1] = f_b  # L2[-1] = f_b
        c[-1] = f_a  # L1[-1] = f_a
        return c

    def decrypt(self, c, keys):
        """
        Decrypt by applying the inverse round function for each given key
        :param c: The ciphertext
        :param keys: A list of round keys
        :return: The decryption of c under the round keys in keys
        """
        state = self.parse_as_bits(c)
        p = super(Katan, self).decrypt(state, keys)
        return self.parse_as_int(p)

    def calc_back(self, c, p=None, variant=1):
        """
        Revert deterministic parts of the round function
        :param c: The ciphertext
        :param p: The initial plaintext
        :param variant: Select the variant of how to calculate back (default is 1; 0 means not calculating back)
            Options are
                - 1: Only revert 4 full rounds (except for key addition)
                - 2: Revert 17 rounds as far as possible
        :return: The inner state after reverting the deterministic transformation at the end of the encryption process
        """
        if variant == 0:
            return c
        c = self.parse_as_bits(c)
        # Calculate 4 rounds back with keys set to zero
        for i in range(4):
            c = self.decrypt_one_round(c, 0, self.get_rc(self.n_rounds - 1 - i))
        if variant == 1:  # Revert 4 rounds except of key addition
            return self.parse_as_int(c)
        if variant == 2:  # Revert 17 rounds as far as possible
            for i in range(13):  # Note: We already reverted 4 rounds
                f_a = c[0]
                f_b = c[self.len_l2 + 0]
                # For all rounds, L1[x3]&L1[x4] is not known
                # For i >= 2, L2[y3]&L2[y4] is not known
                # For i >= 5,L1[8] is also masked by L1[x3]&L1[x4], with L1[x3] masked by a key bit
                # Also, L1[x5] is now masked by a key bit
                # For i >= 6, L2[y5]&L2[y6] is not known
                if i < 2:
                    f_b = f_b ^ (c[13] & c[11])
                if i < 5:
                    f_a = f_a ^ c[self.len_l2 + 8]  # Note that L1[8] may be masked with a key bit
                    if self.get_rc(self.n_rounds - 5 - i):
                        f_a = f_a ^ c[self.len_l2 + 4]
                if i < 6:
                    f_b = f_b ^ (c[9] & c[4])
                f_b = f_b ^ c[8]
                # Shift L1 and L2 by 1
                c = np.roll(c, -1, axis=0)
                # Set value of feedback function
                c[self.len_l2 - 1] = f_b  # L2[-1] = f_b
                c[-1] = f_a  # L1[-1] = f_a
            return self.parse_as_int(c)
        raise Exception(f'ERROR: Variant {variant} of calculating back is not implemented')

    def get_rc(self, r):
        """
        :param r: The round
        :return: The round constant for round r
        """
        return self.IR[r]

    def key_schedule(self, key):
        """
        Applies the key schedule
        :param key: The key
        :return: A list of round keys
        """
        # Parse masterKey as 80 bit vector
        samples = key.shape[1]
        state = np.concatenate([self.parse_as_bits(key[i], 16) for i in range(key.shape[0])], axis=0)
        key_bits = state.shape[0]

        ks = np.zeros((self.n_rounds, samples), dtype=np.uint8)
        for i in range(self.n_rounds):
            k = 0  # First bit (lsb) will be k_a, second k_b
            for j in range(2):
                zero_pos = (2 * i + j) % key_bits
                k ^= state[zero_pos] << j
                h = state[zero_pos + 0] ^ state[(zero_pos + 19) % key_bits] ^ \
                    state[(zero_pos + 30) % key_bits] ^ state[(zero_pos + 67) % key_bits]
                state[zero_pos] = h
            ks[i] = k
        return ks

    @staticmethod
    def get_test_vectors():
        """
        Test vectors from http://www.cs.technion.ac.il/~orrd/KATAN/
        :return: Returns the test vectors used for verifying the correct implementation of the cipher as a list of
            tuples of the form (cipher, plaintext, key, ciphertext), where cipher is an instance of self that will
            be used to verify the test vector
        """
        katan32 = Katan()
        key = np.array([[0xffff,     0],
                        [0xffff,     0],
                        [0xffff,     0],
                        [0xffff,     0],
                        [0xffff,     0]], dtype=np.uint16)
        ks = katan32.key_schedule(key)
        pt = np.array([[0, 0xffffffff]], dtype=np.uint32)
        ct = np.array([[0x7E1FF945, 0x432E61DA]], dtype=np.uint32)
        return [(katan32, pt, ks, ct)]

