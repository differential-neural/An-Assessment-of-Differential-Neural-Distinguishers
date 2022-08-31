import numpy as np

from cipher.abstract_cipher import AbstractCipher


class Skinny(AbstractCipher):

    def __init__(self, n_rounds=32, use_key_schedule=True):
        """
        Initializes a Skinny cipher object
        :param n_rounds: The number of round used for de-/encryption
        :param use_key_schedule: Whether to use the key schedule or independent round keys
        """
        super(Skinny, self).__init__(
            n_rounds, word_size=4, n_words=16, n_main_key_words=16, n_round_key_words=16,
            use_key_schedule=use_key_schedule
        )

    # Index:             0    1    2    3    4    5    6    7    8    9    a    b    c    d    e    f
    SBOX    = np.array([0xC, 0x6, 0x9, 0x0, 0x1, 0xa, 0x2, 0xb, 0x3, 0x8, 0x5, 0xd, 0x4, 0xe, 0x7, 0xf])
    SBOXINV = np.array([0x3, 0x4, 0x6, 0x8, 0xC, 0xa, 0x1, 0xe, 0x9, 0x2, 0x5, 0x7, 0x0, 0xb, 0xd, 0xf])

    @staticmethod
    def substitute(state, sb):
        """
        Applies the s-box sb on each word/nibble of state
        :param state: The state to act on
        :param sb: The s-box
        :return: The resulting state
        """
        result = []
        for s in state:
            result.append(sb[s])
        return result

    def substitution_layer(self, state):
        """
        Applies the substitution layer to state
        :param state: The current state
        :return: The resulting state
        """
        return self.substitute(state, self.SBOX)

    def inv_substitution_layer(self, state):
        """
        Applies the inverse substitution layer to state
        :param state: The current state
        :return: The resulting state
        """
        return self.substitute(state, self.SBOXINV)

    @staticmethod
    def add_constants(state, constants):
        """
        Adds the round constant to state. Constants should contain the values of rc0 to rc5, where rc0 is the lsb
        :param state: The current state
        :param constants: The round constant
        :return: The resulting state
        """
        # Add rc0 to rc3 to the first word/nibble of the state
        state[0] = state[0] ^ (constants & 0xf)
        # Add rc4 and rc5 to the first word/nibble of the second row of the state
        state[4] = state[4] ^ (constants >> 4)
        # Add c2=0x2 to the first word/nibble of the third row of the state
        state[8] = state[8] ^ 0x2
        return state

    @staticmethod
    def add_tweak_key(state, tweakKey):
        """
        Adds the (tweak-)key to the first two rows of the state
        :param state: The current state
        :param tweakKey: The tweak-key to add
        :return: the resulting state
        """
        for i in range(8):
            state[i] = state[i] ^ tweakKey[i]
        return state

    @staticmethod
    def shift_rows(state):
        """
        :param state: The current state
        :return: The state with the rows shifted
        """
        return [state[0], state[1], state[2], state[3],
                state[7], state[4], state[5], state[6],
                state[10], state[11], state[8], state[9],
                state[13], state[14], state[15], state[12]]

    @staticmethod
    def inv_shift_rows(state):
        """
        :param state: The current state
        :return: The state with the rows shifted
        """
        return [state[0], state[1], state[2], state[3],
                state[5], state[6], state[7], state[4],
                state[10], state[11], state[8], state[9],
                state[15], state[12], state[13], state[14]]

    @staticmethod
    def mix_columns(state):
        """
        Applies the mix columns operation
        :param state: The current state
        :return: The resulting state
        """
        # Add 3rd row to 2nd row
        for i in range(4):
            state[4 + i] = state[4 + i] ^ state[8 + i]
        # Add 1st row to 3rd row
        for i in range(4):
            state[8 + i] = state[8 + i] ^ state[i]
        # Add 3rd row to 4th row
        for i in range(4):
            state[12 + i] = state[12 + i] ^ state[8 + i]
        # Permute columns and return
        return [state[12], state[13], state[14], state[15],
                state[0], state[1], state[2], state[3],
                state[4], state[5], state[6], state[7],
                state[8], state[9], state[10], state[11]]

    @staticmethod
    def inv_mix_columns(state):
        """
        Applies the inverse mix columns operation
        :param state: The current state
        :return: The resulting state
        """
        # Revert permutation of columns
        state = [state[4], state[5], state[6], state[7],
                 state[8], state[9], state[10], state[11],
                 state[12], state[13], state[14], state[15],
                 state[0], state[1], state[2], state[3]]
        # Add 3rd row to 4th row
        for i in range(4):
            state[12 + i] = state[12 + i] ^ state[8 + i]
        # Add 1st row to 3rd row
        for i in range(4):
            state[8 + i] = state[8 + i] ^ state[i]
        # Add 3rd row to 2nd row
        for i in range(4):
            state[4 + i] = state[4 + i] ^ state[8 + i]
        return state

    def encrypt_one_round(self, p, k, rc=None):
        """
        Round function of the cipher
        :param p: The plaintext
        :param k: The round key
        :param rc: The round constant
        :return: The one round encryption of p using key k
        """
        if rc is None:
            raise Exception("ERROR: Round constant has to be set for Skinny encryption")
        s = self.substitution_layer(p)
        s = self.add_constants(s, rc)
        s = self.add_tweak_key(s, k)
        s = self.shift_rows(s)
        s = self.mix_columns(s)
        return s

    def decrypt_one_round(self, c, k, rc=None):
        """
        Inverse round function of the cipher
        :param c: The ciphertext
        :param k: The round key
        :param rc: The round constant
        :return: The one round decryption of c using key k
        """
        if rc is None:
            raise Exception("ERROR: Round constant has to be set for Skinny decryption")
        s = self.inv_mix_columns(c)
        s = self.inv_shift_rows(s)
        s = self.add_tweak_key(s, k)
        s = self.add_constants(s, rc)
        s = self.inv_substitution_layer(s)
        return s

    def calc_back(self, c, p=None, variant=1):
        """
        Revert deterministic parts of the round function
        :param c: The ciphertext
        :param p: The initial plaintext
        :param variant: Select the variant of how to calculate back (default is 1; 0 means not calculating back)
            Options are
                - 1: Only revert Mix Columns and Shift Rows
                - 2: Revert Mix Columns, Shift Rows, Add Constants and the SBoxes for last two rows
        :return: The inner state after reverting the deterministic transformation at the end of the encryption process
        """
        if variant == 0:
            return c
        # Always revert Mix Columns and Shift Rows
        s = self.inv_mix_columns(c)
        s = self.inv_shift_rows(s)
        if variant == 1:  # Only revert Mix Columns and Shift Rows
            return s
        if variant == 2:  # Revert Mix Columns, Shift Rows, Add Constants and the SBoxes for last two rows
            # Remove round constants from last round
            s = self.add_constants(s, self.get_rc(self.n_rounds - 1))
            # Revert SBoxes not influenced by the round (tweak-)key
            for i in range(8, 16):
                s[i] = self.SBOXINV[s[i]]
            return s
        raise Exception(f'ERROR: Variant {variant} of calculating back is not implemented')

    def get_rc(self, r):
        """
        :param r: The round
        :return: The round constant for round r
        """
        constant = 0x1
        for key in range(r):
            # Update constant
            constant = ((constant << 1) & 0x3f) ^ ((constant >> 5) & 1) ^ ((constant >> 4) & 1) ^ 1
        return constant

    def key_schedule(self, key):
        """
        Applies the key schedule
        :param key: The key
        :return: A list of round keys
        """
        ks = [key[:8]]
        for i in range(self.n_rounds - 1):
            # Permute tweak-key state by P_T
            key = [key[i] for i in [9,15,8,13,10,14,12,11,0,1,2,3,4,5,6,7]]
            # Add first two rows as round (tweak-)key
            ks.append(key[:8])
        # Bring round keys in the form (roundConstant, roundKey) and return
        return ks

    @staticmethod
    def get_test_vectors():
        """
        Test vectors from https://eprint.iacr.org/2016/660.pdf, page 46
        :return: Returns the test vectors used for verifying the correct implementation of the cipher as a list of
            tuples of the form (cipher, plaintext, key, ciphertext), where cipher is an instance of self that will
            be used to verify the test vector
        """
        # Initialize Skinny64/64 according to specification
        skinny_64_64 = Skinny()
        key = np.array([[0xf], [0x5], [0x2], [0x6], [0x9], [0x8], [0x2], [0x6],
                        [0xf], [0xc], [0x6], [0x8], [0x1], [0x2], [0x3], [0x8]], dtype=np.uint8)
        ks = skinny_64_64.key_schedule(key)
        pt = np.array([[0x0], [0x6], [0x0], [0x3], [0x4], [0xf], [0x9], [0x5],
                       [0x7], [0x7], [0x2], [0x4], [0xd], [0x1], [0x9], [0xd]], dtype=np.uint8)
        ct = np.array([
            [0xb], [0xb], [0x3], [0x9], [0xd], [0xf], [0xb], [0x2],
            [0x4], [0x2], [0x9], [0xb], [0x8], [0xa], [0xc], [0x7]
        ], dtype=np.uint8)
        return [(skinny_64_64, pt, ks, ct)]



