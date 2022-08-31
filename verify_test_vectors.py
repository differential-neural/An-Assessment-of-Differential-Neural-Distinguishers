from cipher.chacha import ChaCha
from cipher.katan import Katan
from cipher.present import Present
from cipher.simon import Simon
from cipher.skinny import Skinny
from cipher.speck import Speck


if __name__ == '__main__':
    res = True
    res &= ChaCha.verify_test_vectors()
    res &= Katan.verify_test_vectors()
    res &= Present.verify_test_vectors()
    res &= Simon.verify_test_vectors()
    res &= Skinny.verify_test_vectors()
    res &= Speck.verify_test_vectors()
    if not res:
        print('Error during test vector verification!')
        exit(1)
    print('Test vectors for all ciphers verified!')
