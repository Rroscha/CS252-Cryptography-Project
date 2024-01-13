import numpy as np


class DES:
    IP = [58, 50, 42, 34, 26, 18, 10, 2,
          60, 52, 44, 36, 28, 20, 12, 4,
          62, 54, 46, 38, 30, 22, 14, 6,
          64, 56, 48, 40, 32, 24, 16, 8,
          57, 49, 41, 33, 25, 17, 9, 1,
          59, 51, 43, 35, 27, 19, 11, 3,
          61, 53, 45, 37, 29, 21, 13, 5,
          63, 55, 47, 39, 31, 23, 15, 7]
    
    IP_INVERSE = [40, 8, 48, 16, 56, 24, 64, 32,
                  39, 7, 47, 15, 55, 23, 63, 31,
                  38, 6, 46, 14, 54, 22, 62, 30,
                  37, 5, 45, 13, 53, 21, 61, 29,
                  36, 4, 44, 12, 52, 20, 60, 28,
                  35, 3, 43, 11, 51, 19, 59, 27,
                  34, 2, 42, 10, 50, 18, 58, 26,
                  33, 1, 41, 9, 49, 17, 57, 25]
    
    PC1 = [57, 49, 41, 33, 25, 17, 9,
           1, 58, 50, 42, 34, 26, 18,
           10, 2, 59, 51, 43, 35, 27,
           19, 11, 3, 60, 52, 44, 36,
           63, 55, 47, 39, 31, 23, 15,
           7, 62, 54, 46, 38, 30, 22,
           14, 6, 61, 53, 45, 37, 29,
           21, 13, 5, 28, 20, 12, 4]
    
    PC2 = [14, 17, 11, 24, 1, 5, 3, 28,
           15, 6, 21, 10, 23, 19, 12, 4,
           26, 8, 16, 7, 27, 20, 13, 2,
           41, 52, 31, 37, 47, 55, 30, 40,
           51, 45, 33, 48, 44, 49, 39, 56,
           34, 53, 46, 42, 50, 36, 29, 32]
    
    LEFT_CIRCULAR_SHIFT = [1, 1, 2, 2,
                           2, 2, 2, 2,
                           1, 2, 2, 2,
                           2, 2, 2, 1]
    
    E = [32, 1, 2, 3, 4, 5,
         4, 5, 6, 7, 8, 9,
         8, 9, 10, 11, 12, 13,
         12, 13, 14, 15, 16, 17,
         16, 17, 18, 19, 20, 21,
         20, 21, 22, 23, 24, 25,
         24, 25, 26, 27, 28, 29,
         28, 29, 30, 31, 32, 1]
    
    P = [16, 7, 20, 21, 29, 12, 28, 17,
         1, 15, 23, 26, 5, 18, 31, 10,
         2, 8, 24, 14, 32, 27, 3, 9,
         19, 13, 30, 6, 22, 11, 4, 25]
    
    S1 = [[14, 4, 13, 1, 2, 15, 11, 8,
           3, 10, 6, 12, 5, 9, 0, 7],
          [0, 15, 7, 4, 14, 2, 13, 1,
           10, 6, 12, 11, 9, 5, 3, 8],
          [4, 1, 14, 8, 13, 6, 2, 11,
           15, 12, 9, 7, 3, 10, 5, 0],
          [15, 12, 8, 2, 4, 9, 1, 7,
           5, 11, 3, 14, 10, 0, 6, 13]]
    
    S2 = [[15, 1, 8, 14, 6, 11, 3, 4,
           9, 7, 2, 13, 12, 0, 5, 10],
          [3, 13, 4, 7, 15, 2, 8, 14,
           12, 0, 1, 10, 6, 9, 11, 5],
          [0, 14, 7, 11, 10, 4, 13, 1,
           5, 8, 12, 6, 9, 3, 2, 15],
          [13, 8, 10, 1, 3, 15, 4, 2,
           11, 6, 7, 12, 0, 5, 14, 9]]
    
    S3 = [[10, 0, 9, 14, 6, 3, 15, 5,
           1, 13, 12, 7, 11, 4, 2, 8],
          [13, 7, 0, 9, 3, 4, 6, 10,
           2, 8, 5, 14, 12, 11, 15, 1],
          [13, 6, 4, 9, 8, 15, 3, 0,
           11, 1, 2, 12, 5, 10, 14, 7],
          [1, 10, 13, 0, 6, 9, 8, 7,
           4, 15, 14, 3, 11, 5, 2, 12]]
    
    S4 = [[7, 13, 14, 3, 0, 6, 9, 10,
           1, 2, 8, 5, 11, 12, 4, 15],
          [13, 8, 11, 5, 6, 15, 0, 3,
           4, 7, 2, 12, 1, 10, 14, 9],
          [10, 6, 9, 0, 12, 11, 7, 13,
           15, 1, 3, 14, 5, 2, 8, 4],
          [3, 15, 0, 6, 10, 1, 13, 8,
           9, 4, 5, 11, 12, 7, 2, 14]]
    
    S5 = [[2, 12, 4, 1, 7, 10, 11, 6,
           8, 5, 3, 15, 13, 0, 14, 9],
          [14, 11, 2, 12, 4, 7, 13, 1,
           5, 0, 15, 10, 3, 9, 8, 6],
          [4, 2, 1, 11, 10, 13, 7, 8,
           15, 9, 12, 5, 6, 3, 0, 14],
          [11, 8, 12, 7, 1, 14, 2, 13,
           6, 15, 0, 9, 10, 4, 5, 3]]
    
    S6 = [[12, 1, 10, 15, 9, 2, 6, 8,
           0, 13, 3, 4, 14, 7, 5, 11],
          [10, 15, 4, 2, 7, 12, 9, 5,
           6, 1, 13, 14, 0, 11, 3, 8],
          [9, 14, 15, 5, 2, 8, 12, 3,
           7, 0, 4, 10, 1, 13, 11, 6],
          [4, 3, 2, 12, 9, 5, 15, 10,
           11, 14, 1, 7, 6, 0, 8, 13]]
    
    S7 = [[4, 11, 2, 14, 15, 0, 8, 13,
           3, 12, 9, 7, 5, 10, 6, 1],
          [13, 0, 11, 7, 4, 9, 1, 10,
           14, 3, 5, 12, 2, 15, 8, 6],
          [1, 4, 11, 13, 12, 3, 7, 14,
           10, 15, 6, 8, 0, 5, 9, 2],
          [6, 11, 13, 8, 1, 4, 10, 7,
           9, 5, 0, 15, 14, 2, 3, 12]]
    
    S8 = [[13, 2, 8, 4, 6, 15, 11, 1,
           10, 9, 3, 14, 5, 0, 12, 7],
          [1, 15, 13, 8, 10, 3, 7, 4,
           12, 5, 6, 11, 0, 14, 9, 2],
          [7, 11, 4, 1, 9, 12, 14, 2,
           0, 6, 10, 13, 15, 3, 5, 8],
          [2, 1, 14, 7, 4, 10, 8, 13,
           15, 12, 9, 0, 3, 5, 6, 11]]

    S = [S1, S2, S3, S4, S5, S6, S7, S8]
    
    @staticmethod
    def permutation(block : np.uint64, permutation, offset=0, bits=64):
        block = np.array([int(i) for i in bin(block)[2:].zfill(bits)])
        block = block[np.array(permutation) - offset]
        return int(''.join([str(i) for i in block]), 2)
    
    @staticmethod
    def left_circular_shift(block : np.uint64, shift, bits=56):
        middle = bits // 2
        block       = np.array([int(i) for i in bin(block)[2:].zfill(bits)])
        block_left  = block[:middle]
        block_right = block[middle:]
        block_left  = np.roll(block_left, -shift)
        block_right = np.roll(block_right, -shift)
        block = np.concatenate((block_left, block_right))
        return int(''.join([str(i) for i in block]), 2)
    
    @staticmethod
    def s_box(bits_48, bits=48):
        bits_48 = np.array([int(i) for i in bin(bits_48)[2:].zfill(bits)])
        bits_32 = []
        for k in range(8):
            bits_6 = bits_48[k * 6: (k + 1) * 6]
            row = int(''.join([str(i) for i in bits_6[[0, 5]]]), 2)
            col = int(''.join([str(i) for i in bits_6[1:5]]), 2)
            bits_32.extend([int(i) for i in bin(DES.S[k][row][col])[2:].zfill(4)])
        bits_32 = int(''.join([str(i) for i in np.array(bits_32)]), 2)
        return bits_32

    @staticmethod
    def mangler_function(R : np.uint32, round_key):
        expanded_R = np.uint64(DES.permutation(R, DES.E, 1, 32))  # 32 -> 48
        bits_48 = expanded_R ^ np.uint64(round_key)
        bits_32 = DES.s_box(bits_48)
        bits_32 = DES.permutation(bits_32, DES.P, 1, 32)
        return bits_32
    
    @staticmethod
    def feistel_network(block : np.uint64, round_keys):
        middle = 64 // 2
        block  = np.array([int(i) for i in bin(block)[2:].zfill(64)])
        L = int(''.join([str(i) for i in np.array(block[:middle])]), 2)
        R = int(''.join([str(i) for i in np.array(block[middle:])]), 2)
        for i in range(len(round_keys)):
            L, R = R, L ^ DES.mangler_function(R, round_keys[i])
        block = np.concatenate((np.array([int(i) for i in bin(L)[2:].zfill(32)]),
                                np.array([int(i) for i in bin(R)[2:].zfill(32)])))
        return int(''.join([str(i) for i in block]), 2)
    
    @staticmethod
    def swap(block : np.uint64):
        middle = 64 // 2
        block       = np.array([int(i) for i in bin(block)[2:].zfill(64)])
        block_left  = block[:middle]
        block_right = block[middle:]
        block = np.concatenate((block_right, block_left))
        return int(''.join([str(i) for i in block]), 2)
    
    @staticmethod
    def internal_encrypt(block : np.uint64, round_keys):
        block = DES.permutation(block, DES.IP, 1, 64)
        block = DES.feistel_network(block, round_keys)
        block = DES.swap(block)
        block = DES.permutation(block, DES.IP_INVERSE, 1, 64)
        return block
    
    @staticmethod
    def internal_decrypt(block : np.uint64, round_keys):
        block = DES.permutation(block, DES.IP, 1, 64)
        block = DES.swap(block)
        block = DES.feistel_network(DES.swap(block), round_keys[::-1])
        block = DES.swap(block)
        block = DES.permutation(block, DES.IP_INVERSE, 1, 64)
        return block

    def __init__(self, master_key):
        self.master_key = master_key
        self.round_keys = self.key_schedule()
    
    def key_schedule(self):
        round_keys = []
        key = DES.permutation(self.master_key, DES.PC1, 1, 64)  # 64->56
        for i in range(16):
            key = DES.left_circular_shift(key, DES.LEFT_CIRCULAR_SHIFT[i], 56)
            round_keys.append(DES.permutation(key, DES.PC2, 1, 56))
        return round_keys
    
    def encrypt(self, block : np.uint64):
        return DES.internal_encrypt(block, self.round_keys)
    
    def decrypt(self, block : np.uint64):
        return DES.internal_decrypt(block, self.round_keys)


def generate_key():
    return np.random.randint(2 ** 64, dtype=np.uint64)


if __name__ == '__main__':
    key = generate_key()
    des = DES(key)
    print('Key: 0x{:016x}'.format(key))
    block = np.random.randint(2 ** 64, dtype=np.uint64)
    print('Block: 0x{:016x}'.format(block))
    encrypted_block = des.encrypt(block)
    print('Encrypted block: 0x{:016x}'.format(encrypted_block))
    decrypted_block = des.decrypt(encrypted_block)
    print('Decrypted block: 0x{:016x}'.format(decrypted_block))
    print('Block == Decrypted block:', block == decrypted_block)
