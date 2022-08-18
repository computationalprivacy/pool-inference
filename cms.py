import numpy as np
import math
from tqdm.auto import tqdm
from scipy.linalg import hadamard


class CMS():
    """
    CMS client algorithm that obfuscates records from a user
    """

    def __init__(self, U=2600, m=1024, eps=4, k=65536,
                 hash_table=None, hadamard_matrix=None):
        """
        Define CMS

        Parameters
        ----------
        U : int
            size of universe
        m : int
            hash space
        eps : float
            privacy parameter
        k : int
            number of hash functions
        hash_table : np.ndarray (optional)
            U x k hash table (hash_table[e, h] is the hash of e computed on h)
        hadamard_matrix : np.ndarray (optional)
            m x m hadamard matrix with 0/1 elements instead of -1/+1
        """
        self.need_hash_table = U != m
        self.eps = eps
        self.k = k
        self.m = m
        self.U = U
        self.hash_table = hash_table
        self.hadamard_matrix = hadamard_matrix

        if self.need_hash_table:
            if hash_table is None:
                self.gen_hash_table()

            if hadamard_matrix is None:
                self.hadamard_matrix = hadamard(self.m)
                for i in range(len(self.hadamard_matrix)):
                    for j in range(len(self.hadamard_matrix[0])):
                        # change -1s to 0
                        h_i_j = self.hadamard_matrix[i, j]
                        self.hadamard_matrix[i, j] = 1 if h_i_j == 1 else 0
                self.hadamard_matrix = self.hadamard_matrix.astype(np.int8)

    def gen_hash_table(self):
        """
        Creates a table such that table[e, h] is the hash of e computed by h.
        Here, e and h are ints ranging in [0, U-1] and [0, k-1] respectively.
        The hash functions will be selected at random.
        """

        self.hash_table = np.zeros((self.U, self.k), dtype=np.uint16)
        for e in tqdm(range(self.U)):
            if self.U == self.m:
                # If the hash table size is the size of the universe, use the
                #  identity mapping.
                self.hash_table[e, :] = np.ones(self.k) * e
            else:
                self.hash_table[e, :] = np.random.randint(self.m, size=self.k)
        hash_mb = self.hash_table.nbytes / (1024 * 1024)
        print(f'Generated hash table ({hash_mb:.2f} MB)')

    def hadamard_original_bit(self, hash_index, l):
        """
        Returns the bit before being obfuscated
        """
        return self.hadamard_matrix[l, hash_index]

    def hash_record_index(self, original_record, j):
        """
        Returns the hash of the original_record by hash function j
        """
        if self.need_hash_table:
            return self.hash_table[original_record, j]
        return original_record

    def privatize_records(self, original_records, hadamard=False,
                          track_progress=False):
        """
        Privatize each record by hashing then perturbing

        Parameters
        ----------
        original_records : list[int]
            list of objects chosen by user
        hadamard : boolean (optional)
            obfuscate records with HCMS
            note: type of records returned will be list[(int, int, int)]
        track_progress: boolean (optional)
            whether to track progress with tqdm

        Returns
        -------
        list[(list[int], int)]
            list of obfuscated records
        """
        # probability of flipping each bit
        if self.eps == math.inf:
            self.p = 0
        elif hadamard:
            self.p = 1 / (1 + math.exp(self.eps))
        else:
            self.p = 1 / (1 + math.exp(self.eps / 2))

        privatized_records = []
        bar = tqdm(original_records, leave=False, desc="Privatizing: ") \
            if track_progress else original_records
        for original_record in bar:
            j = 0
            if self.need_hash_table:
                j = np.random.randint(self.k)
            hash_index = self.hash_record_index(original_record, j)

            if hadamard:
                l = np.random.randint(self.m)
                initial_bit = self.hadamard_original_bit(hash_index, l)
                to_flip = np.random.binomial(1, self.p)
                bit = np.abs(initial_bit - to_flip)  # XOR
                privatized_records.append((bit, j, l))
            else:
                v = np.zeros(self.m, dtype=int)
                v[hash_index] = 1
                bits_to_flip = np.random.binomial(1, self.p, len(v))
                privatized_record = np.abs(v - bits_to_flip)  # XOR
                privatized_records.append((privatized_record, j))

        return privatized_records
