import math
import numpy as np
from tqdm.auto import tqdm


class CMSServer():
    """
    CMS server algorithm that calculates the approximation of the frequency
    of each object based on the privatized records
    """
    def __init__(self, privatized_records, cms, hadamard=False, M=None):
        """
        Define CMS Server

        Parameters
        ----------
        privatized_records : list[(list[int], int)]
            list of all privatized records
        cms : CMS
            cms object used to obfuscate records
        hadamard : boolean (optional)
            records were obfuscated with HCMS
            note: type of records should be list[(int, int, int)]
        M : np.ndarray (optional)
            k x m sketch matrix to update (for multiprocessing)
        """
        self.privatized_records = privatized_records
        self.cms = cms
        self.p = 1 / (1 + math.exp(cms.eps / 2))
        self.hadamard = hadamard
        self.M = np.zeros((self.cms.k, self.cms.m)) if M is None else M

    def sketch_cms(self, calc_freqs=True, track_progress=False):
        """
        Run Sketch-CountMeanSketch (A_server) algorithm

        Parameters
        ----------
        track_progress : boolean (optional)
            track progress using tqdm
        calc_freqs : boolean (optional)
            whether to stop after creating sketch matrix or calculate freqs
            (for multiprocessing, set to false for optimization)
        """
        if self.cms.eps == math.inf:
            c_eps = 1
        else:
            e_eps = math.exp(self.cms.eps / 2)
            c_eps = (e_eps + 1) / (e_eps - 1)
        U = self.cms.U
        k = self.cms.k
        m = self.cms.m
        n = len(self.privatized_records)

        # pre processing, convert records from being 0/1 to -1/+1 and applying
        # transformation from CMS Server algorithm
        bar = tqdm(self.privatized_records, leave=False,
                   desc="Preprocessing: ")
        bar = enumerate(bar) if track_progress else \
            enumerate(self.privatized_records)
        halves = np.ones((m, )) * 0.5
        for (i, record) in bar:
            curr_record = np.where(record[0] == 1, 1, -1)
            curr_record = k * ((c_eps / 2) * curr_record + halves)
            self.privatized_records[i] = curr_record, record[1]

        # calculate sketch matrix
        bar = tqdm(range(n), leave=False, desc="Generating sketch matrix: ") \
            if track_progress else range(n)
        for i in bar:
            for l in range(m):
                x_i, j_i = self.privatized_records[i]
                self.M[j_i, l] += x_i[l]

        if calc_freqs:
            # extract freqs from sketch matrix
            f_ds = []
            bar = tqdm(range(U), leave=False, desc="Generating freqs: ") \
                if track_progress else range(U)
            for d in bar:
                curr_sum = 0
                for l in range(k):
                    curr_sum = curr_sum + \
                        self.M[l, self.cms.hash_record_index(d, l)]
                f_ds.append((m / (m - 1)) * ((1 / k) * curr_sum - (n / m)))

            return f_ds

    def sketch_hcms(self, calc_freqs=True):
        """
        Run Sketch-HadamardCountMeanSketch (A_server) algorithm

        Parameters
        ----------
        track_progress : boolean (optional)
            track progress using tqdm
        calc_freqs : boolean (optional)
            whether to stop after creating sketch matrix or calculate freqs
            (for multiprocessing, set to false for optimization)
        """
        e_eps = math.exp(self.cms.eps)
        c_eps = (e_eps + 1) / (e_eps - 1)
        U = self.cms.U
        k = self.cms.k
        m = self.cms.m
        n = len(self.privatized_records)

        # pre processing, convert records (w_i, j_i, l_i) to (x_i, j_i, l_i)
        processed_records = []
        for (w_i, j_i, l_i) in self.privatized_records:
            w_i = 1 if w_i == 1 else -1
            x_i = k * c_eps * w_i
            processed_records.append((x_i, j_i, l_i))

        # calculate sketch matrix
        M = np.zeros((k, m))
        for (x_i, j_i, l_i) in processed_records:
            M[j_i, l_i] += x_i

        H_T = np.transpose(self.cms.hadamard_matrix)
        M = np.matmul(M, H_T)

        if calc_freqs:
            # extract freqs from sketch matrix
            f_ds = []
            for d in range(U):
                curr_sum = 0
                for l in range(k):
                    curr_sum = curr_sum + \
                        M[l, self.cms.hash_record_index(d, l)]
                f_ds.append((m / (m - 1)) * ((1 / k) * curr_sum - (n / m)))

            return f_ds
