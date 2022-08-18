import pandas as pd
import os
import pickle
import uuid
from pathlib import Path


class ExpParams:
    """
    Wrapper object that encapsulates all parameters for running experiments
    """

    def __init__(self, cms, pool_sizes, n=1, reps=1, EXP_DB=None,
                 recompute=False, max_workers=38, hadamard=False,
                 exp_db_filename=None, prior=None, p_omega=None,
                 sigma=None):

        # cms & pool_sizes required for ExpParams
        self.cms = cms
        self.pool_sizes = pool_sizes

        self.n = n
        self.reps = reps
        self.recompute = recompute
        self.max_workers = max_workers
        self.hadamard = hadamard
        self.prior = prior
        self.p_omega = p_omega
        self.sigma = sigma

        # make sure there is space for output to be written to
        if exp_db_filename is None:
            unique_id = uuid.uuid4().hex
            exp_db_filename = f'pickles/experiments/{unique_id}.pickle'
            print(f'No output filename given, writing to {exp_db_filename}')
        parent_folder = str(Path(exp_db_filename).parent.absolute())
        os.makedirs(f'{parent_folder}/exp_pickles', exist_ok=True)
        self.exp_db_filename = exp_db_filename

        # initialize EXP_DB
        if os.path.isfile(exp_db_filename):
            # EXP_DB has been previously generated, load it
            EXP_DB = pickle.load(open(exp_db_filename, "rb"))
        elif EXP_DB is None:
            # create EXP_DB
            column_names = ['unique_id', 'U', 'm', 'k', 'pool_sizes',
                            'eps', 'n', 'accuracy', 'hadamard']
            EXP_DB = pd.DataFrame(columns=column_names)
        self.EXP_DB = EXP_DB
