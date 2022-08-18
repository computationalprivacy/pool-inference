# module imports
import numpy as np
import copy
import pickle
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import uuid
from tqdm.auto import tqdm
from multiprocessing.shared_memory import SharedMemory
import json

# my imports
from cms import CMS
from base_distribution import BaseDistribution
from user import User
from bayesian_attacker import BayesianAttacker
from cms_server import CMSServer


def estimate_prior(f_ds, total, niter=1000):
    """
    Utility function to convert the f_ds generated from the CMSServer
    into a proper probability distribution

    Parameters
    ----------
    f_ds : list[float]
        frequencies from running the CMSServer algorithm on auxiliary data
    total : int
        total number of objects that the CMSServer algorithm was run on
    niter : int (optional)
        max number of iterations for convergence

    Returns
    -------
    list[float]
        prior probabilities of choosing each object
    """
    p = np.array(f_ds) / total
    m = len(p)
    old_p = p

    # The following loop uses averaged projection steps to project the
    # noisy distribution to the n-dimensional simplex, which can be written
    # as the intersection of {p: p >= 0} and {p: sum(p) = 1}.
    for _ in range(niter):
        # Step 1: project over {p: p>=0}.
        p1 = np.maximum(p, 0)
        # Step 2: projection over {p: sum(p) = 1}.
        p2 = p + (1 - p.sum()) / m
        # Averaged projection step.
        p = (p1 + p2) / 2
        err = np.sqrt(np.mean((old_p - p)**2))
        if err < 1e-6:
            break
        old_p = p

    if _ >= niter - 1:
        # Decide what to do if there's an issue here.
        print(f'Did not converge (RMSE: {err})')

    # Ensure this is a probability (technically adds error, but fine).
    p = np.maximum(p, 0)
    p = p / p.sum()
    return p


def load_prior(U, m, k, eps, pool_sizes, p_omega=None,
               prior_filename=None, n=1000000, objects=None):
    """
    Check if the prior probabilities for the set of parameters already exists
    if not generate them, save and return the filename

    Parameters
    ----------
    U, m, k, eps, pool_sizes
        parameters for CMS
    p_omega : string
        filename for within pool probabilities of users
    prior_filename : string
        location to save prior to
    n : int
        number of objects to generate for all users to be used as aux data
    objects : list[int]
        instead of providing p_omega and n, a list of objects from the aux data
        can be supplied to generate the prior

    """

    # check if prior file for these params already exists
    if os.path.isfile(prior_filename):
        # prior file already exists
        return prior_filename

    if objects is None:
        p_omega = pickle.load(open(p_omega, 'rb'))
    else:
        p_omega = None
        n = len(objects)

    # load cms
    cms = load_cms(U, m, k, eps=eps, shared=False)

    # create probability distribution to encompass records
    class SpecialDistribution:
        def __init__(self, univ_size, p_omega=None):
            self.universe = np.arange(univ_size)
            self.p_omega = p_omega

        def sample(self, n=1):
            if objects is not None:
                return objects[:n]

            return np.random.choice(self.universe, size=n, p=self.p_omega)

    # prepare and privatize records
    user = User(SpecialDistribution(U, p_omega), cms)
    user.gen_obs(n, track_progress=True)

    # run sketch cms and extract prior
    cms_server = CMSServer(user.privatized_records, user.cms)
    f_ds = cms_server.sketch_cms(track_progress=True)
    prior = estimate_prior(f_ds, n)

    # write prior to file
    pickle.dump(prior, open(prior_filename, "wb"))


def get_cms_values(cms):
    """
    Extract params from cms
    """
    if isinstance(cms, tuple):
        # shared memory
        return cms
    else:
        # actual object
        return cms.U, cms.m, cms.eps, cms.k, None, None


def set_cms_eps(cms, eps):
    """
    Set epsilon value for cms
    """
    if isinstance(cms, tuple):
        # shared memory
        return (cms[0], cms[1], eps, cms[3], cms[4], cms[5])
    else:
        # actual object
        cms.eps = eps
        return cms


def params_exists(exp_params):
    """
    Construct query to check if experiment has already been done
    """
    U, m, eps, k, _, _ = get_cms_values(exp_params.cms)

    pool_sizes_str = str(tuple(exp_params.pool_sizes))
    params_query = (
        (exp_params.EXP_DB['U'] == U) &
        (exp_params.EXP_DB['m'] == m) &
        (exp_params.EXP_DB['k'] == k) &
        (exp_params.EXP_DB['n'] == exp_params.n) &
        (exp_params.EXP_DB['pool_sizes'].astype(str) == pool_sizes_str) &
        (exp_params.EXP_DB['hadamard'] == exp_params.hadamard))

    return params_query


def sample_gamma_delta(num_pools):
    """
    Sample gamma and delta uniformly from a rectangle
    """
    gamma = np.random.uniform()
    delta = np.random.uniform(1 / num_pools, 1)

    return gamma, delta


def load_cms(U, m, k, eps=4, shared=False, save_to_file=True):
    """
    Check if CMS object exists if not create it and save it

    Parameters
    ----------
    U, m, k, eps
        parameters for CMS
    shared : bool
        load CMS object in shared memory and return a tuple reference to it
    save_to_file : bool
        save CMS object to file
    """
    if U == m:
        # non-private setting
        if shared:
            return (U, m, eps, k, None, None)
        return CMS(U=U, m=m, k=k, eps=eps)

    hash_table = None
    hadamard_matrix = None

    hash_folder = 'pickles/hash_tables/'
    hash_table_filename = f'{hash_folder}/hash_table_{U}_{m}_{k}.pickle'
    if os.path.isfile(hash_table_filename):
        # hash table has been previously generated for these params
        # load hash table from file
        hash_table = pickle.load(open(hash_table_filename, "rb"))

    hadamard_folder = 'pickles/hadamard_matrices/'
    hadamard_matrix_filename = f'{hadamard_folder}/hadamard_{m}.pickle'
    if os.path.isfile(hadamard_matrix_filename):
        # hadamard matrix has been previously generated for these params
        # load hadamard matrix from file
        hadamard_matrix = pickle.load(open(hadamard_matrix_filename, "rb"))

    cms = CMS(U=U, m=m, k=k, eps=eps, hash_table=hash_table,
              hadamard_matrix=hadamard_matrix)

    if save_to_file:
        if not os.path.isfile(hash_table_filename):
            # save hash table for reuse
            pickle.dump(cms.hash_table, open(hash_table_filename, "wb"))

        if not os.path.isfile(hadamard_matrix_filename):
            # save hadamard matrix for reuse
            pickle.dump(
                cms.hadamard_matrix, open(
                    hadamard_matrix_filename, "wb"))

    if not shared:
        return cms

    # Share cms on shared_memory to conserve space across processes
    shm_hash_table = SharedMemory(create=True, size=cms.hash_table.nbytes)
    shared_hash_table = \
        np.ndarray(shape=cms.hash_table.shape,
                   dtype=cms.hash_table.dtype,
                   buffer=shm_hash_table.buf)
    shared_hash_table[:] = cms.hash_table[:]

    shm_hadamard_matrix = SharedMemory(
        create=True, size=cms.hadamard_matrix.nbytes)
    shared_hadamard_matrix = \
        np.ndarray(shape=cms.hadamard_matrix.shape,
                   dtype=cms.hadamard_matrix.dtype,
                   buffer=shm_hadamard_matrix.buf)
    shared_hadamard_matrix[:] = cms.hadamard_matrix[:]

    return (
        cms.U,
        cms.m,
        cms.eps,
        cms.k,
        shm_hash_table.name,
        shm_hadamard_matrix.name)


def unload_cms(shared_cms):
    """
    Unload CMS from shared memory
    """
    if isinstance(shared_cms, tuple):
        shm = SharedMemory(shared_cms[4])
        shm.close()
        shm.unlink()

        shm = SharedMemory(shared_cms[5])
        shm.close()
        shm.unlink()


def single_rep(exp_params, user_seed=None):
    """
    Run attack experiment on a single user. This involves:
    1) define user distribution from params and randomly chosen favourite pool
    2) generate observations
    3) shuffle P_u and P_is for attacker to ensure no bias
    4) run attack, get scores
    5) return whether this attack was a success and the list of scores

    Parameters
    ----------
    exp_params : ExpParams
        experiment parameters
    user_seed : int
        seed for current user to ensure that when running experiment on the
        same user across different number of observations, the objects
        are consistent throughout

    Returns
    -------
    success : int (0/1)
        whether the attacker correctly identified the user's favourite pool
    scores : list[float]
        list of scores for each pool
    gamma : float
        gamma chosen for current user
    delta : float
        delta chosen for current user
    """

    if user_seed is not None:
        np.random.seed(user_seed)

    # sample gamma and delta uniformly and perform change of variable to
    # alpha, beta for integration
    num_pools = len(exp_params.pool_sizes)
    gamma, delta = sample_gamma_delta(num_pools)

    # if cms is a shared object construct object from tuple
    cms = exp_params.cms
    if isinstance(cms, tuple):
        U, m, eps, k, shared_hash_table_name, shared_hadamard_matrix_name = cms
        hash_table = None
        if shared_hash_table_name is not None:
            shared_hash_table = SharedMemory(shared_hash_table_name)
            hash_table = np.ndarray(
                shape=(
                    U,
                    k),
                dtype=np.uint16,
                buffer=shared_hash_table.buf)

        hadamard_matrix = None
        if shared_hadamard_matrix_name is not None:
            shared_hadamard_matrix = SharedMemory(
                shared_hadamard_matrix_name)
            hadamard_matrix = np.ndarray(
                shape=(
                    m,
                    m),
                dtype=np.int8,
                buffer=shared_hadamard_matrix.buf)
        cms = CMS(U, m, eps, k, hash_table=hash_table,
                  hadamard_matrix=hadamard_matrix)

    # setup user to be attacked
    iota = np.random.randint(num_pools)  # choose favourite pool randomly
    p_omega = pickle.load(open(exp_params.p_omega, 'rb'))  # load user dist
    if exp_params.sigma is not None:  # (optional) add noise to user dist
        p_omega = p_omega + \
            np.random.normal(scale=exp_params.sigma, size=p_omega.shape)
        p_omega += np.abs(p_omega.min())
        p_omega /= p_omega.sum()
    universe = [x for x in range(cms.U)]  # setup pools of interest
    P_u = []
    P_is = []
    next_index = 0
    for i in range(0, len(exp_params.pool_sizes)):
        if i == iota:
            P_u = universe[next_index: next_index +
                           exp_params.pool_sizes[i]]
        else:
            P_is.append(
                universe[next_index: next_index + exp_params.pool_sizes[i]])
        next_index = next_index + exp_params.pool_sizes[i]
    P_rest = universe[next_index: cms.U]
    dist = BaseDistribution(gamma, delta, P_u, P_is, P_rest, p_omega)
    user = User(dist, cms)
    user.gen_obs(exp_params.n, hadamard=exp_params.hadamard)  # generate objs

    phi_bar = None
    if exp_params.prior is not None:  # load prior
        prior = pickle.load(open(exp_params.prior, 'rb'))

        def phi_bar(x):
            return prior[x].sum()

    # shuffle pools around to ensure attacker cannot be biased
    P_is = copy.deepcopy(dist.P_is)
    P_is.append(dist.P_u.copy())
    P_rest = dist.P_rest
    np.random.shuffle(P_is)

    # define and run attack
    attacker = BayesianAttacker(
        user,
        P_is,
        P_rest,
        phi_bar=phi_bar,
        hadamard=exp_params.hadamard)
    P_u_index, scores = attacker.attack()

    # we only need to compare the first elem of the lists
    # to determine if they are the same pool since an elem
    # can only be in 1 pool
    result = 1 if dist.P_u[0] == P_is[P_u_index][0] else 0

    return result, scores, gamma, delta


def attack(exp_params, user_seeds=None, pbar=None):
    """
    Runs attack experiment (in parallel) on many users averaging the results

    Parameters
    ----------
    exp_params : ExpParams
        experiment parameters
    user_seeds : list[int] (optional)
        seeds to ensure consistency when running attack on the same set of
        users across different number of observations
    pbar: tqdm (optional)
        a predefined progress bar to be used

    Returns
    -------
    accuracy : float
        accuracy of attack on all users
    results : list[(int, list[float], float, float)]
        list of (accuracy, scores list, gamma, delta) from `single_exp`
    """
    if pbar is None:
        pbar = tqdm(total=exp_params.reps)
    with ProcessPoolExecutor(max_workers=exp_params.max_workers) as executor:
        # start attack for each user in parallel
        futures = []
        for i in range(exp_params.reps):
            user_seed = user_seeds[i] if user_seeds is not None else None
            futures.append(executor.submit(single_rep, exp_params, user_seed))

        # wait for attacks on all users to finish
        results = []
        for future in as_completed(futures):
            pbar.update(1)
            results.append(future.result())

        # average accuracy and return
        accuracy = sum([row[0] for row in results]) / len(results)
        return accuracy, results


def single_exp(exp_params, user_seeds=None, pbar=None):
    """
    Runs attack experiment (in parallel) on many users and save results to
    pandas dataframe and to filesystem

    Parameters
    ----------
    exp_params : ExpParams
        experiment parameters
    user_seeds : list[int]
        seeds to ensure consistency when running attack on the same set of
        users across different number of observations
    pbar: tqdm (optional)
        a predefined progress bar to be used
    """
    # get parameters of cms
    U, m, eps, k, _, _ = get_cms_values(exp_params.cms)

    # check if experiment previously exists
    params_query = params_exists(exp_params)
    if params_query.any():
        # row with params previously exists if recompute is set, delete rows.
        # else skip current experiment
        if exp_params.recompute:
            row_indexes = exp_params.EXP_DB[params_query].index.tolist()
            exp_params.EXP_DB.drop(row_indexes, inplace=True)
        else:
            return

    # get result from attack
    result = attack(exp_params, user_seeds=user_seeds, pbar=pbar)
    accuracy, results = result

    # update dataframe
    unique_id = uuid.uuid4().hex[:5]  # unique id for filename
    record = {
        'unique_id': unique_id,
        'U': U,
        'm': m,
        'k': k,
        'pool_sizes': tuple(exp_params.pool_sizes),
        'eps': eps,
        'n': exp_params.n,
        'accuracy': accuracy,
        'hadamard': exp_params.hadamard,
    }
    exp_params.EXP_DB = exp_params.EXP_DB.append(record, ignore_index=True)

    # write intermediate results to file
    dump_dir = f'{os.path.dirname(exp_params.exp_db_filename)}/exp_pickles/'
    pickle.dump(results, open(f'{dump_dir}/{unique_id}.pickle', "wb"))

    return


def run_utility_vs_eps_helper(n=1000, U=2000, m=1024, k=65536, eps=8,
                              p_omega_f=None, seed=None, shared_M_name=None):
    """
    Process a single batch of observations for utility vs eps experiment

    Parameters
    ----------
    n : int
        number of observations
    U, m, k, eps
        parameters for CMS
    p_omega_f : string
        filename where user distribution is stored
    seed : int
        seed for ensuring diversity between helper processes
    shared_M_name : string
        name of sketch matrix in Shared Memory
    """

    # load sketch matrix from shared memory
    shared_M = SharedMemory(shared_M_name)
    M = np.ndarray(shape=(k, m), dtype=np.float64, buffer=shared_M.buf)

    if seed is not None:
        # make sure different helpers dont generate same results
        np.random.seed(seed)

    p_omega = pickle.load(open(p_omega_f, 'rb'))

    cms = load_cms(U, m, k, eps=eps, shared=False)

    # create probability distribution
    class SpecialDistribution:
        def __init__(self, univ_size, p_omega):
            self.universe = np.arange(univ_size)
            self.p_omega = p_omega

        def sample(self, n=1):
            return np.random.choice(self.universe, size=n, p=self.p_omega)

    user = User(SpecialDistribution(U, p_omega), cms)
    user.gen_obs(n)

    cms_server = CMSServer(user.privatized_records, cms, M=M)
    cms_server.sketch_cms(calc_freqs=False, track_progress=False)
    return


def run_utility_vs_eps(n, U, m, k, eps, p_omega_f, shm_Ms, shared_Ms,
                       max_workers=38):
    """
    Run experiment to calculate the utility of CMSServer at various privacy
    parameters (eps) and number of observations (n) (Table 8)

    Parameters
    ----------
    n : int
        number of observations
    U, m, k, eps
        parameters for CMS
    p_omega_f : string
        filename where user distribution is stored
    shm_Ms, shared_Ms : list[string]
        shared memory objects for holding the sketch matrices
    max_workers : int
        max number of workers/cpus that can be used for parallelization

    Returns
    -------
    list[float]
        frequencies for each object in universe (f_ds) from CMS Server
    """
    cms = load_cms(U, m, k, eps=eps, shared=False)

    # make sure shared arrays are zeroed
    M = np.zeros((k, m))
    for shared_M in shared_Ms:
        shared_M[:] = M[:]

    # split large n into smaller batches to be processed in parallel
    n_part_size = 10000
    ns = []
    seeds = []
    tmp_n = n
    while tmp_n > 0:
        curr_n = n_part_size if tmp_n > n_part_size else tmp_n
        ns.append(curr_n)
        tmp_n -= curr_n
        seeds.append(np.random.randint(0, high=2147483647))

    with tqdm(total=len(ns), leave=False) as pbar, \
         ProcessPoolExecutor(max_workers=max_workers) as executor:
        # start CMSServer for each batch of observations in parallel
        futures = []
        for (proc_id, (curr_n, seed)) in enumerate(zip(ns, seeds)):
            futures.append(executor.submit(
                run_utility_vs_eps_helper, curr_n, U, m, k, eps, p_omega_f,
                seed, shm_Ms[proc_id % max_workers].name))

        # wait for all batches to finish
        results = []
        for future in as_completed(futures):
            results.append(future.result())
            pbar.update(1)

    # combine all sketch matrices into one
    for shared_M in shared_Ms:
        M += shared_M

    # calculate frequencies from combined sketch matrix
    f_ds = []
    for d in tqdm(range(U), leave=False):
        curr_sum = 0
        for l in range(k):
            curr_sum = curr_sum + M[l, cms.hash_record_index(d, l)]
        f_ds.append((m / (m - 1)) * ((1 / k) * curr_sum - (n / m)))

    return f_ds


def attack_real_user(user_id, n, users_file, cms, pool_sizes, prior,
                     user_seed):
    """
    Similar to `single_rep` but attacks a real user instead of modelling one
    1) load user from `users_file` at `user_id`
    2) select first n observations
    3) run attack, get scores
    4) return whether this attack was a success, the list of scores

    Parameters
    ----------
    user_id : string
        id of user to load from all users
    n : int
        number of observations to restrict user to
    users_file : string
        file containing observations from all users (in JSON format)
    cms : CMS | tuple (if loaded onto shared memory)
        CMS object for privatizing records
    pool_sizes : list[int]
        list of pool sizes
    prior : string
        file to load prior distribution from
    user_seed : int
        seed to ensure consistency when running attack on the same
        user across different number of observations


    Returns
    -------
    success : int (0/1)
        whether the attacker correctly identified the user's favourite pool
    scores : list[float]
        list of scores for each pool
    gamma : float
        gamma chosen for current user
    delta : float
        delta chosen for current user
    """
    np.random.seed(user_seed)

    # load user
    with open(users_file, 'r') as f:
        users_data = json.load(f)
    user_data = users_data[user_id]

    # get gamma, delta
    gamma, delta = user_data['gamma'], user_data['delta']

    if isinstance(cms, tuple):
        # load cms from shared memory
        U, m, eps, k, shared_hash_table_name, shared_hadamard_matrix_name = cms
        hash_table = None
        if shared_hash_table_name is not None:
            shared_hash_table = SharedMemory(shared_hash_table_name)
            hash_table = np.ndarray(
                shape=(
                    U,
                    k),
                dtype=np.uint16,
                buffer=shared_hash_table.buf)

        hadamard_matrix = None
        if shared_hadamard_matrix_name is not None:
            shared_hadamard_matrix = SharedMemory(shared_hadamard_matrix_name)
            hadamard_matrix = np.ndarray(
                shape=(
                    m,
                    m),
                dtype=np.int8,
                buffer=shared_hadamard_matrix.buf)
        cms = CMS(U, m, eps, k, hash_table=hash_table,
                  hadamard_matrix=hadamard_matrix)

    # get favourite pool
    iota = user_data['fav_pool']

    # prepare pools for attacker
    universe = [x for x in range(cms.U)]
    P_is = []
    next_index = 0
    for pool_size in pool_sizes:
        P_is.append(universe[next_index: next_index + pool_size])
        next_index += pool_size
    P_rest = universe[next_index:]

    # choose first n records to privatize
    user = User(None, cms, user_records=user_data['records'])
    user.gen_obs(n)

    phi_bar = None
    if prior is not None:
        # load prior
        prior = pickle.load(open(prior, 'rb'))

        def phi_bar(x):
            return prior[x].sum()

    # run attack
    attacker = BayesianAttacker(user, P_is, P_rest, phi_bar=phi_bar)
    P_u_index, scores = attacker.attack()

    # since the pools are not shuffled in this case, we just need to check that
    # the identified pool index is the expected one
    result = 1 if P_u_index == iota else 0

    return result, scores, gamma, delta


def attack_real_users(n, user_ids, users_file, user_seeds, cms, pool_sizes,
                      prior, max_workers=38, pbar=None):
    """
    Similar to `attack` but attacks real users instead of modelling them

    Parameters
    ----------
    n : int
        number of observations to restrict user to
    user_ids : string
        list of ids present in `users_file`
    users_file : string
        file containing observations from all users (in JSON format)
    cms : CMS | tuple (if loaded onto shared memory)
        CMS object for privatizing records
    pool_sizes : list[int]
        list of pool sizes
    prior : string
        file to load prior distribution from
    user_seed : int
        seed to ensure consistency when running attack on the same
        user across different number of observations


    Returns
    -------
    accuracy : float
        accuracy of attack on all users
    results : list[(int, list[float], float, float)]
        list of (accuracy, scores list, gamma, delta) from `attack_real_user`
    """
    if pbar is None:
        pbar = tqdm(total=len(user_ids))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # start attack for each user in parallel
        futures = []
        for user_id, user_seed in zip(user_ids, user_seeds):
            futures.append(executor.submit(
                attack_real_user,
                user_id=user_id,
                n=n,
                users_file=users_file,
                cms=cms,
                pool_sizes=pool_sizes,
                prior=prior,
                user_seed=user_seed))

        # wait for attack on all users to finish
        results = []
        for future in as_completed(futures):
            pbar.update(1)
            results.append(future.result())

    # average accuracy and return results
    accuracy = sum([row[0] for row in results]) / len(results)
    return accuracy, results
