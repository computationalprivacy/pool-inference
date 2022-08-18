"""
BPIA algorithm
note: this implementation uses a change of variable from gamma, delta to
alpha, beta. alpha = gamma * delta, beta = gamma * (1 - delta)
"""
import numpy as np
import quadpy
import mpmath as mp


class BayesianAttacker():
    def __init__(self, user, P_is, P_rest, phi_bar=None, hadamard=False):
        """
        An attacker that uses the BPIA algorithm to attempt to recover a given
        user's favourite pool

        Parameters
        ----------
        user : User
            user to attack
        P_is : list[list[int]]
            pools of interest to attacker
        P_rest : list[int]
            rest of objects in universe
        phi_bar : list[int] (optional)
            prior within pool probabilities of objects
        hadamard : boolean (optional)
            records were obfuscated with HCMS
        """
        self.user = user
        self.P_is = P_is
        self.P_rest = P_rest
        self.phi_bar = phi_bar
        self.p = user.cms.p
        self.hadamard = hadamard

        if phi_bar is None:
            # set uniform prior over U
            def phi_bar_fn(z):
                if isinstance(z, list):
                    if isinstance(z[0], list):
                        # list of pool of objects (e.g. P_is)
                        return sum([len(pool) for pool in z])
                    else:
                        return len(z)  # pool of objects
                else:
                    return 1  # single object
            self.phi_bar = phi_bar_fn

    def attack(self):
        """
        Perform attack on user

        Returns
        -------
        fav_pool: int
            pool with max score
        scores: list[float]
            list of scores for each pool
        """
        scores = []

        # memoize for performance
        self.generate_likelihoods_matrix()
        self.generate_pool_masses()
        self.generate_pool_sums_matrix()

        # memoize coeffs_matrix so that when integrating,
        # sum_pu, sum_pis and sum_rest are not recomputed
        # coeff_matrix[u][t] = [coeff_alpha, coeff_beta, constant]
        # s.t Pr[(x_tilde_t, j_t) | alpha, beta, P_u] = coeff_alpha * alpha +
        # coeff_beta * beta + constant
        n_pools = len(self.P_is)
        n_records = len(self.user.privatized_records)
        coeffs_matrix = []
        for u in range(n_pools):
            curr_matrix = []

            for t in range(n_records):
                curr_coeffs = self.prob_obs(t, u)
                curr_matrix.append(curr_coeffs)

            coeffs_matrix.append(curr_matrix)
        coeffs_matrix = np.array(coeffs_matrix)

        for u in range(len(self.P_is)):
            curr_matrix = coeffs_matrix[u]

            # define function to be integrated
            def integrand(v):
                params = zip(v[0], v[1])
                results = []
                for alpha, beta in params:
                    if len(curr_matrix) == 0:
                        results.append(1)
                    else:
                        var = np.array([alpha, beta, 1])
                        prod_arr = curr_matrix.dot(var)

                        # prevent underflow
                        prod_arr = prod_arr[prod_arr != 0.0]

                        if len(prod_arr) > 0:
                            # in order to combat overflow issues,
                            # calculate product of nth root values
                            root_arr = np.power(prod_arr, 1 / len(prod_arr))
                            initial = np.power(1, 1 / len(prod_arr))
                            result = np.prod(root_arr, initial=initial)
                            result = mp.power(result, len(prod_arr))
                            
                            # Note: 1 / (alpha + beta) is the Jacobian
                            # determinant, which is necessary for
                            # correctness with respect to the hierarchical
                            # model used in the paper. The original code
                            # used to produce the results in the paper
                            # (wrongly) did not include this line. In our
                            # tests the impact on the results is negligible.
                            result = result * 1 / (alpha + beta)

                            # and raise to the power of n
                            # using mpmath to retain high precision
                            results.append(result)
                        else:
                            results.append(0.0)
                return results


            # integration domain setup
            mid_pt = [1 / n_pools, 1 - 1 / n_pools]
            triangle = np.array([[0.0, 0.0], [1.0, 0.0], mid_pt])
            scheme = quadpy.t2.get_good_scheme(min(n_records + 1, 50))

            # run integration
            obj_prior = self.phi_bar(self.P_is[u])
            score = obj_prior * scheme.integrate(integrand, triangle)
            scores.append(score)

        return scores.index(max(scores)), scores

    def prob_obs(self, t, u):
        """
        Generate function for calculating probability of observing object t
        given P_u is the favourite pool

        Returns
        -------
        [coeff_alpha, coeff_beta, constant] s.t.
            Pr[(x_tilde_t, j_t) | alpha, beta, P_u] =
            coeff_alpha * alpha + coeff_beta * beta + constant
        """
        sum_pu = self.pool_sums[t][u] / self.pool_masses[u]
        sum_pis = 0
        for (i, P_i) in enumerate(self.P_is):
            if i != u:
                pool_ratio = self.pool_sums[t][i] / self.pool_masses[i]
                sum_pis += pool_ratio * 1 / (len(self.P_is) - 1)
        sum_rest = self.pool_sums[t][-1] / self.pool_masses[-1]

        coeff_alpha = sum_pu - sum_rest
        coeff_beta = sum_pis - sum_rest
        constant = sum_rest

        return [coeff_alpha, coeff_beta, constant]

    def generate_likelihoods_matrix(self):
        """
        Memoize Pr[(x_t, j_t) | z] for each x_t and z
        so that they are not recalculated for each P_u

        POST
        ----
        (CMS)
        self.likelihoods_matrix[t][z] = Pr[(x_t, j_t) | z]

        (HCMS)
        self.likelihoods_matrix[t][z] = Pr[(w_t, j_t, l_t) | z]
        """
        self.likelihoods_matrix = []
        # posterior probability that the bit wasn't flipped
        prob_same = 1 if self.p == 0 else (1 - self.p) / self.p
        # posterior probability that the bit was flipped
        prob_diff = 1 if self.p == 1 else self.p / (1 - self.p)

        if self.hadamard:
            for (w_t, j_t, l_t) in self.user.privatized_records:
                likelihoods_for_record = []
                for z in range(self.user.cms.U):
                    # hash record
                    v_index = self.user.cms.hash_record_index(z, j_t)
                    original_bit = self.user.cms.hadamard_original_bit(
                        v_index, l_t)

                    curr_prob = prob_same if original_bit == w_t else prob_diff
                    likelihoods_for_record.append(curr_prob)
                self.likelihoods_matrix.append(likelihoods_for_record)
        else:
            for (x_t, j_t) in self.user.privatized_records:
                likelihoods_for_record = []
                for z in range(self.user.cms.U):
                    v_index = self.user.cms.hash_record_index(z, j_t)
                    curr_prob = prob_same if x_t[v_index] == 1 else prob_diff
                    likelihoods_for_record.append(curr_prob)
                self.likelihoods_matrix.append(likelihoods_for_record)

    def generate_pool_masses(self):
        """
        Memoize pool masses for each pool in P_is and for P_rest
        also memoize the total pool mass of P_is so that the mass
        of P \\ P_i can be easily calculated

        POST
        ----
        self.pool_masses[i] = \\Phi(P_i)
        self.pool_masses[-1] = \\Phi(P_rest)
        """
        self.pool_masses = []

        for P_i in self.P_is:
            curr_pool_mass = self.phi_bar(P_i)
            self.pool_masses.append(curr_pool_mass)

        self.pool_masses.append(self.phi_bar(self.P_rest))

    def generate_pool_sums_matrix(self):
        """
        Memoize \\sum_{z \\in P_i} \\Phi(z)Pr[(x_t, j_t) | z]
        to be reused by Pr[(x_tilde_t, j_t) | alpha, beta, P_u]
        also memoize \\sum_{z \\in P} to be reused by \\sum_{z \\in P \\ P_i}

        POST
        ----
        self.pool_sums[t][i] = \\sum_{z \\in P_i} \\Phi(z)Pr[(x_t, j_t) | z]
        self.pool_sums[t][-1] =
            \\sum_{z \\in P_rest} \\Phi(z)Pr[(x_t, j_t) | z]
        """
        self.pool_sums = []

        for t in range(len(self.user.privatized_records)):
            curr_pool_sums = []
            for P_i in self.P_is:
                curr_pool_sum = 0
                for z in P_i:
                    curr_pool_sum = curr_pool_sum + \
                        self.phi_bar(z) * self.likelihoods_matrix[t][z]
                curr_pool_sums.append(curr_pool_sum)

            curr_pool_sum = 0
            for z in self.P_rest:
                curr_pool_sum = curr_pool_sum + \
                    self.phi_bar(z) * self.likelihoods_matrix[t][z]
            curr_pool_sums.append(curr_pool_sum)

            self.pool_sums.append(curr_pool_sums)
