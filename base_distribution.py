import itertools
import numpy as np
from scipy.stats import zipfian


def random_pdf(n):
    x = np.random.random(size=(n,))
    return x / x.sum()


def zip_pdf(kappa, n):
    universe = np.arange(1, n + 1)
    x = zipfian.pmf(universe, kappa, n)
    return x / x.sum()


class BaseDistribution:
    """
    Common interface to model different user distributions
    """

    def __init__(self, gamma, delta, P_u, P_is, P_rest, probs):
        """
        Define distribution based on pre-defined pools

        Parameters
        ----------
        gamma : float
            relevant interest
        delta : float
            polarization
        P_u : list[int]
            objects in user's favourite pool (chosen with prob gamma*delta)
        P_is : list[list[int]]
            pools of interest to attacker (chosen with prob gamma*(1 - delta))
        P_rest : list[int]
            rest of objects in universe (chosen with prob 1-gamma)
        probs : list[float]
            probabilities of choosing object within each pool
        """
        self.gamma = gamma
        self.delta = delta

        self.P_u = P_u
        self.P_is = P_is
        self.P_alt = list(itertools.chain.from_iterable(self.P_is))
        self.P_rest = P_rest

        self.probs = probs

        # correct numerical errors and normalize probs
        self.probs[self.P_u] /= np.sum(self.probs[self.P_u])
        for P_i in self.P_is:
            self.probs[P_i] /= np.sum(self.probs[P_i])
            self.probs[P_i] *= 1 / len(self.P_is)
        self.probs[self.P_rest] /= np.sum(self.probs[self.P_rest])

    def sample(self, n=1):
        """
        Generate samples from distribution

        Parameters
        ----------
        n : int
            number of objects to sample

        Returns
        -------
        list[int]
            list of sampled objects
        """
        samples = np.zeros((n,))

        y = np.random.random(size=(n,))
        use_P_u = (y <= (self.delta * self.gamma))
        n_P_u = use_P_u.sum()
        use_P_alt = (y > (self.delta * self.gamma)) & (y <= self.gamma)
        n_P_alt = use_P_alt.sum()
        use_P_rest = y > self.gamma
        n_P_rest = use_P_rest.sum()

        samples[use_P_u] = np.random.choice(self.P_u, size=n_P_u,
                                            p=self.probs[self.P_u])
        samples[use_P_alt] = np.random.choice(self.P_alt, size=n_P_alt,
                                              p=self.probs[self.P_alt])
        samples[use_P_rest] = np.random.choice(self.P_rest, size=n_P_rest,
                                               p=self.probs[self.P_rest])

        return samples.astype(int)
