class User:
    """
    Interface to model users (sampling and obfuscation).
    """

    def __init__(self, distribution=None, cms=None, user_records=None):
        """
        Define user with a distribution or with a set of user records

        Parameters
        ----------
        distribution : BaseDistribution
            user's distribution to sample objects from.
        cms : CMS
            user's CountMeanSketch object to obfuscate records
        user_records : list[int]
            instead of providing a distribution, a set of records belonging to
            the user.

        Exactly one of distribution or user_records must be provided.
        """
        assert (distribution is None) or (user_records is None), \
            "Cannot provide both a distribution and user records."
        assert not ((distribution is None) and (user_records is None)), \
            "Must provide either a distribution or user records."
        self.distribution = distribution
        self.cms = cms
        self.user_records = user_records
        self.original_records = []
        self.privatized_records = []

    def gen_obs(self, n, hadamard=False, track_progress=False):
        """
        Generate/choose n samples and privatize them

        Parameters
        ----------
        n : int
            number of objects to sample
        hadamard : bool (optional)
            use HCMS to obfuscate instead
        track_progress: boolean (optional)
            whether to track progress with tqdm
        """
        if self.user_records is None:
            self.original_records = self.distribution.sample(n)
        else:
            self.original_records = self.user_records[:n]
        self.privatized_records = self.cms.privatize_records(
            self.original_records, hadamard=hadamard,
            track_progress=track_progress)
