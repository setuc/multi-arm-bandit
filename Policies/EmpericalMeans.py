import numpy as np

try:
    from .IndexPolicy import IndexPolicy
except ImportError:
    from IndexPolicy import IndexPolicy


class EmpiricalMeans(IndexPolicy):
    """ 
    The naive Empirical Means policy for bounded bandits: like UCB but without a bias correction term. 
    Note that it is equal to UCBalpha with alpha=0, only quicker."""

    def computeIndex(self, arm):
        """ 
        Compute the current index, at time t and after N pulls of arm k:
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            return self.rewards[arm] / self.pulls[arm]

    def computeAllIndex(self):
        """ 
        Compute the current indexes for all arms, in a vectorized manner.
        """
        indexes = self.rewards / self.pulls
        indexes[self.pulls < 1] = float('+inf')
        self.index[:] = indexes