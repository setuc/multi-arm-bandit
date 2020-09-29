# -*- coding: utf-8 -*-
from math import sqrt, log
import numpy as np

try:
    from .UCB import UCB
except ImportError:
    from UCB import UCB

#: Default parameter for alpha
ALPHA = 1
ALPHA = 4


class UCBalpha(UCB):
    """ 
    The UCB1 (UCB-alpha) index policy, modified to take a random permutation order for the initial exploration of each arm (reduce collisions in the multi-players setting).
    Reference: [Auer et al. 02].
    """

    def __init__(self, nbArms, alpha=ALPHA, lower=0., amplitude=1.):
        super(UCBalpha, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        
        #: Parameter alpha
        self.alpha = alpha  

    def __str__(self):
        return r"UCB($\alpha={:.3g}$)".format(self.alpha)

    def computeIndex(self, arm):
        """ 
        Compute the current index, at time t and after N_k(t) pulls of arm k:
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            return (self.rewards[arm] / self.pulls[arm]) + sqrt((self.alpha * log(self.t)) / (2 * self.pulls[arm]))