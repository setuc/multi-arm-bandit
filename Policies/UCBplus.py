# -*- coding: utf-8 -*-
from math import sqrt, log
import numpy as np

try:
    from .UCB import UCB
except ImportError:
    from UCB import UCB


class UCBplus(UCB):
    """ 
    The UCB+ policy for bounded bandits, with a small trick on the index.
    - Reference: [Auer et al. 2002], and [[Garivier et al. 2016](https://arxiv.org/pdf/1605.08988.pdf)]
     (it is noted UCB in the second article).
    """

    def __str__(self):
        return "UCB+"

    def computeIndex(self, arm):
        """ 
        Compute the current index, at time t and after N_k(t) pulls of arm k:
        
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            return (self.rewards[arm] / self.pulls[arm]) + sqrt(max(0., log(self.t / (self.pulls[arm]))) / (2 * self.pulls[arm]))
