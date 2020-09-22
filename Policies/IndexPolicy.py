# -*- coding: utf-8 -*-
""" Generic index policy.
- If rewards are not in [0, 1], be sure to give the lower value and the amplitude. Eg, if rewards are in [-3, 3], lower = -3, amplitude = 6.
"""

import numpy as np

try:
    from .BasePolicy import BasePolicy
except (ImportError, SystemError):
    from BasePolicy import BasePolicy


class IndexPolicy(BasePolicy):
    """ Class that implements a generic index policy."""

    def __init__(self, nbArms, lower=0., amplitude=1.):
        """ New generic index policy.
        - nbArms: the number of arms,
        - lower, amplitude: lower value and known amplitude of the rewards.
        """
        super(IndexPolicy, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        
        # Numerical index for each arms
        self.index = np.zeros(nbArms) 


    def startGame(self):
        """ 
        Initialize the policy for a new game.
        """
        super(IndexPolicy, self).startGame()
        self.index.fill(0)

    def computeIndex(self, arm):
        """
        Compute the current index of arm 'arm'.
        """
        raise NotImplementedError("This method computeIndex(arm) has to be implemented in the child class inheriting from IndexPolicy.")

    def computeAllIndex(self):
        """ 
        Compute the current indexes for all arms. 
        """
        for arm in range(self.nbArms):
            self.index[arm] = self.computeIndex(arm)

    # --- Basic choice() method

    def choice(self):
        r""" In an index policy, choose an arm with maximal index (uniformly at random):
        .. math:: A(t) \sim U(\arg\max_{1 \leq k \leq K} I_k(t)).
        .. warning:: In almost all cases, there is a unique arm with maximal index, so we loose a lot of time with this generic code, but I couldn't find a way to be more efficient without loosing generality.
        """
        # I prefer to let this be another method, so child of IndexPolicy only needs to implement it (if they want, or just computeIndex)
        self.computeAllIndex()
        # Uniform choice among the best arms
        try:
            return np.random.choice(np.nonzero(self.index == np.max(self.index))[0])
        except ValueError:
            print("Warning: unknown error in IndexPolicy.choice(): the indexes were {} but couldn't be used to select an arm.".format(self.index))
            return np.random.randint(self.nbArms)