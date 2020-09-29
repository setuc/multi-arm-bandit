# -*- coding: utf-8 -*-
from math import sqrt, log
import numpy as np
try:
    from .UCB import UCB
except ImportError:
    from UCB import UCB


class UCBV(UCB):
    """
    The UCB-V policy for bounded bandits, with a variance correction term.
    Reference: [Audibert, Munos, & Szepesvári - Theoret. Comput. Sci., 2009].
    """
    def __str__(self):
        return "UCB-V"

    def __init__(self, nbArms, lower=0., amplitude=1.):
        super(UCBV, self).__init__(nbArms, lower=lower, amplitude=amplitude)
        self.rewardsSquared = np.zeros(self.nbArms)  #: Keep track of squared of rewards, to compute an empirical variance

    def startGame(self):
        super(UCBV, self).startGame()
        self.rewardsSquared.fill(0)

    def getReward(self, arm, reward):
        """
        Give a reward: increase t, pulls, and update cumulated sum of rewards and of rewards squared for that arm (normalized in [0, 1]).
        """
        super(UCBV, self).getReward(arm, reward)
        self.rewardsSquared[arm] += ((reward - self.lower) / self.amplitude) ** 2

    def computeIndex(self, arm):
        """ 
        Compute the current index, at time t and after N_k(t) pulls of arm k:
        """
        if self.pulls[arm] < 1:
            return float('+inf')
        else:
            mean = self.rewards[arm] / self.pulls[arm]   # Mean estimate
            variance = (self.rewardsSquared[arm] / self.pulls[arm]) - mean ** 2  # Variance estimate
            return mean + sqrt(2.0 * log(self.t) * variance / self.pulls[arm]) + 3.0 * self.amplitude * log(self.t) / self.pulls[arm]