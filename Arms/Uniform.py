# -*- coding: utf-8 -*-

from random import random

# Local imports
try:
    from .Arm import Arm
except ImportError:
    from Arm import Arm


class Uniform(Arm):
    """ Uniformly distributed arm, default in [0, 1],
    - default to (mini, maxi),
    - or [lower, lower + amplitude], if (lower=lower, amplitude=amplitude) is given.
    """

    def __init__(self, mini=0., maxi=1., mean=None, lower=0., amplitude=1.):
        mini = max(mini, lower)
        maxi = min(maxi, lower + amplitude)

        # Lower value of rewards
        self.lower = mini  
        assert maxi <= lower + amplitude, "Error: 'maxi' = {} argument for UniformArm has to be >= 'lower + amplitude' = {}...".format(maxi, lower + amplitude)

        # Amplitude of rewards
        self.amplitude = maxi - mini  

        # self.mean = (mini + maxi) / 2.0
        self.mean = self.lower + (self.amplitude / 2.0)  #: Mean for this UniformArm arm


    def draw(self, t=None):
        """ Draw one random sample. The parameter t is ignored in this Arm."""
        return self.lower + (random() * self.amplitude)