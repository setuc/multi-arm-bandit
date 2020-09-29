# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import binomial

# Local imports
try:
    from .Arm import Arm
except ImportError:
    from Arm import Arm



class Bernoulli(Arm):
    """ Bernoulli distributed arm."""

    def __init__(self, probability):
        """New arm."""
        assert 0 <= probability <= 1, "Error, the parameter probability for Bernoulli class has to be in [0, 1]."  # DEBUG
        self.probability = probability  #: Parameter p for this Bernoulli arm
        self.mean = probability  #: Mean for this Bernoulli arm

    # --- Random samples

    def draw(self, t=None):
        """ Draw one random sample."""
        return binomial(1, self.probability)
        # return np.asarray(binomial(1, self.probability), dtype=float)

    def draw_nparray(self, shape=(1,)):
        """ Draw a numpy array of random samples, of a certain shape."""
        return np.asarray(binomial(1, self.probability, shape), dtype=float)

    def set_mean_param(self, probability):
        self.probability = self.mean = probability