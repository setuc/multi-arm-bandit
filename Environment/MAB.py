# -*- coding: utf-8 -*-
""" 
:class:`MAB` class to wrap the arms of some Multi-Armed Bandit problem.
Such class has to have *at least* these methods:
- ``draw(armId, t)`` to draw *one* sample from that ``armId`` at time ``t``,
"""


import numpy as np
import matplotlib.pyplot as plt


class MAB(object):
    """ Basic Multi-Armed Bandit problem, for stochastic and i.i.d. arms.
    - configuration can be a dict with 'arm_type' and 'params' keys. 'arm_type' is a class from the Arms module, and 'params' is a dict, used as a list/tuple/iterable of named parameters given to 'arm_type'. Example::
        configuration = {
            'arm_type': Bernoulli,
            'params':   [0.1, 0.5, 0.9]
        }
        configuration = {  # for fixed variance Gaussian
            'arm_type': Gaussian,
            'params':   [0.1, 0.5, 0.9]
        }
    - But it can also accept a list of already created arms::
        configuration = [
            Bernoulli(0.1),
            Bernoulli(0.5),
            Bernoulli(0.9),
        ]
    - Both will create three Bernoulli arms, of parameters (means) 0.1, 0.5 and 0.9.
    """

    def __init__(self, configuration):
        """New MAB."""
        print("\n\nCreating a new MAB problem ...")  
        self.arms = []  #: List of arms

        if isinstance(configuration, dict):
            print("  Reading arms of this MAB problem from a dictionnary 'configuration' = {} ...".format(configuration))  
            arm_type = configuration["arm_type"]
            print(" - with 'arm_type' =", arm_type)  
            params = configuration["params"]
            print(" - with 'params' =", params)  
            # Each 'param' could be one value (eg. 'mean' = probability for a Bernoulli) or a tuple (eg. '(mu, sigma)' for a Gaussian) or a dictionnary
            for param in params:
                self.arms.append(arm_type(*param) if isinstance(param, (dict, tuple, list)) else arm_type(param))
            # XXX try to read sparsity
            self._sparsity = configuration["sparsity"] if "sparsity" in configuration else None
        else:
            print("  Taking arms of this MAB problem from a list of arms 'configuration' = {} ...".format(configuration))  
            for arm in configuration:
                self.arms.append(arm)

        # Compute the means and stats
        print(" - with 'arms' =", self.arms)  
        
        # Means of arms
        self.means = np.array([arm.mean for arm in self.arms])  
        print(" - with 'means' =", self.means)  
        
        # Number of arms
        self.nbArms = len(self.arms)  
        print(" - with 'nbArms' =", self.nbArms)  
        
        # Max mean of arms
        self.maxArm = np.max(self.means) 
        print(" - with 'maxArm' =", self.maxArm)  
        
        # Min mean of arms
        self.minArm = np.min(self.means)  
        print(" - with 'minArm' =", self.minArm)  

    def __repr__(self):
        return "{}(nbArms: {}, arms: {}, minArm: {:.3g}, maxArm: {:.3g})".format(self.__class__.__name__, self.nbArms, self.arms, self.minArm, self.maxArm)

    # --- Draw samples

    def draw(self, armId, t=1):
        """ Return a random sample from the armId-th arm, at time t. Usually t is not used."""
        return self.arms[armId].draw(t)

    #
    # --- Helper to compute vector of min arms, max arms, all arms

    def get_minArm(self, horizon=None):
        """Return the vector of min mean of the arms.
        - It is a vector of length horizon.
        """
        return np.full(horizon, self.minArm)
        # return self.minArm  # XXX Nope, it's not a constant!

    def get_maxArm(self, horizon=None):
        """Return the vector of max mean of the arms.
        - It is a vector of length horizon.
        """
        return np.full(horizon, self.maxArm)
        # return self.maxArm  # XXX Nope, it's not a constant!

    def get_maxArms(self, M=1, horizon=None):
        """Return the vector of sum of the M-best means of the arms.
        - It is a vector of length horizon.
        """
        return np.full(horizon, self.sumBestMeans(M))

    def get_allMeans(self, horizon=None):
        """Return the vector of means of the arms.
        - It is a numpy array of shape (nbArms, horizon).
        """
        # allMeans = np.tile(self.means, (horizon, 1)).T
        allMeans = np.zeros((self.nbArms, horizon))
        for t in range(horizon):
            allMeans[:, t] = self.means
        return allMeans

