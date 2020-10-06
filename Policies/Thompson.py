# -*- coding: utf-8 -*-


try:
    from .BayesianIndexPolicy import BayesianIndexPolicy
except (ImportError, SystemError):
    from BayesianIndexPolicy import BayesianIndexPolicy


class Thompson(BayesianIndexPolicy):
    """
    The Thompson (Bayesian) index policy.
        - By default, it uses a Beta posterior (:class:`Policies.Posterior.Beta`), one by arm.
        - Reference: [Thompson - Biometrika, 1933].
    """

    def __str__(self):
        return "Thompson Sampling"
    
    def computeIndex(self, arm):
        
        """ 
        Compute the current index, at time t and after N_k(t) pulls of arm k, giving S_k(t) rewards of 1, by sampling from the Beta posterior:

        """
        return self.posterior[arm].sample()