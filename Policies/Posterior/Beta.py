# -*- coding: utf-8 -*-
""" 
Manipulate posteriors of Bernoulli/Beta experiments.
Rewards not in {0, 1} are handled with a trick, see bernoulliBinarization, with a "random binarization",
- See https://en.wikipedia.org/wiki/Bernoulli_distribution#Related_distributions
- And https://en.wikipedia.org/wiki/Conjugate_prior#Discrete_distributions
.. [Agrawal12] http://jmlr.org/proceedings/papers/v23/agrawal12/agrawal12.pdf
"""

from random import random
try:
    from numpy.random import beta as betavariate  # Faster! Yes!
except ImportError:
    from random import betavariate


# Local imports
try:
    from .Posterior import Posterior

    from .with_proba import with_proba
except:
    from Posterior import Posterior

    from with_proba import with_proba


# --- Utility functions


def bernoulliBinarization(r_t):
    """ 
    Return a (random) binarization of a reward :math:`r_t`, in the continuous interval :math:`[0, 1]` as an observation in discrete :math:`{0, 1}`.
    - Useful to allow to use a Beta posterior for non-Bernoulli experiments,
    - That way, :class:`Thompson` sampling can be used for any continuous-valued bounded rewards.
    """
    if r_t == 0:
        return 0  # Returns a int!
    elif r_t == 1:
        return 1  # Returns a int!
    else:
        assert 0 <= r_t <= 1, "Error: only bounded rewards in [0, 1] are supported by this Beta posterior right now."
        return int(with_proba(r_t))


# --- Class

class Beta(Posterior):
    """ Manipulate posteriors of Bernoulli/Beta experiments."""

    def __init__(self, a=1, b=1):
        r""" Create a Beta posterior :math:`\mathrm{Beta}(\alpha, \beta)` with no observation, i.e., :math:`\alpha = 1` and :math:`\beta = 1` by default."""
        assert a >= 0, "Error: parameter 'a' for Beta posterior has to be >= 0."  # DEBUG
        self._a = a
        assert b >= 0, "Error: parameter 'b' for Beta posterior has to be >= 0."  # DEBUG
        self._b = b
        self.N = [a, b]  #: List of two parameters [a, b]

    def __str__(self):
        return r"Beta(\alpha={:.3g}, \beta={:.3g})".format(self.N[1], self.N[0])

    def reset(self, a=None, b=None):
        """Reset alpha and beta, both to 1 as when creating a new default Beta."""
        if a is None:
            a = self._a
        if b is None:
            b = self._b
        self.N = [a, b]

    def sample(self):
        """Get a random sample from the Beta posterior (using :func:`numpy.random.betavariate`).
        - Used only by :class:`Thompson` Sampling and :class:`AdBandits` so far.
        """
        return betavariate(self.N[1], self.N[0])

    
    def mean(self):
        """Compute the mean of the Beta posterior (should be useless)."""
        return self.N[1] / float(sum(self.N))

    def forget(self, obs):
        """Forget the last observation."""
        # print("Info: calling Beta.forget() with obs = {} ...".format(obs))  # DEBUG
        # FIXED update this code, to accept obs that are FLOAT in [0, 1] and not just in {0, 1}...
        self.N[bernoulliBinarization(obs)] -= 1

    def update(self, obs):
        r"""Add an observation.
        - If obs is 1, update :math:`\alpha` the count of positive observations,
        - If it is 0, update :math:`\beta` the count of negative observations.
        .. note:: Otherwise, a trick with :func:`bernoulliBinarization` has to be used.
        """
        # print("Info: calling Beta.update() with obs = {} ...".format(obs))  # DEBUG
        # FIXED update this code, to accept obs that are FLOAT in [0, 1] and not just in {0, 1}...
        self.N[bernoulliBinarization(obs)] += 1