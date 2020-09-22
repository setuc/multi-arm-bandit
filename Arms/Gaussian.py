# -*- coding: utf-8 -*-

from random import gauss
try:
    from .Arm import Arm
except ImportError:
    from Arm import Arm

# Default value for the variance of a [0, 1] Gaussian arm
VARIANCE = 0.05

class Gaussian(Arm):
    """ 
    Gaussian distributed arm, possibly truncated.
        - Default is to truncate into [0, 1] 
    """

    def __init__(self, mu, sigma=VARIANCE, mini=0, maxi=1):
        # Mean of Gaussian arm
        self.mu = self.mean = mu  

        # Variance of Gaussian arm
        self.sigma = sigma  
        
        # Lower value of rewards
        self.min = mini  
        
        #: Higher value of rewards
        self.max = maxi  

    # --- Random samples

    def draw(self, t=None):
        """ Draw one random sample. The parameter t is ignored in this Arm."""
        return min(max(gauss(self.mu, self.sigma), self.min), self.max)    