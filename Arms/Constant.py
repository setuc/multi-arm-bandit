# -*- coding: utf-8 -*-

import numpy as np

# Local imports
try:
    from .Arm import Arm
except ImportError:
    from Arm import Arm


class Constant(Arm):
    """ 
    Arm with a constant reward.
    """

    def __init__(self, constant_reward=0.5, lower=0., amplitude=1.):
        constant_reward = float(constant_reward)

        # Constant value of rewards
        self.constant_reward = constant_reward  
         
        # Known lower value of rewards
        lower = min(lower, np.floor(constant_reward))
        self.lower = lower 
        
        # Known amplitude of rewards
        self.amplitude = amplitude  #: Known amplitude of rewards

        # Mean for the constant arm
        self.mean = constant_reward

    
    def draw(self, t=None):
        """ 
        Draw one constant sample.
        """
        return self.constant_reward