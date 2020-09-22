# -*- coding: utf-8 -*-
""" Result.Result class to wrap the simulation results."""
import numpy as np


class Result(object):
    """
    Result accumulators.
    """

    def __init__(self, nbArms, horizon):
        """ 
        Create Result Array
        """
        # Store all the choices.
        self.choices = np.zeros(horizon, dtype=int)  
        
        # Store all the rewards, to compute the mean.
        self.rewards = np.zeros(horizon)  
        
        # Store the pulls.
        self.pulls = np.zeros(nbArms, dtype=int)  


    def store(self, time, choice, reward):
        """ 
        Store results.
        """
        self.choices[time] = choice
        self.rewards[time] = reward
        self.pulls[choice] += 1
