from .BasePolicy import BasePolicy
from .Posterior import Beta

# --- Naive or less naive epsilon-greedy policies
from .EpsilonGreedy import EpsilonGreedy
# --- Mine, simple exploratory policies
# from .EmpiricalMeans import EmpiricalMeans

# --- Variants on EpsilonFirst, Explore-Then-Commit from E.Kaufmann's slides at IEEE ICC 2017
from .ExploreThenCommit import ETC_KnownGap, ETC_RandomStop

# --- Simple UCB policies
from .UCB import UCB
from .UCBalpha import UCBalpha  # Different indexes
from .UCBplus import UCBplus    # Different indexes

# --- UCB policies with variance terms
from .UCBV import UCBV          # Different indexes

# --- Thompson sampling index policy
from .Thompson import Thompson

from .with_proba import with_proba
