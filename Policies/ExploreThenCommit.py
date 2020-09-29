import numpy as np
from scipy.special import lambertw

# Local import
try:
    from .EpsilonGreedy import EpsilonGreedy
    from .BasePolicy import BasePolicy
    from .with_proba import with_proba
except ImportError:
    from EpsilonGreedy import EpsilonGreedy
    from BasePolicy import BasePolicy
    from with_proba import with_proba


class ETC_KnownGap(EpsilonGreedy):
    r""" 
    Variant of the Explore-Then-Commit policy, with known horizon 
    - Reference: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    - Reference: Eyal Even-Dar, Shie Mannor, and Yishay Mansour. Action Elimination and Stopping Conditions for the Multi-Armed Bandit and Reinforcement Learning Problems. Journal of Machine Learning Research, 7:1079–1105, 2006
                    https://jmlr.csail.mit.edu/papers/volume7/evendar06a/evendar06a.pdf
    """

    def __init__(self, nbArms, horizon=None, gap=GAP, lower=0., amplitude=1.):
        super(ETC_KnownGap, self).__init__(nbArms, epsilon=0.5, lower=lower, amplitude=amplitude)
        # Arguments

        #: Parameter :math:`T` = known horizon of the experiment.
        self.horizon = int(horizon)  

        #: Known gap parameter for the stopping rule. Between 0 and 1
        self.gap = gap 

        # Compute the time m
        m = max(0, int(np.floor(((4. / gap**2) * np.log(horizon * gap**2 / 4.)))))

        #: Time until pure exploitation, ``m_`` steps in each arm.
        self.max_t = self.nbArms * m  

    def __str__(self):
        return r"ETC_KnownGap($T={}$, $\Delta={:.3g}$, $T_0={}$)".format(self.horizon, self.gap, self.max_t)

    @property
    def epsilon(self):
        r""" 
        1 while :math:`t \leq T_0`, 0 after, where :math:`T_0` is defined by:
        .. math:: T_0 = \lfloor \frac{4}{\Delta^2} \log(\frac{T \Delta^2}{4}) \rfloor.
        """
        if self.t <= self.max_t:
            # First phase: randomly explore!
            return 1
        else:
            # Second phase: just exploit!
            return 0

ALPHA = 4
class ETC_RandomStop(EpsilonGreedy):
    r""" 
    Variant of the Explore-Then-Commit policy, with known horizon :math:`T` and random stopping time. Uniform exploration until the stopping time.
    - Reference: https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies
    - Reference: Eyal Even-Dar, Shie Mannor, and Yishay Mansour. Action Elimination and Stopping Conditions for the Multi-Armed Bandit and Reinforcement Learning Problems. Journal of Machine Learning Research, 7:1079–1105, 2006
                    https://jmlr.csail.mit.edu/papers/volume7/evendar06a/evendar06a.pdf
    
    A variation was also available in the recent paper, refer Algorithm 1 for best arm selection
    - Reference: A.Garivier et al. On Explore-Then-Commit Strategies, NIPS, 2016, https://arxiv.org/pdf/1605.08988.pdf                    
    """

    def __init__(self, nbArms, horizon=None, alpha=ALPHA,
                 lower=0., amplitude=1.):
        super(ETC_RandomStop, self).__init__(nbArms, epsilon=0.5, lower=lower, amplitude=amplitude)

        # Arguments

        #: Parameter :math:`T` = known horizon of the experiment.
        self.horizon = int(horizon)  

        #: Parameter :math:`\alpha` in the formula (4 by default).
        self.alpha = alpha  

        #: Still randomly exploring?
        self.stillRandom = True  

    def __str__(self):
        return r"ETC_RandomStop($T={}$)".format(self.horizon)

    @property
    def epsilon(self):
        if np.min(self.pulls) > 0:
            means = self.rewards / self.pulls
            
            largestDiffMean = max([abs(mi - mj) for mi in means for mj in means if mi != mj])

            if largestDiffMean > np.sqrt((self.alpha * np.log(self.horizon / self.t)) / self.t):
                self.stillRandom = False
        # Done
        if self.stillRandom:
            # First phase: randomly explore!
            return 1
        else:
            # Second phase: just exploit!
            return 0 
