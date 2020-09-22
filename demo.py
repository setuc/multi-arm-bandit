
from Policies.UCB import UCB
from Policies.EpsilonGreedy import EpsilonGreedy
from Arms import Gaussian
from Environment.MAB import MAB
from Environment.Results import Result
from tqdm import tqdm 
from collections import Counter

policy = EpsilonGreedy(nbArms=3, epsilon=0.1)

armConfiguration = [
            Gaussian(0.1),
            Gaussian(0.5),
            Gaussian(0.9),
        ]


env = MAB(armConfiguration)

horizon = 1000
results = Result(env.nbArms, horizon)

prettyRange = tqdm(range(horizon), desc="Time t")
for t in prettyRange:
        # 1. The player's policy choose an arm
        choice = policy.choice()

        # 2. A random reward is drawn, from this arm at this time
        reward = env.draw(choice, t)

        # 3. The policy sees the reward
        policy.getReward(choice, reward)

        # 4. Finally we store the results
        results.store(t, choice, reward)

print(Counter(results.choices))
# print(results.choices)
# print(results.rewards)