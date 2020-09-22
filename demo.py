
from Policies.UCB import UCB
from Arms import Gaussian
from Environment.MAB import MAB
from Environment.Results import Result

policy = UCB(nbArms = 3)

armConfiguration = [
            Gaussian(0.1),
            Gaussian(0.5),
            Gaussian(0.9),
        ]


env = MAB(armConfiguration)

horizon = 1000
results = Result(env.nbArms, horizon)

for t in range(horizon):
        choice = policy.choice()

        # 2. A random reward is drawn, from this arm at this time
        reward = env.draw(choice, t)

        # 3. The policy sees the reward
        policy.getReward(choice, reward)

        # 4. Finally we store the results
        results.store(t, choice, reward)


print(results.choices)
print(results.rewards)