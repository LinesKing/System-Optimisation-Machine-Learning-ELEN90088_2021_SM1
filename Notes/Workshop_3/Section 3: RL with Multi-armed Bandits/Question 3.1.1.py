import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

univ_data = pd.read_csv('univ-latencies.csv')
univ_data = -univ_data.sample(n=20, axis=1) #choose 20 columns randomly for computational simplicity
univ_data = univ_data.to_numpy()

## Problem 3.1.1

def bandit_epsgreedy(data=univ_data, eps=0.1):
    # epsilon-greedy bandit algorithm

    # parameters
    num_bandits = data.shape[1]
    T = data.shape[0]

    # init storage arrays
    Q = np.zeros(num_bandits)
    N = np.zeros(num_bandits)
    selections = np.zeros(T)  # sequence of lever selections
    step_rewards = np.zeros(T)  # sequence of step selections
    cum_rewards = np.zeros(T)  # sequence of cumulative rewards
    # main loop
    for t in range(T):

        # pull lever
        if np.random.rand() < eps:
            # make a random selection
            sel = random.randrange(num_bandits)
        else:
            # choose the best expected reward
            sel = np.argmax(Q)

        # update nbr of selections made
        N[sel] = N[sel] + 1
        # update mean reward estimate
        Q[sel] = Q[sel] + (1 / N[sel]) * (data[t, sel] - Q[sel])

        # store values
        selections[t] = sel
        step_rewards[t] = data[t, sel]
        if t > 0:
            cum_rewards[t] = step_rewards[t] + cum_rewards[t - 1]
        else:
            cum_rewards[t] = step_rewards[t]

    total_reward = cum_rewards[-1]  # the last one is total reward!

    return (selections, step_rewards, cum_rewards, total_reward)


(selections, step_rewards, cum_rewards, total_reward) = bandit_epsgreedy(eps=0.15)

print(total_reward)
plt.figure()
plt.title('Cumulative Reward vs Time')
plt.xlabel('Time')
plt.ylabel('Cumulative Reward')
plt.plot(cum_rewards)
plt.show()
