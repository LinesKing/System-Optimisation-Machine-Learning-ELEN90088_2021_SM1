import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt

univ_data = pd.read_csv('univ-latencies.csv')
univ_data = -univ_data.sample(n=20, axis=1)  # choose 20 columns randomly for computational simplicity
univ_data = univ_data.to_numpy()

## Problem 3.1.2
# parameters
num_bandits = univ_data.shape[1]
T = univ_data.shape[0]
c = math.sqrt(0.15)  # the degree of exploration

# Implementing UCB
selections = np.zeros(T)  # sequence of lever selections
step_rewards = np.zeros(T)  # sequence of step selections
cum_rewards = np.zeros(T)  # sequence of cumulative rewards
bandits_selected = np.zeros(num_bandits)  # each bandit selected times
bandits_rewards = np.zeros(num_bandits)  # each bandit reward
total_reward = 0  # the total reward

for t in range(0, T):
    sel = 0
    max_upper_bound = -1e400
    for i in range(0, num_bandits):
        if bandits_selected[i] > 0:
            Q = bandits_rewards[i] / bandits_selected[i]  # exploitation
            delta = c * math.sqrt(math.log(t + 1) / bandits_selected[i])  # exploration
            upper_bound = Q + delta
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            sel = i
    selections[t] = sel
    bandits_selected[sel] = bandits_selected[sel] + 1  # update each bandit selected times
    step_rewards[t] = univ_data[t, sel]  # step reward at time t
    cum_rewards[t] = cum_rewards[t - 1] + step_rewards[t]  # update sequence of cumulative rewards
    bandits_rewards[sel] = bandits_rewards[sel] + step_rewards[t]  # update each bandit reward

total_reward = cum_rewards[-1]  # the last one is total reward!

print(total_reward)
plt.figure()
plt.title('Cumulative Reward vs Time')
plt.xlabel('Time')
plt.ylabel('Cumulative Reward')
plt.plot(cum_rewards)
plt.show()
