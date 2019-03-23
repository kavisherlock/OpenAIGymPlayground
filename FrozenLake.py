import numpy as np
import gym
import random
import matplotlib.pyplot as plt

from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78,  # optimum = .8196
)

# Initialisations

env = gym.make("FrozenLake-v0")
action_space_size = env.action_space.n
state_space_size = env.observation_space.n

total_episodes = 10000        # Total episodes
learning_rate = 0.8           # Learning rate
max_steps = 99                # Max steps per episode
gamma = 0.95                  # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.005            # Exponential decay rate for exploration prob

rewards = []

# The Q Learning algorithm

# Frozen Lake is a slippery place. Sometimes training doesn't help. Try again if so
for i in range(100):
    if sum(rewards) > 0:
        break

    # 1. Initialise Q
    qtable = np.zeros((state_space_size, action_space_size))

    rewards = []
    # 2. Repeat steps 3-5 until learning is stopped
    for episode in range(total_episodes):
        state = env.reset()
        step = 0
        done = False
        total_rewards = 0

        for step in range(max_steps):
            # 3. Choose an action in the current environment, s, based on the current Q-value estimates
            if random.uniform(0, 1) > epsilon:
                action = np.argmax(qtable[state, :])  # exploit
            else:
                action = env.action_space.sample()  # explore

            # 4. Take the action and get the next state and reward
            new_state, reward, done, info = env.step(action)

            # 5. Update Q(s, a) = Q(s, a) + alpha(r + gamma * max_a'(Qs', a') - Q(s,a))
            qtable[state, action] += learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])

            total_rewards += reward
            state = new_state

            if done:
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        rewards.append(total_rewards)

print("Score over time: " + str(sum(rewards)/total_episodes))
print(qtable)

# plt.plot(np.cumsum(rewards))
# plt.xlabel('Episodes')
# plt.ylabel('Rewards')
# plt.show()

# q table after a million episodes. Average score: 0.7468 onn 10000 episodes
million_episode_qtable = np.array([
    [2.98396023e-01, 7.66571318e-02, 7.80825998e-02, 8.88989071e-02],
    [6.69061330e-03, 6.89843970e-03, 1.50016811e-03, 1.36866338e-01],
    [2.73346157e-03, 3.64319240e-02, 3.13747535e-02, 7.42601242e-02],
    [7.66059560e-04, 3.17757135e-03, 1.92627962e-04, 4.78098119e-02],
    [2.26569885e-01, 4.10442846e-02, 5.36603891e-02, 1.08220092e-01],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
    [4.33216062e-06, 2.49407605e-06, 4.31648524e-02, 1.75580791e-09],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
    [7.55799093e-02, 3.39922136e-06, 7.32687091e-02, 5.58709879e-01],
    [2.07822697e-02, 6.54105951e-01, 1.19559484e-01, 6.43942936e-04],
    [8.88152297e-01, 1.14299643e-03, 5.70222370e-05, 9.96250471e-03],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
    [4.97762588e-03, 2.81570006e-02, 8.25623218e-01, 9.33422472e-02],
    [1.46635179e-01, 9.64308467e-01, 3.25015293e-01, 4.27730183e-01],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])

env.reset()

steps = []
total_rewards = 0
test_episodes = 10000

for episode in range(test_episodes):
    state = env.reset()
    step = 0
    done = False

    for step in range(max_steps):
        action = np.argmax(million_episode_qtable[state, :])
        new_state, reward, done, info = env.step(action)

        if done:
            total_rewards += reward
            steps.append(step)
            break
        state = new_state

env.close()

plt.plot(steps)
plt.ylabel('Steps')
# plt.show()

print("Test score over time: " + str(total_rewards/test_episodes))
