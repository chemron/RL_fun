import gym
import numpy as np
from gym.wrappers import Monitor
# import matplotlib.pyplot as plt

# hyperparemeters
LEARNING_RATE = 0.1

DISCOUNT = 0.95
EPISODES = 4000
SHOW_EVERY = 100

# stats
ep_rewards = []
aggr_ep_rewards = {"ep": [], "avg": [], "max": [], "min": []}
STATS_EVERY = 100


def record(episode):
    return episode % SHOW_EVERY == 0


env = gym.make("MountainCar-v0")
env = Monitor(env, "recording_exploring", video_callable=record, force=True)


DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high
                        - env.observation_space.low) / DISCRETE_OS_SIZE
# Exploration settings
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

q_table = np.random.uniform(low=-2,
                            high=0,
                            size=(DISCRETE_OS_SIZE + [env.action_space.n]))


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


for episode in range(EPISODES):
    # reset
    discrete_state = get_discrete_state(env.reset())
    done = False
    episode_reward = 0

    if episode % SHOW_EVERY == 0:
        render = True
    else:
        render = False

    while not done:

        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(q_table[discrete_state])
        else:
            # Get random action
            action = np.random.randint(0, env.action_space.n)

        new_state, reward, done, _ = env.step(action)
        episode_reward += reward

        new_discrete_state = get_discrete_state(new_state)

        if episode % SHOW_EVERY == 0:
            env.render()
        # new_q = (1 - LEARNING_RATE) * current_q +
        # LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        # If simulation did not end yet after last step - update Q table
        if not done:

            # Maximum possible Q value in next step (for new state)
            max_future_q = np.max(q_table[new_discrete_state])

            # Current Q value (for current state and performed action)
            current_q = q_table[discrete_state + (action,)]

            # Equation for a new Q value for current state and action
            new_q = (1 - LEARNING_RATE) * current_q\
                + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # Update Q table with new Q value
            q_table[discrete_state + (action,)] = new_q

        # Simulation ended (for any reson)
        # if goal position is achived - update Q value with reward directly
        elif new_state[0] >= env.goal_position:
            # q_table[discrete_state + (action,)] = reward
            q_table[discrete_state + (action,)] = 0

        discrete_state = new_discrete_state

    # Decaying is being done every episode if episode number is
    # within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episode_reward)
    if episode % STATS_EVERY == 0:
        reward_batch = ep_rewards[-STATS_EVERY:]
        average_reward = sum(reward_batch)/STATS_EVERY
        aggr_ep_rewards["ep"].append(episode)
        aggr_ep_rewards["avg"].append(average_reward)
        aggr_ep_rewards["max"].append(max(reward_batch))
        aggr_ep_rewards["min"].append(min(reward_batch))
        print(f'Episode: {episode:>5d}, Average reward: {average_reward:>6.1f}'
              f', Current epsilon: {epsilon:>1.2f}')

# plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label="average rewards")
# plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label="max rewards")
# plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label="min rewards")
# plt.legend(loc=4)
# plt.show()

env.close()
