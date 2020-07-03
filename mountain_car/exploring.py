import gym
import numpy as np

np.random.seed(1)
env = gym.make("MountainCar-v0")

# number of position (state[0]) and velocity (state[1]) buckets in q table
DISCRETE_OS_SIZE = [20, 20]
# size of each bucket
discrete_os_win_size = (env.observation_space.high
                        - env.observation_space.low)/DISCRETE_OS_SIZE

# initialise q table
# size=[20, 20, 3](20 positions x 20 velocities x 3 actions)
q_table = np.random.uniform(low=-2,
                            high=0,
                            size=(DISCRETE_OS_SIZE + [env.action_space.n])
                            )


def get_discrete_state(state):
    """find position on q table of state

    Args:
        state (array of length 2): current state
    """
    discrete_state = (state - env.observation_space.low)/discrete_os_win_size
    return tuple(discrete_state.astype(np.int))


# hyperparemeters
LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 4001
SHOW_EVERY = 1000

# Exploration settings
epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# run training
for episode in range(EPISODES):
    discrete_state = get_discrete_state(env.reset())
    done = False

    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False

    # individual episode
    while not done:
        # pick action:
        if np.random.random() > epsilon:
            # get action from q table
            action = np.argmax(q_table[discrete_state])
        else:
            # get random action
            action = np.random.randint(0, env.action_space.n)

        # take step:
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if render:
            env.render()

        # if this wasn't the last step
        if not done:
            # maximum q value in next step:
            max_future_q = np.max(q_table[new_discrete_state])

            # current q
            current_q = np.max(q_table[discrete_state])

            # new q
            new_q = (1 - LEARNING_RATE) * current_q
            + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            # update table with new q
            q_table[discrete_state + (action,)] = new_q
        # if the simulation ended and car is past flag:
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0.

        # update discrete state for next step
        discrete_state = new_discrete_state

        # epsilon decaying:
        if START_EPSILON_DECAYING <= episode <= END_EPSILON_DECAYING:
            epsilon -= epsilon_decay_value

    env.close()
