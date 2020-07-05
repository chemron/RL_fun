import gym
import numpy as np
from gym.wrappers import Monitor

def record(episode):
    return episode % 500 == 0


env = gym.make("MountainCar-v0")
env = Monitor(env, "recording_no_exploring", video_callable=record, force=True)

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
EPISODES = 25000
SHOW_EVERY = 500

# run training
for episode in range(EPISODES):

    if episode % SHOW_EVERY == 0:
        render = True
        print(episode)
    else:
        render = False

    discrete_state = get_discrete_state(env.reset())
    done = False

    # individual episode
    while not done:
        action = np.argmax(q_table[discrete_state])
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

        discrete_state = new_discrete_state

env.close()
