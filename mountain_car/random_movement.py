import gym
import numpy as np

np.random.seed(6)

env = gym.make("MountainCar-v0")
state = env.reset()

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


done = False
reward = -1.0
# print(state, reward, done)
for i in range(200):
    action = np.argmax(q_table[get_discrete_state(state)])
    state, reward, done, _ = env.step(action)
    # print(new_state, reward, done)
    env.render()
