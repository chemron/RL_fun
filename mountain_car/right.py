import gym

# initialise mountaincar environment
env = gym.make("MountainCar-v0")

# number of possible actions 0 = left, 1 = stay still, 2 = right
print(env.action_space.n)

env.reset()

# render
for i in range(200):
    action = 2  # go right
    env.step(action)
    env.render()
