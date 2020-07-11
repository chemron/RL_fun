import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from matplotlib.animation import FuncAnimation
import os
style.use('ggplot')


def is_max(array):
    return array == max(array)


# get q tables
# shape = # tables x velocities x positions x actions
q_tables = os.listdir("./qtables/")
q_tables.sort()
q_tables = np.array([np.load("./qtables/" + table) for table in q_tables])

# hyperparemeters
EPISODES, HEIGHT, WIDTH, ACTIONS = q_tables.shape

# true if maximum quality action, false otherwise
c_map = np.apply_along_axis(is_max, -1, q_tables)


# set up figure and axes
fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(3, 3)

axs = [None] * 3
scat = [None] * 3
axs[0] = fig.add_subplot(gs[0, :])
axs[1] = fig.add_subplot(gs[1, :])
axs[2] = fig.add_subplot(gs[2, :])

# x's
x = np.ravel([np.full(HEIGHT, i) for i in range(WIDTH)])
# y's
y = np.ravel([np.arange(HEIGHT) for _ in range(WIDTH)])


def update_plot(i, data, scat):
    for j in range(3):
        scat[j].set_array(data[j][i])
    return scat[0], scat[1], scat[2]


def get_color(episode, x, y, action):
    if c_map[episode, x, y, action]:
        return 0.2
    else:
        return 0.7


def get_color_list(episode, action):
    return [get_color(episode, x[i], y[i], action)
            for i in range(WIDTH*HEIGHT)]


color_data = np.array([[get_color_list(episode, action)
                        for episode in range(EPISODES)]
                       for action in range(3)])

scats = [axs[i].scatter(x, y, c=color_data[i][0])
         for i in range(3)]

ani = FuncAnimation(fig, update_plot, frames=np.arange(EPISODES),
                    fargs=(color_data, scats))

plt.show()
