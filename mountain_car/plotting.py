import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from matplotlib.animation import FuncAnimation
import os
style.use('ggplot')


def get_q_color(value, vals):
    if value == max(vals):
        return 'green', 1
    else:
        return 'red', 0.5


print("Loading...")
q_tables = os.listdir("./qtables/")
q_tables.sort()
q_tables = [np.load("./qtables/" + table) for table in q_tables]
print("Loaded")

EPISODES = len(q_tables)

fig = plt.figure(figsize=(12, 9))
gs = fig.add_gridspec(3, 3)


axs = [None] * 3
axs[0] = fig.add_subplot(gs[0, :])
axs[1] = fig.add_subplot(gs[1, :])
axs[2] = fig.add_subplot(gs[2, :])

for x, x_vals in enumerate(q_tables[0]):
        for y, y_vals in enumerate(x_vals):
            for i in range(3):
                color, blend = get_q_color(y_vals[i], y_vals)
                axs[i].scatter(x, y,
                               marker="o",
                               c=color,
                               alpha=blend)
                axs[i].set_ylabel(f"Action {i}")

plt.show()

def update(i):
    for x, x_vals in enumerate(q_tables[i]):
        for y, y_vals in enumerate(x_vals):
            for i in range(3):
                color, blend = get_q_color(y_vals[i], y_vals)
                axs[i].set_offsets()

    return axs[0], axs[1], axs[2]


anim = FuncAnimation(fig, update, frames=np.arange(EPISODES), interval=200)
anim.save('qtables.gif', dpi=80, writer='imagemagick')
