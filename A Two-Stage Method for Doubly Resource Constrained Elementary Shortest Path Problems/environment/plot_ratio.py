#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time : 2022/6/23 14:38
# @Author : Jin Echo
# @File : plot_ratio.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
# https://stackoverflow.com/questions/16006572/plotting-different-colors-in-matplotlib/16006929#16006929


num_nodes = np.arange(100, 500)
ratio_edges_batch = [0.3, 0.4, 0.5, 0.6]

cmap = plt.get_cmap('jet')
colors = [cmap(i) for i in np.linspace(0, 1, len(ratio_edges_batch))]
print(colors)
for color, ratio_edges in zip(colors, ratio_edges_batch):
    num_edges = (num_nodes ** ratio_edges).astype(int)
    # num_edges = np.round(num_nodes ** ratio_edges)
    ratio_estimated = 1.0 * (num_nodes * num_edges) / (num_nodes - num_edges)
    ratio_lower, ratio_upper = ratio_estimated / 3, ratio_estimated
    plt.plot(num_nodes, ratio_lower, color=color, label='ratio_lower  ratio_edges={:.2f}'.format(ratio_edges))
    plt.plot(num_nodes, ratio_upper, color=color, label='ratio_upper  ratio_edges={:.2f}'.format(ratio_edges))
# axes = plt.axes()
# axes.set_ylim(
#     [np.min([res_nature, res_double, res_dueling]) - 5, np.max([res_nature, res_double, res_dueling]) + 5])
plt.xlabel('num_nodes')
plt.ylabel('ratio')
plt.legend(loc='upper left')
plt.grid()
# plt.savefig(save_dir+'fig.png')
plt.show()
