#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time : 2022/4/22 16:59
# @Author : Jin Echo
# @File : remove_alley_nodes.py
import copy
import numpy as np


def remove_alley_nodes(adj, index_start, index_end):
    """
    Reductions by elementarity.

    :param adj: ndarray (max_num_nodes, K*max_num_nodes) inf
    :param index_start:
    :param index_end:
    :return:
    """
    # assert adj.ndim == 2, 'dim error'
    n = adj.shape[0]
    K = adj.shape[1] // n
    adj_ = copy.copy(adj)
    adj_1 = np.copy(adj[:, :n])
    mask_v = np.zeros(n, np.bool)

    while 1:
        adj_k = adj_1 != np.inf
        mask_cur = np.zeros(n, np.bool)  # in this iteration
        # 1. Remove nodes with no outgoing or incoming arcs.
        mask_1 = (np.add.reduce(adj_k, 0) == 0) | (np.add.reduce(adj_k, 1) == 0)
        mask_1[[index_start, index_end]] = False
        if np.add.reduce(mask_1) != 0:
            mask_cur[mask_1] = True

        # 2. Remove the node that has only one outgoing arc and one incoming arc and the tail of the incoming arc and the head of the outgoing arc are the same node.
        mask_2 = (np.add.reduce(adj_k, 0) == 1) & (np.add.reduce(adj_k, 1) == 1)  # 出度、入度均为1的节点
        mask_2[[index_start, index_end]] = False
        mask_2 = np.where(mask_2)[0]
        if len(mask_2):
            rows = adj_k[mask_2, :]
            columns = adj_k[:, mask_2]
            mask_2 = mask_2[np.add.reduce(rows & (columns.T), axis=1) == 1]
        mask_cur[mask_2] = True

        mask_cur = np.where(mask_cur)[0]
        mask_new = (~mask_v)[mask_cur]
        mask_new = mask_cur[mask_new]
        if len(mask_new) == 0:
            break
        else:
            adj_1[mask_new, :] = np.inf
            adj_1[:, mask_new] = np.inf
            mask_v[mask_cur] = True

    mask_v = np.where(mask_v)[0]
    adj_[mask_v, :] = np.inf
    for k in range(K):
        adj_[:, k * n + mask_v] = np.inf

    return adj_, mask_v


if __name__ == '__main__':
    import networkx as nx
    import random
    import matplotlib.pyplot as plt

    np.random.seed(2)
    random.seed(1)

    def TEST1():
        max_num_nodes, num_nodes, num_edges = 20, 20, 4
        index_end = num_nodes - 1

        G = nx.barabasi_albert_graph(num_nodes, num_edges)  # 生成图结构，生成的为无向图
        edges_idx = np.array(G.edges)
        num_edges = (num_nodes - num_edges) * num_edges  # 图中的边数，也等于edges_idx.shape[0]
        assert num_edges == edges_idx.shape[0]

        # 代价
        adj_cost = np.ones([max_num_nodes] * 2) * np.inf
        adj_cost[edges_idx[:, 0], edges_idx[:, 1]] = np.random.rand(num_edges)
        adj_cost[edges_idx[:, 1], edges_idx[:, 0]] = adj_cost[edges_idx[:, 0], edges_idx[:, 1]]
        adj_cost = np.expand_dims(adj_cost, axis=0)
        # 时延
        adj_delay = np.ones([max_num_nodes] * 2) * np.inf
        adj_delay[edges_idx[:, 0], edges_idx[:, 1]] = np.random.rand(num_edges)
        adj_delay[edges_idx[:, 1], edges_idx[:, 0]] = adj_delay[edges_idx[:, 0], edges_idx[:, 1]]
        adj_delay = np.expand_dims(adj_delay, axis=0)

        adj = np.concatenate((adj_cost, adj_delay), axis=-1)
        adj = adj.squeeze(0)

        adj_, mask_v = remove_alley_nodes(adj, 0, index_end)
        print(adj_)
        print(mask_v)

        plt.figure(1)
        nx.draw(G, with_labels=True, font_weight='bold')
        plt.show()
        print(1)

    def TEST2():
        # 一次 无出口或无入口
        adj = [[np.inf, 1, np.inf, np.inf],
               [np.inf, np.inf, np.inf, 1],
               [np.inf, 1, np.inf, np.inf],
               [np.inf, np.inf, np.inf, np.inf]]
        adj = np.array(adj)
        adj_, mask_v = remove_alley_nodes(adj, 0, 3)
        print(adj)
        print(adj_, mask_v)

    def TEST3():
        # 一次 仅有一个出口和一个入口(出度、入度均为1),且出口和入口为同一个节点
        adj = [[np.inf, 1, np.inf, np.inf],
               [np.inf, np.inf, 1, 1],
               [np.inf, 1, np.inf, np.inf],
               [np.inf, np.inf, np.inf, np.inf]]
        adj = np.array(adj)
        adj_, mask_v = remove_alley_nodes(adj, 0, 3)
        print(adj)
        print(adj_, mask_v)  # 2

    def TEST4():
        # 2次 仅有一个出口和一个入口(出度、入度均为1),且出口和入口为同一个节点
        adj = [[np.inf, 1, np.inf, np.inf, np.inf],
               [np.inf, np.inf, 1, np.inf, 1],
               [np.inf, 1, np.inf, 1, np.inf],
               [np.inf, np.inf, 1, np.inf, np.inf],
               [np.inf, np.inf, np.inf, np.inf, np.inf]]
        adj = np.array(adj)
        adj_, mask_v = remove_alley_nodes(adj, 0, 4)
        print(adj)
        print(adj_, mask_v)  # 2 3

    def TEST5():
        # 2次 无出口或无入口
        adj = [[np.inf, 1, np.inf, np.inf, np.inf],
               [np.inf, np.inf, 1, np.inf, 1],
               [np.inf, np.inf, np.inf, np.inf, np.inf],
               [np.inf, np.inf, 1, np.inf, np.inf],
               [np.inf, np.inf, np.inf, np.inf, np.inf]]
        adj = np.array(adj)
        adj_, mask_v = remove_alley_nodes(adj, 0, 4)
        print(adj)
        print(adj_, mask_v)  # 2 3

    def TEST6():
        #  无出口或无入口 + 仅有一个出口和一个入口(出度、入度均为1),且出口和入口为同一个节点
        adj = [[np.inf, 1, np.inf, np.inf, np.inf],
               [np.inf, np.inf, 1, np.inf, 1],
               [np.inf, np.inf, np.inf, 1, np.inf],
               [np.inf, np.inf, 1, np.inf, np.inf],
               [np.inf, np.inf, np.inf, np.inf, np.inf]]
        adj = np.array(adj)
        adj_, mask_v = remove_alley_nodes(adj, 0, 4)
        print(adj)
        print(adj_, mask_v)  # 2 3

    TEST6()