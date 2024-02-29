#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time : 2022/10/3 0:59
# @Author : Jin Echo
# @File : toposort.py
import numpy as np
from collections import deque

def is_acyclic(adj, neighbors_list):
    """
    :param adj: (n,n)
    :param neighbors_list:
    :return:  True - acyclic graph

    """
    n = len(adj)
    # neighbors_list = {}
    # for i in range(n):
    #     neighbors_list[i] = np.where(adj[i] != np.inf)[0]
    # Calculate the initial in-degree for each node.
    indegree = np.sum(adj != np.inf, axis=0)
    res = []

    q = deque([])

    nodes = np.where(indegree == 0)[0]
    cnt = len(nodes)  # Number of nodes queued
    for item in nodes:
        q.append(item)
    while len(q):
        node_cur = q.popleft()
        res.append(node_cur)
        for neighbor in neighbors_list[node_cur]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                q.append(neighbor)
                cnt += 1

    return cnt == n, res  # res is meaningless if there is a loop.


def cal_longest_path(adj, topolist, neighbors_list, index_start):
    distance = -np.ones((len(adj),)) * np.inf
    distance[index_start] = 0
    for node_cur in topolist:
        for neighbor in neighbors_list[node_cur]:
            dis = distance[node_cur] + adj[node_cur, neighbor]
            if dis > distance[neighbor]:
                distance[neighbor] = dis
    return distance


def toposort(adj, index_start):
    """
    :param adj: (n,n)
    """
    # Calculate the list of neighbours for each node.
    n = len(adj)
    neighbors_list = {}
    for i in range(n):
        neighbors_list[i] = np.where(adj[i] != np.inf)[0]
    # Calculate the initial in-degree for each node.
    indegree = np.sum(adj != np.inf, axis=0)
    res = []

    q = deque([])


    nodes = np.where(indegree == 0)[0]
    cnt = len(nodes)  # Number of nodes queued
    for item in nodes:
        q.append(item)
    while len(q):
        node_cur = q.popleft()
        res.append(node_cur)
        for neighbor in neighbors_list[node_cur]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                q.append(neighbor)
                cnt += 1

    distance = -np.ones((n,)) * np.inf
    distance[index_start] = 0

    if cnt == n:  # acyclic
        for node_cur in res:
            for neighbor in neighbors_list[node_cur]:
                dis = distance[node_cur] + adj[node_cur, neighbor]
                if dis > distance[neighbor]:
                    distance[neighbor] = dis

        return True, distance
    else:
        return False, distance


if __name__ == '__main__':
    from algorithm.q_paths import q_paths
    from environment.graph import generate_a_barabasi_albert_graph_with_k_features
    import random
    import networkx as nx
    import time
    import matplotlib.pyplot as plt
    np.random.seed(2)
    random.seed(4)

    def TEST1():
        n = 5
        adj = np.ones([n] * 2) * np.inf
        w = [3, 4, 2, 1, 5, 1]
        x = [0, 0, 1, 2, 2, 3, ]
        y = [1, 2, 3, 1, 3, 4, ]
        adj[x, y] = w
        # is_acyclic, distance = toposort(adj, 0)

        neighbors_list = {}
        for i in range(n):
            neighbors_list[i] = np.where(adj[i] != np.inf)[0]
        flag_acyclic, topolist = is_acyclic(adj, neighbors_list)
        print(flag_acyclic)
        if flag_acyclic:
            distance = cal_longest_path(adj, topolist, neighbors_list, index_start=0)
            print(distance)  # [ 0.  5.  4.  9. 10.]

        print('q-paths')
        for node_end in range(n):
            c, p, q = q_paths(adj, n-1, s=0, t=node_end)
            print(c, q)

    def TEST2():
        n, m = 1000, 10
        # g = nx.dense_gnm_random_graph(1000, 10000)
        adj = generate_a_barabasi_albert_graph_with_k_features(n, m, n, 1)[0]
        neighbors_list = {}
        for i in range(n):
            neighbors_list[i] = np.where(adj[i] != np.inf)[0]
        num_iters = 10

        t0 = time.time()
        for _ in range(num_iters):
            is_acyclic_ = is_acyclic(adj, neighbors_list)
        print(is_acyclic_[0], time.time() - t0)


    TEST1()