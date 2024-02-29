#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time : 2022/1/31 14:14
# @Author : Jin Echo
# @File : dijkstra.py

import heapq
import random
import numpy as np


def dijkstra(adj, s, d=None, bound=np.inf, mask=None):
    """

    :param adj: (n, n)  inf
    :param s: id_start
    :param d: id_end  if d is not None, then algorithm will terminate once the determined node is d.
    :param bound: float
    :param mask: np.bool (n,)
    :return:
    """
    # assert (adj > 0).all()
    n = adj.shape[0]
    pqueue = []
    heapq.heappush(pqueue, (0., s))
    visited = np.zeros(n, dtype=bool)  # [False] * n  # visited = set()
    if mask is not None:
        visited[np.nonzero(mask)] = True
    # visited = set()
    pre = np.arange(n)
    distance = np.ones((n,)) * np.inf
    distance[s] = 0.

    while len(pqueue) > 0:
        dist, v = heapq.heappop(pqueue)

        if dist > bound:
            break

        if visited[v]:
            if all(visited):
                break
            continue
        else:
            visited[v] = True
            if v == d:
                break

        for w in np.where(adj[v, :] != np.inf)[0]:
            if not visited[w]:
                if dist + adj[v][w] < distance[w]:
                    distance[w] = dist + adj[v][w]
                    heapq.heappush(pqueue, (distance[w], w))
                    pre[w] = v
    return pre, distance


def get_sequence(pre, start, end):
    """
    @param pre:
    @param start:
    @param end:
    @return: list
    """
    res = [end]
    while end != start:
        if end == pre[end]:
            res = [-1]
            return res
        end = pre[end]
        res.append(end)
    res.reverse()
    return res


if __name__ == '__main__':
    import networkx as nx
    import time
    import matplotlib.pyplot as plt
    np.random.seed(2)
    random.seed(4)

    def TEST0():
        n, m = 1000, 100
        G = nx.barabasi_albert_graph(n, m)
        # plt.figure(1)
        # nx.draw(G, with_labels=True, font_weight='bold')

        edges_idx = np.array(G.edges)
        # print(edges_idx)
        num_edges = (n - m) * m
        assert num_edges == edges_idx.shape[0]

        adj = np.ones([n] * 2) * np.inf
        adj[edges_idx[:, 0], edges_idx[:, 1]] = np.random.rand(num_edges)
        adj[edges_idx[:, 1], edges_idx[:, 0]] = adj[edges_idx[:, 0], edges_idx[:, 1]]

        t0 = time.time()
        for _ in range(100):
            pre, distance = dijkstra(adj, 0)  # If a point is unreachable, its parent is itself
        print(time.time() - t0)  # ndarray:7.35  7.40 7.36  set:6.99 6.87 6.94 6.83
        # print(pre)   # [0 2 0 5 1 2]
        # print(distance)   # [0.         0.64815807 0.13457995 0.70846226 1.43349321 0.62881678]
        print(get_sequence(pre, 0, n - 1))   # [0, 2, 5]  [0, 900, 390, 999]
        # plt.show()

    def TEST1():
        n, m = 6, 2
        G = nx.barabasi_albert_graph(n, m)
        # plt.figure(1)
        # nx.draw(G, with_labels=True, font_weight='bold')

        edges_idx = np.array(G.edges)
        # print(edges_idx)
        num_edges = (n - m) * m
        assert num_edges == edges_idx.shape[0]

        adj = np.ones([n] * 2) * np.inf
        # adj[range(n), range(n)] = 0
        adj[edges_idx[:, 0], edges_idx[:, 1]] = np.random.rand(num_edges)
        adj[edges_idx[:, 1], edges_idx[:, 0]] = adj[edges_idx[:, 0], edges_idx[:, 1]]


        pre, distance = dijkstra(adj, 0)
        print(pre)   # [0 2 0 5 1 2]  [0 2 0 2 1 2]
        print(distance)   # [0.         0.46192113 0.4359949  0.8563627  0.89724353 0.76632972]
        print(get_sequence(pre, 0, n - 1))   # [0, 2, 5]
        # plt.show()

    def TEST2():
        n = 8
        adj = np.ones([n] * 2) * np.inf
        w = np.random.randint(1, 10, (13,))   # dijkstra中最大边权要置为14及以上
        x = [0, 0, 0, 1, 2, 1, 2, 3, 4, 5, 4, 5, 6]
        y = [1, 2, 3, 2, 3, 4, 5, 6, 5, 6, 7, 7, 7]
        adj[x, y] = w
        # adj[y, x] = w
        print(adj)
        print(w)
        pre, distance = dijkstra(adj, n-1)
        print(pre)
        print(distance)   # [inf inf inf inf inf inf inf  0.]
        print(get_sequence(pre, n-1, 0))          # [-1]

    def TEST3():
        # test directed graphs
        n = 8
        adj = np.ones([n] * 2) * np.inf
        w = np.random.randint(1, 10, (13,))
        x = [0, 0, 0, 1, 2, 1, 2, 3, 4, 5, 4, 5, 6]
        y = [1, 2, 3, 2, 3, 4, 5, 6, 5, 6, 7, 7, 7]
        adj[x, y] = w
        # adj[y, x] = w

        # print(adj)
        # print(w)
        pre, distance = dijkstra(adj, 0)
        print(pre)
        print(distance)
        print(get_sequence(pre,0, n-1))

        pre, distance = dijkstra(adj, n-1)
        print(pre)
        print(distance)
        print(get_sequence(pre, n-1, 0))

        pre, distance = dijkstra(adj.T, n - 1)
        print(pre)  # [3 2 5 6 7 7 7 7]
        print(distance)  # [17. 12.  9. 10.  5.  6.  8.  0.]
        print(get_sequence(pre, n - 1, 0))  #  [7, 6, 3, 0]

    def TEST4():
        import os, sys
        sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
        from environment.graph import generate_a_barabasi_albert_graph_with_no_constraints
        import time

        np.random.seed(1)
        random.seed(1)


        def dijkstra1(adj, s):
            n = adj.shape[0]
            pqueue = []
            heapq.heappush(pqueue, (0., s))
            visited = set()
            pre = np.arange(n)
            distance = np.ones((n,)) * np.inf
            distance[s] = 0.

            while len(pqueue) > 0:
                dist, v = heapq.heappop(pqueue)
                visited.add(v)

                for w in np.where((adj[v, :] > 0) & (adj[v, :] != np.inf))[0]:
                    if w not in visited:
                        if dist + adj[v][w] < distance[w]:
                            distance[w] = dist + adj[v][w]
                            heapq.heappush(pqueue, (distance[w], w))
                            pre[w] = v

            return pre, distance

        def dijkstra2(adj, s):
            n = adj.shape[0]
            pqueue = []
            heapq.heappush(pqueue, (0., s))
            # visited = set()
            pre = np.arange(n)
            distance = np.ones((n,)) * np.inf
            distance[s] = 0.

            while len(pqueue) > 0:
                dist, v = heapq.heappop(pqueue)

                for w in np.where((adj[v, :] > 0) & (adj[v, :] != np.inf))[0]:
                    # if w not in visited:
                    if dist + adj[v][w] < distance[w]:
                        distance[w] = dist + adj[v][w]
                        heapq.heappush(pqueue, (distance[w], w))  # O(logn)
                        pre[w] = v

            return pre, distance

        n, m, max_n = 1000, 100, 1000
        adj = generate_a_barabasi_albert_graph_with_no_constraints(n, m, max_n)
        adj = adj.squeeze(0)

        t0 = time.time()
        dijkstra1(adj, 0)
        print((time.time()-t0))  # 0.39  0.29

        t0 = time.time()
        dijkstra2(adj, 0)
        print((time.time() - t0) )  # 0.86  0.83

        t0 = time.time()
        dijkstra(adj, 0)
        print((time.time() - t0))  # 0.17  0.14

    TEST0()
