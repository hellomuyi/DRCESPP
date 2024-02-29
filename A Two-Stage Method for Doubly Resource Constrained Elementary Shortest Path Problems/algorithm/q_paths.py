#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time : 2022/8/24 9:52
# @Author : Jin Echo
# @File : q_paths.py

import numpy as np

def op_add(C, P, adj, left_v):
    """
    Define the addition operator in the original paper.
    :param C: The longest path length from s to each node at the previous q.
    :param P: ndarray (n, q)
    :param adj:
    :param left_v:
    """

    # n = len(adj)
    # C_ = np.copy(C)  # (n,)
    # P_ = [[]] * n
    # # for i in range(n):  # 求下一跳终点为i的最大路径及长度
    # if left_v is None:
    #     left_v = range(n)
    # for i in left_v:
    #     idx = np.argmax(C + adj[:, i])  # 选中idx->i
    #     C_[i] = C[idx] + adj[idx, i]
    #     P_[i] = P[idx] + [i]
    # return C_, P_

    # 整体运算, 在复杂图上改进明显 100->80
    next_hop = np.arange(len(adj))
    idx_ = np.argmax(np.expand_dims(C,1) + adj, axis=0)
    C__ = C[idx_] + adj[idx_, next_hop]
    P__ = np.concatenate((P[idx_], np.expand_dims(next_hop, 1)), axis=1)
    return C__, P__


"""
Romesh Saigal, (1968) Letter to the Editor—A Constrained Shortest Route Problem. Operations Research 16(1):205-209.
https://doi.org/10.1287/opre.16.1.205

Marc Rosseel, (1968) Letter to the Editor—Comments on a Paper By Romesh Saigal: “A Constrained Shortest Route Problem”.
Operations Research 16(6):1232-1234. http://dx.doi.org/10.1287/opre.16.6.1232
"""
def q_paths(adj, max_q, s, t, masked_v=None):
    """
    Calculate the longest path from s to t through at most max_q arcs, which may contain repeated arcs.
    =max{f(q=1), f(q=2),..., f(q=q)}
    cf. Letter to the Editor—A Constrained Shortest Route Problem

    :param adj:  (n, n)
    :param max_q: int
    :param s: id_start
    :param t: id_end
    :param masked_v:
    :return C_max: The length of the longest path. If the value is -inf, it indicates that no s-t path exists under any q.
            P_max: longest path
            q_star: q corresponds to the longest path

    """
    adj = np.copy(adj)
    n = len(adj)

    # init
    adj[adj == np.inf] = -np.inf  # !!!!
    C = adj[s, :]
    P = [[s, i] for i in range(n)]
    P = np.array(P)  # final shape will be (max_q+1, n)

    C_max, P_max, q_star = C[t], np.copy(P[t]), 1

    for q in range(2, max_q+1):
        C, P = op_add(C, P, adj, left_v=None)
        if C[t] > C_max:
            C_max, P_max, q_star = C[t], np.copy(P[t]), q

    return C_max, P_max, q_star


def q_paths_all(adj, max_q, s, masked_v=None):
    """
    Calculate the longest path length from s to all nodes through at most max_q arcs, which may contain repeated arcs.

    :param adj:  (n,n)
    :param max_q: int
    :param s: id_start
    :param t: id_end
    :param masked_v:
    :return WC: (n, max_q)  WC(i,j):  The longest path length from s to i via at most j+1 hops (not exactly via j+1)
                                        <=> max{q=1,2,...,j+1}
    """
    adj = np.copy(adj)
    n = len(adj)
    WC = np.zeros((n, max_q))

    # init
    adj[adj == np.inf] = -np.inf
    C = adj[s, :]
    WC[:, 0] = C
    P = [[s, i] for i in range(n)]
    P = np.array(P)  # final shape will be (max_q+1, n)


    for q in range(2, max_q+1):
        C, P = op_add(C, P, adj, left_v=None)
        WC[:, q-1] = np.max((WC[:, q-2], C), axis=0)
    return WC


if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    from environment.graph import generate_a_barabasi_albert_graph
    import random

    def TEST1():
        np.random.seed(1)
        random.seed(1)

        max_num_nodes, num_nodes, num_edges = 4, 4, 2
        index_end = num_nodes - 1

        adj = generate_a_barabasi_albert_graph(num_nodes, num_edges, max_num_nodes)
        adj = adj.squeeze(0)  # (n, n)

        max_q = 4
        C_max, P_max, q_star = q_paths(adj[:, :max_num_nodes], max_q, 0, index_end)
        WC = q_paths_all(adj[:, :max_num_nodes], max_q, 0)
        print(WC[index_end, max_q-1])  # 2.16097

        print(C_max, q_star)  # 2.16097  3
        print(P_max)  # [0 3 0 3]
        print(1)

    def TEST2():
        adj = np.array([[0., 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0,]])
        adj[adj==0] = np.inf
        print(q_paths_all(adj.T, 3, 3))
        print(1)

    TEST1()