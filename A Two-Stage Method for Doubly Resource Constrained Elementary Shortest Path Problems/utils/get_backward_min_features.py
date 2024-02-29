#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time : 2022/4/5 0:14
# @Author : Jin Echo
# @File : get_backward_min_features.py
import numpy as np
from algorithm.dijkstra import dijkstra
from algorithm.q_paths import q_paths_all


def get_backward_min_features(adj, node_idx):
    """
    Calculate the shortest path starting from node_idx to each node in the reverse graph.
    @param adj: ndarray (n, nK)
    @param node_idx:
    @return: ndarray (max_num_nodes, K)
    """
    # assert adj.ndim == 2
    n = adj.shape[0]
    K = adj.shape[1] // n

    back_constraint = np.zeros((n, K))
    for k in range(K):
        _, back_constraint[:, k] = dijkstra(adj[:, k * n:(k + 1) * n].T, node_idx)  # Transpose here.
    return back_constraint


def get_backward_max_features(adj, node_idx, max_q) -> np.ndarray:
    """
    return R
    R[k, i, q]: in k-th feature graph, based on q-paths algorithm, the longest path length from node_idx to i through at most max_q arcs.
    If it is acyclic graph, the topological sorting is reasonable.
    @param adj: ndarray (n, nK)
    @param node_idx: int
    @return: (K, num_nodes, 1+maxq), R[:, :, 0] = 0 !!!!
    """
    # assert adj.ndim == 2
    n = adj.shape[0]
    K = adj.shape[1] // n

    back_max_constraint = np.zeros((K, n, max_q+1))
    for k in range(K):
        back_max_constraint[k, :, 0] = np.zeros(n)
        back_max_constraint[k, :, 1:] = q_paths_all(adj[:, k * n:(k + 1) * n].T, max_q, node_idx)  # Transpose here.
    # 定义起点经0跳到自己的距离为0，则经q>=1跳的最长距离>=0，WC[:, s, q>=1]本应=np.max(0,WC[:, s, q>=1])
    # 又该环境中，pulse不可能有环不存在s->...->s，故WC[:, s, q>=1]=-inf，故取max后全0
    back_max_constraint[:, node_idx, :] = 0
    return back_max_constraint


if __name__ == '__main__':
    import random
    from environment.graph import generate_a_barabasi_albert_graph, generate_a_barabasi_albert_graph_with_k_features
    # import sys, os
    # sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

    def TEST1():
        # get_back_constraint_batch
        np.random.seed(2)
        random.seed(1)

        max_num_nodes, num_nodes, num_edges = 20, 20, 4
        index_end = num_nodes - 1
        max_l = [(num_nodes - 1) * 2] * 2
        max_c = [(num_nodes - 1) * 2 * 1.] * 2
        max_feature = [2] * 2

        adj = generate_a_barabasi_albert_graph(num_nodes, num_edges, max_num_nodes)
        adj = adj.squeeze(0)
        # print(adj)
        back_constraint = get_backward_min_features(adj[:, max_num_nodes:], node_idx=0)
        print(back_constraint)
        print(1)


    def TEST2():
        import sys, os
        sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
        from environment.graph import generate_a_barabasi_albert_graph
        import random
        np.random.seed(1)
        random.seed(1)

        max_num_nodes, num_nodes, num_edges = 10, 10, 4  # 4, 4, 2
        index_end = num_nodes - 1

        adj = generate_a_barabasi_albert_graph(num_nodes, num_edges, max_num_nodes)
        adj = adj.squeeze(0)  # (n, 2n)

        max_q = 4
        backward_max_features = get_backward_max_features(adj, index_end, max_q)
        print(backward_max_features)
        print(1)
    TEST2()
