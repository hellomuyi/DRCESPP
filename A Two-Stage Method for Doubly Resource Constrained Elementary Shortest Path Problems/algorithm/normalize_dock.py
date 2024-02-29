#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time : 2022/3/26 12:55
# @Author : Jin Echo
# @File : normalize_dock.py

import numpy as np
import copy

import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from algorithm.normalize_topology import normalize_k
from algorithm.dock import dock_tree1_k
from algorithm.remove_alley_nodes import remove_alley_nodes
from utils.rcespp_utils import is_resource_feasible
from algorithm.q_paths import q_paths
from algorithm.toposort import toposort, is_acyclic, cal_longest_path


def fathoming_test_on_feasibility(adj, resource_limits, index_start, index_end,mask, is_aggregate=False):
    """
    final test by q-paths algorithm
    :param adj:  (n, Kn)
    :param resource_limits:
    :param index_start:
    :param index_end:
    :param mask:
    :param is_aggregate:
    :return:
    """
    # assert adj.ndim == 2, 'dim error'
    n = adj.shape[0]
    K = adj.shape[1] // n
    # assert adj.shape[1] == k*n

    adj_ = copy.copy(adj)

    if is_aggregate and (K >= 3):
        adj_agg = np.add.reduce(np.split(adj_, K, axis=1)[1:])
        constraint_agg = np.add.reduce(resource_limits, axis=0)
        adj_ = np.hstack((adj_, adj_agg))
        resource_limits = np.vstack((resource_limits, constraint_agg))
        K_ = K + 1
    else:
        K_ = K

    masked_v = np.where(mask > 0.5)[0]
    for k in range(1, K_):
        adj_k = adj_[:, k * n:(k + 1) * n]

        adj_k_inf = np.copy(adj_k)
        adj_k_inf = np.delete(adj_k_inf, masked_v, axis=0)  # # delete rows
        adj_k_inf = np.delete(adj_k_inf, masked_v, axis=1)  # # delete columns
        t = index_end - sum(masked_v < index_end)
        s = index_start - sum(masked_v < index_start)

        # bound_q, seq_tmp, q = q_paths(adj_k, n-len(masked_v)-1, index_start, index_end, masked_v)  # 15.44m
        bound_q, seq_tmp, q = q_paths(adj_k_inf, n - len(masked_v) - 1, s, t, masked_v=None)

        # ----------------------------------------------------------------------------------------
        if bound_q < resource_limits[k - 1, 0]:
            # print('违背资源下限')
            return True
    return False


def final_test_topsort(adj, resource_limits, index_start, index_end,mask, is_aggregate=False):
    """
    Check if there is a loop.
    final test by by topo sorting if no loop; otherwise by q-paths algorithm
    :param adj:  (n, Kn)
    :param resource_limits:
    :param index_start:
    :param index_end:
    :param mask:
    :param is_aggregate:
    :return:
    """

    # assert adj.ndim == 2, 'dim error'
    n = adj.shape[0]
    K = adj.shape[1] // n
    # assert adj.shape[1] == k*n

    adj_ = copy.copy(adj)
    masked_v = np.where(mask > 0.5)[0]

    adj_1 = adj_[:, :n]

    adj_1_inf = np.copy(adj_1)
    adj_1_inf = np.delete(adj_1_inf, masked_v, axis=0)
    adj_1_inf = np.delete(adj_1_inf, masked_v, axis=1)
    neighbors_list = {}
    for i in range(len(adj_1_inf)):
        neighbors_list[i] = np.where(adj_1_inf[i] != np.inf)[0]
    flag_acyclic, topolist = is_acyclic(adj_1_inf, neighbors_list)
    if not flag_acyclic:
        # print('有环')  # 执行q-path
        # return False
        return fathoming_test_on_feasibility(adj, resource_limits, index_start, index_end,mask, is_aggregate=is_aggregate)
    # else:
    #     print('无环')

    if is_aggregate and (K >= 3):
        adj_agg = np.add.reduce(np.split(adj_, K, axis=1)[1:])
        constraint_agg = np.add.reduce(resource_limits, axis=0)
        adj_ = np.hstack((adj_, adj_agg))
        resource_limits = np.vstack((resource_limits, constraint_agg))
        K_ = K + 1
    else:
        K_ = K


    for k in range(1, K_):
        adj_k = adj_[:, k * n:(k + 1) * n]

        adj_k_inf = np.copy(adj_k)
        adj_k_inf = np.delete(adj_k_inf, masked_v, axis=0)
        adj_k_inf = np.delete(adj_k_inf, masked_v, axis=1)
        t = index_end - len(masked_v)
        # max estimate by topo sorting
        distance = cal_longest_path(adj_k_inf, topolist, neighbors_list, index_start)

        # ----------------------------------------------------------------------------------------
        if distance[t] < resource_limits[k - 1, 0]:
            # print('违背资源下限')
            return True
    return False


def normalize_dock(adj, resource_limits, index_start, index_end):
    """
    Preprocessing Function
    @param adj:  (n, Kn)  cost || resource
    @param resource_limits: ndarray (K-1,2)
    @param index_start:
    @param index_end:
    @return: masked_adj  reduced network
             cost_min: current best cost after preprocessing
             path_best: tuple, corresponds to cost_min
             mask: ndarray (num_mask,) max {num_mask}=n-2  if len==n-2, solved !!
    """

    is_aggregate = True
    # assert adj.ndim == 2, 'dim error'
    n = adj.shape[0]
    K = adj.shape[1] // n
    adj_ = copy.copy(adj)
    # delete the incoming arcs of node s and the outgoing arcs of node t;
    adj_[:, index_start + np.arange(0, n * K, n)] = np.inf
    adj_[index_end, :] = np.inf

    cost_min, path_best = np.inf, ()
    # process path (s,t) separately
    if adj[index_start][index_end] != np.inf:
        # delete arc (s, t)
        adj_[index_start, index_end + np.arange(0, n * K, n)] = np.inf

        if is_resource_feasible([index_start, index_end], adj[:, n:], resource_limits):
            cost_min = adj[index_start, index_end]
            path_best = (index_start, index_end)

    mask = np.zeros(n, int)

    nodes_deleted = []
    cnt = 0
    while True:
        cnt += 1
        # print('cnt:', cnt)
        flag_modify = False  # c_min is updated?
        flag_break = False

        # Reductions by elementarity;
        adj_, _ = remove_alley_nodes(adj_, index_start, index_end)

        """
        通过性能分析工具，先执行对接树算法，再执行拓扑规范化算法，这个顺序比逆序要快(得多,不可忽略)，
        # 无论是运行时间还是dijkstra、get_sequence、normalize_topology_k、dock_tree_k等关键函数的调用次数
        部分参数组合下的结果略有差异，逆序结果稍好，但可忽略不计
        """

        # Tests/Reductions by optimality on cost bounds
        adj_, cost_min, path_best_, flag_v_dock, flag_e_dock = dock_tree1_k(adj_, resource_limits, cost_min,
                                                                        index_start, index_end, mask)
        if path_best_ != ():  # cost_min is updated
            path_best = path_best_
            flag_modify = True
        if (n - 2) == np.add.reduce(mask):
            break

        # Tests/Reductions by infeasibility on resource limits;
        adj_, cost_min, path_best_, flag_v_normalize, flag_e_normalize = normalize_k(adj_, resource_limits, cost_min,
                                                                                 index_start, index_end, mask,
                                                                                 is_aggregate=is_aggregate,
                                                                                 is_qpaths=False)
        if path_best_ != ():
            path_best = path_best_
            flag_modify = True
        if (n - 2) == np.add.reduce(mask):
            break

        if (not flag_v_dock) and (not flag_e_dock) and (not flag_v_normalize) and (not flag_e_normalize)\
                and (not flag_modify):
            flag_break = True  # deadlock
            break  # The break in this case may not be to find an optimal solution, but may be a deadlock caused by multiple resources conflicting with each other

        if 0 == np.sum(adj_[:, :n] != np.inf):
            # print('\n\nTrue', np.add.reduce(mask))
            mask[np.arange(n)] = 1
            mask[[index_start, index_end]] = 0
            break

        # if cnt >= 1:
        #     break

    # final test
    # 使用这个机制对resource limits较松的情况时间没有增加，在约束紧的时候增加严重。235m-305m
    if flag_break and (n - 2) != np.add.reduce(mask) and fathoming_test_on_feasibility(adj_, resource_limits, index_start,
                                                                             index_end, mask, is_aggregate=is_aggregate):
        mask[np.arange(n)] = 1
        mask[[index_start, index_end]] = 0


    mask = np.where(mask == 1)[0]
    assert len(mask) <= n - 2
    return adj_, cost_min, path_best, mask


if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    import time

    def TEST1():
        # test normalize_dock
        import random
        np.random.seed(1)
        random.seed(1)
        from environment.graph import generate_a_barabasi_albert_graph

        max_num_nodes, num_nodes, num_edges = 20, 20, 4
        index_end = num_nodes - 1

        delay_l, delay_u = 0.6 * 1.4, 0.6 * 3.5
        constrain = np.array([[delay_l, delay_u]])  # [1,2]

        adj = generate_a_barabasi_albert_graph(num_nodes, num_edges, max_num_nodes)
        adj = adj.squeeze(0)  # (n, n)
        # print(adj)
        num_iters = 1000
        t0 = time.time()
        for _ in range(num_iters):
            adj_, cost_best, path_best, mask = normalize_dock(adj, constrain, 0, index_end)
        print(time.time() - t0)  # 0.766  0.759
        # print(adj_)
        print(cost_best)
        print(path_best)  # (0, 7, 6, 19)
        # print(mask)
        print(1)

    TEST1()
