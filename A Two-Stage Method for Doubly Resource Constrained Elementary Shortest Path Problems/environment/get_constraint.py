#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time : 2022/3/29 21:10
# @Author : Jin Echo
# @File : get_constraint.py
import numpy as np
from algorithm.dijkstra import dijkstra, get_sequence


def get_constraint(adj, index_start, index_end, **kwargs):
    """
    Generate upper and lower limits on resource constraints.
    # 注意一个特例,\pi1 == \pi2,此时会有一些问题需要处理
    @param adj: ndarray (n, nk)
    @param index_start: int
    @param index_end:
    @param ratio_lower: float  constraint_lower=min * ratio_lower
    @param ratio_upper: float  constraint_upper=min * ratio_upper
    @param p_lower: float or ndarray(K-1,) i.e., p is not shared for each constraint
    @param p_upper: the tightness of the constraint
    @param q:   q \in (0,1]  float or ndarray(K-1,) i.e., p is not shared for each constraint
    @return: ndarray (K-1, 2)
    """
    # assert adj.ndim == 2
    n = adj.shape[0]
    K = adj.shape[1] // n
    # assert adj.shape[1] == k*n

    constraint = np.zeros((K-1, 2))
    if 'ratio_lower' in kwargs:
        # Cost is not needed in this case.
        ratio_lower, ratio_upper = kwargs['ratio_lower'], kwargs.get('ratio_upper')
        for k in range(1, K):
            _, f_sum = dijkstra(adj[:, k * n:(k + 1) * n], index_start, index_end)
            constraint[k-1] = f_sum[index_end] * np.array([ratio_lower, ratio_upper])  # f_sum[[index_end]] * [ratio_lower, ratio_upper]
    elif 'q' in kwargs:
        q = kwargs['q']
        pre_cost_min, _ = dijkstra(adj[:, :n], index_start, index_end)
        path_cost_min = get_sequence(pre_cost_min, index_start, index_end)  # \pi1
        for k in range(1, K):
            adj_k = adj[:, k * n:(k + 1) * n]
            _, constraint_min_all = dijkstra(adj_k, index_start, index_end)
            constraint_min = constraint_min_all[index_end]  # # f_2(\pi2)
            constraint_by_cost_min = np.add.reduce(adj_k[path_cost_min[:-1], path_cost_min[1:]])  # f_2(\pi1)
            interval = constraint_by_cost_min - constraint_min  #  - 1e-5
            # q为占interval的中心比例
            eps = 1e-5
            if isinstance(q, float) or isinstance(q, int):
                constraint[k - 1, 0] = constraint_min + interval * (1 - q) / 2 - eps
                constraint[k - 1, 1] = constraint_by_cost_min - interval * (1 - q) / 2 + eps
            elif isinstance(q, np.ndarray) or isinstance(q, list):
                constraint[k - 1, 0] = constraint_min + interval * (1 - q[k-1]) / 2 - eps
                constraint[k - 1, 1] = constraint_by_cost_min - interval * (1 - q[k-1]) / 2 + eps
            # assert (1+q)/(1-q) <= constraint_by_cost_min / constraint_min
            # if isinstance(q, float) or isinstance(q, int):
            #     constraint[k - 1] = np.array([1+q, 1-q]) * [constraint_min, constraint_by_cost_min]
            # elif isinstance(q, np.ndarray) or isinstance(q, list):
            #     constraint[k - 1] = np.array([1 + q[k-1], 1 - q[k-1]]) * [constraint_min, constraint_by_cost_min]
            else:
                raise ValueError('type error')
    elif 'q_hat' in kwargs:
        q_hat = kwargs['q_hat']
        # q为偏离interval右侧的比例
        eps = 5e-4  # -4而不是-5
        # chunks = 10
        chunks = kwargs.get('chunks', 10)
        pre_cost_min, _ = dijkstra(adj[:, :n], index_start, index_end)
        path_cost_min = get_sequence(pre_cost_min, index_start, index_end)  # \pi1
        for k in range(1, K):
            adj_k = adj[:, k * n:(k + 1) * n]
            _, constraint_min_all = dijkstra(adj_k, index_start, index_end)
            constraint_min = constraint_min_all[index_end]  # # f_2(\pi2)
            constraint_by_cost_min = np.add.reduce(adj_k[path_cost_min[:-1], path_cost_min[1:]])  # f_2(\pi1)
            interval = constraint_by_cost_min - constraint_min  # - 1e-5
            if isinstance(q_hat, float) or isinstance(q_hat, int):
                constraint[k - 1, 0] = constraint_by_cost_min + interval * q_hat + eps
                constraint[k - 1, 1] = constraint[k - 1, 0] + interval / chunks - 2*eps
            elif isinstance(q_hat, np.ndarray) or isinstance(q_hat, list):
                constraint[k - 1, 0] = constraint_by_cost_min + interval * q_hat[k - 1] + eps
                constraint[k - 1, 1] = constraint[k - 1, 0] + interval / chunks - 2*eps
    elif 'p_lower' in kwargs:
        # Santos L, Coutinho-Rodrigues J, Current JR. An improved solution algorithm for the constrained shortest path problem.
        # Transportation Research Part B 2007;41:756–71
        p_lower, p_upper = kwargs['p_lower'], kwargs.get('p_upper')
        assert (np.array(p_lower) < np.array(p_upper)).all()
        pre_cost_min, _ = dijkstra(adj[:, :n], index_start, index_end)
        path_cost_min = get_sequence(pre_cost_min, index_start, index_end)
        for k in range(1, K):
            adj_k = adj[:, k * n:(k + 1) * n]
            pre_constraint_min, constraint_min_all = dijkstra(adj_k, index_start, index_end)
            constraint_min = constraint_min_all[index_end]
            constraint_by_cost_min = np.add.reduce(adj_k[path_cost_min[:-1], path_cost_min[1:]])
            if isinstance(p_lower, np.ndarray) or isinstance(p_lower, list):
                constraint[k - 1] = constraint_min + np.array([p_lower[k - 1], p_upper[k - 1]]) * (constraint_by_cost_min - constraint_min)
            elif isinstance(p_lower, float) or isinstance(p_lower, int):
                constraint[k - 1] = constraint_min + np.array([p_lower, p_upper]) * (constraint_by_cost_min - constraint_min)
            else:
                raise ValueError('type error')
    else:
        raise NotImplementedError
    return constraint


if __name__ == '__main__':
    import random
    import os, sys
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    from environment.graph import generate_a_barabasi_albert_graph_with_k_features

    def TEST1():
        # 测试get_constraint
        np.random.seed(1)
        random.seed(1)

        max_num_nodes, num_nodes, num_edges, K = 20, 20, 4, 3
        index_end = num_nodes - 1
        num_instances = 4
        adj = generate_a_barabasi_albert_graph_with_k_features(num_nodes, num_edges, max_num_nodes, K)
        adj = adj.squeeze(0)
        p_lower, p_upper = [0.4, 0.5], [0.7, 0.8]
        # constraint_bounds = get_constraint(adj, 0, index_end, p_lower=p_lower, p_upper=p_upper)
        constraint_bounds = get_constraint(adj, 0, index_end, q=0.)  # 0.7  0
        print(constraint_bounds)
        print(1)


    def TEST2():
        # 测试get_constraint
        np.random.seed(1)
        random.seed(1)

        max_num_nodes, num_nodes, num_edges, K = 20, 20, 4, 3
        index_end = num_nodes - 1
        num_instances = 4
        adj = generate_a_barabasi_albert_graph_with_k_features(num_nodes, num_edges, max_num_nodes, K)
        adj = adj.squeeze(0)
        p_lower, p_upper = [0.4, 0.5], [0.7, 0.8]
        constraint_bounds = get_constraint(adj, 0, index_end, p_lower=p_lower, p_upper=p_upper)
        # constraint_bounds = get_constraint(adj, 0, index_end, q_hat=0.)
        print(constraint_bounds)
        """
        [[1.41595764 2.06413763]
 [1.71267781 2.41268133]]
 
        [[2.71281762 2.92787762]
 [2.87985033 3.11218484]]
        """
        print(1)


    TEST2()