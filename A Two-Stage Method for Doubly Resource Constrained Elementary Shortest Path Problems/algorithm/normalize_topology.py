#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time : 2022/3/17 15:45
# @Author : Jin Echo
# @File : normalize_topology.py

import numpy as np
import copy

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from algorithm.dijkstra import dijkstra, get_sequence
from utils.rcespp_utils import is_elementary, is_resource_feasible, recovery, remove


def normalize_k(adj, resource_limits, cost_min, index_start, index_end, mask=None, is_aggregate=False, is_qpaths=False):
    """
    Preprocessing for feature in {2,...,K'} in one iteration, i.e.,  Tests/Reductions by infeasibility on resource limits;
    @param adj:  (n, Kn)
    @param resource_limits: ndarray (K-1,2)
    @param cost_min: current best cost
    @param index_start:
    @param index_end:
    @param mask: ndarray.bool (n, )  mutable object
    @param is_aggregate: aggregate constraints?
    return normalized_adj
           cost_min、path_best
           flag_v： True-new nodes are removed for feature in {2,...,K'} in this iteration
           flag_e: True-new edges are removed for feature in {2,...,K'} in this iteration
    """
    # assert adj.ndim == 2, 'dim error'
    n = adj.shape[0]
    K = adj.shape[1] // n
    # assert adj.shape[1] == k*n

    adj_ = copy.copy(adj)
    flag_e = False
    flag_v = False
    path_best = ()

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
        lower_limit, upper_limit = resource_limits[k-1]

        p_f, d_f = dijkstra(adj_k, index_start, d=None, bound=upper_limit, mask=mask)
        # Tests on upper limits
        if d_f[index_end] > upper_limit:
            mask[np.arange(n)] = 1
            mask[[index_start, index_end]] = 0
            # flag_e = False
            # flag_v = False
            adj_ = np.ones_like(adj_) * np.inf
            break
        else:
            seq_tmp = get_sequence(p_f, index_start, index_end)
            if is_elementary(seq_tmp):
                cost_cur = np.add.reduce(adj_[seq_tmp[:-1], seq_tmp[1:]])
                if cost_cur < cost_min and is_resource_feasible(seq_tmp, adj_[:, n:K * n], resource_limits[:K - 1, :]):
                        cost_min, path_best = cost_cur, np.copy(seq_tmp)

        # Tests on lower limits
        adj_k_noinf = np.copy(adj_k)
        adj_k_noinf[adj_k_noinf == np.inf] = 0
        bound_out = np.add.reduce(np.max(adj_k_noinf, axis=1)) - np.max(adj_k_noinf[index_end, :])
        bound_in = np.add.reduce(np.max(adj_k_noinf, axis=0)) - np.max(adj_k_noinf[:, index_start])
        bound = np.min((bound_in, bound_out))

        if bound < lower_limit:
            mask[np.arange(n)] = 1
            mask[[index_start, index_end]] = 0
            # flag_e = False
            # flag_v = False
            adj_ = np.ones_like(adj_) * np.inf
            break

        p_b, d_b = dijkstra(adj_k.T, index_end, d=None, bound=upper_limit, mask=mask)  # transposition !!

        # Mapping tables of forward paths, backward paths of each node
        dict_f, dict_b = {}, {}

        # 1.a  Reductions on upper limits through nodes，找到需要去掉(不满足约束上限)的节点
        idx = d_f + d_b > upper_limit
        idx[[index_start, index_end]] = False  # only for intermediate nodes
        idx = np.where(idx)[0]
        # mask_vidx_new = mask_v[idx] == False  # new mask node
        # mask_vidx_new = idx[mask_vidx_new]
        # mask_v[mask_vidx_new] = 1
        mask_vidx_new = np.where(mask[idx] == False)[0]  # new mask node
        if len(mask_vidx_new):
            flag_v = True
            mask_vidx_new = idx[mask_vidx_new]
            mask[mask_vidx_new] = 1

        # 1.b modify graph
        adj_[mask_vidx_new, :] = np.inf
        for i in range(K_):
            adj_[:, i * n + mask_vidx_new] = np.inf

        # 2.a Reductions on upper limits through arcs，找到需要去掉(不满足约束上限)的边
        """
        注意：邻接矩阵中节点自己到自己的边权设定为inf，即邻接矩阵中，主对角线元素全为inf
        下面第一行代码，对邻接矩阵的nxn条潜在边，去除边权为inf的边
        边权为inf的边包括：无边、自己到自己的边(其实也是无边)
        """
        nodes_i, nodes_j = np.where(adj_k != np.inf)
        mask_e_idx = np.where(d_f[nodes_i] + adj_k[nodes_i, nodes_j] + d_b[nodes_j] > resource_limits[k - 1, 1])[0]

        # 2.b modify graph
        if 0 != len(mask_e_idx):
            flag_e = True
            for i in range(K_):
                adj_[nodes_i[mask_e_idx], i * n + nodes_j[mask_e_idx]] = np.inf
        # left_e_idx = np.setdiff1d(np.arange(len(nodes_i)), mask_e_idx, assume_unique=True)
        left_e_idx = np.ones(len(nodes_i), np.int)
        left_e_idx[mask_e_idx] = 0
        left_e_idx = np.where(left_e_idx>0.5)[0]

        # 2.a_  Reductions on lower limits through arcs，嵌入对资源下限的判断,进一步去除边
        if len(left_e_idx):
            adj_k_noinf = np.copy(adj_k)
            adj_k_noinf[adj_k_noinf == np.inf] = 0
            max_row = np.max(adj_k_noinf, axis=1)  # 每行的最大值,对应所有节点的出弧的最大值
            max_col = np.max(adj_k_noinf, axis=0)  # 每列的最大值,对应所有节点的入弧的最大值
            bound_out = np.add.reduce(max_row) - max_row[index_end]
            bound_in = np.add.reduce(max_col) - max_col[index_start]

            if np.min((bound_in, bound_out)) - np.max(adj_k_noinf) + np.min(adj_k) < lower_limit:
                nodes_i_, nodes_j_ = nodes_i[left_e_idx], nodes_j[left_e_idx]
                bounds_out = bound_out - max_row[nodes_i_] + adj_k[nodes_i_, nodes_j_]
                bounds_in = bound_in - max_col[nodes_j_] + adj_k[nodes_i_, nodes_j_]
                bounds = np.min((bounds_in, bounds_out), axis=0)

                mask_eidx_new = np.where(bounds < lower_limit)[0]
                mask_eidx_new = left_e_idx[mask_eidx_new]
                # print(len(mask_eidx_new))

                # 2.b modify graph
                if 0 != len(mask_eidx_new):
                    flag_e = True
                    for i in range(K_):
                        adj_[nodes_i[mask_eidx_new], i * n + nodes_j[mask_eidx_new]] = np.inf
                    left_e_idx = np.setdiff1d(left_e_idx, mask_eidx_new, assume_unique=True)

        # 2.c  improve UB，对剩余边,求拼接路径,更新UB
        # mask_e_idx为在当前第k个约束矩阵上超过约束上限的边，一定不会满足所有约束
        for edge_cur in left_e_idx:
            node_i, node_j = nodes_i[edge_cur], nodes_j[edge_cur]
            if node_i not in dict_f.keys():
                seq_f = get_sequence(p_f, index_start, node_i)  # forward partial path
                # dict_f[node_i] = seq_f
                dict_f[node_i] = tuple(seq_f)
            else:
                # seq_f = dict_f[node_i]
                seq_f = list(dict_f[node_i])

            if node_j not in dict_b.keys():
                seq_b = get_sequence(p_b, index_end, node_j)
                # dict_b[node_j] = copy.copy(seq_b)
                dict_b[node_j] = tuple(seq_b)
            else:
                # seq_b = copy.copy(dict_b[node_j])
                seq_b = list(dict_b[node_j])
            seq_b.reverse()
            seq_tmp = seq_f + seq_b  # list   joined path via arc <i,j>

            if is_elementary(seq_tmp):
                cost_cur = np.add.reduce(adj_[seq_tmp[:-1], seq_tmp[1:]])
                if cost_cur < cost_min and is_resource_feasible(seq_tmp, adj_[:, n:K * n], resource_limits[:K - 1, :]):
                    cost_min, path_best = cost_cur, np.copy(seq_tmp)
                    # print('normalize topology:edge')

    # mask_v = np.where(mask_v == 1)[0]
    if is_aggregate:
        adj_ = adj_[:, :K*n]
    return adj_, cost_min, path_best, flag_v, flag_e


if __name__ == '__main__':
    def TEST1():
        # test normalize
        import random
        np.random.seed(2)
        random.seed(1)
        from environment.graph import generate_a_barabasi_albert_graph

        max_num_nodes, num_nodes, num_edges = 20, 20, 4
        index_end = num_nodes - 1

        delay_l, delay_u = 0.6, 0.6 * 2
        constrain = [[delay_l, delay_u]]
        constrain = np.array(constrain)

        adj = generate_a_barabasi_albert_graph(num_nodes, num_edges, max_num_nodes)
        adj = adj.squeeze(0)
        # print(adj)
        mask = np.zeros(max_num_nodes, int)
        adj_, c_min, path_best, flag_v, flag_e = normalize_k(adj, constrain, np.inf, 0, index_end, mask)
        print(c_min, path_best)   # 0.9154462957740362 [ 0  6 19]
        print(np.where(mask)[0])  # [11 14 15 16 17 18]
        print(1)

    TEST1()
