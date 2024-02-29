#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time : 2022/3/25 14:46
# @Author : Jin Echo
# @File : dock.py

import numpy as np
import copy

import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from algorithm.dijkstra import dijkstra, get_sequence
from utils.rcespp_utils import is_elementary, is_resource_feasible


def dock_tree1_k(adj, resource_limits, cost_min, index_start, index_end, mask=None):
    r"""
    Preprocessing for feature in {1} in one iteration, i.e., Tests/Reductions by optimality on cost bounds.
    @param adj: (n, Kn)
    @param resource_limits: ndarray (K-1,2)
    @param cost_min: current best cost
    @param index_start: int
    @param index_end: int
    @param mask: (n, ) mutable object
    @return: masked_adj
             cost_min、path_best
             flag_v：True-new nodes are removed for feature in {1} in this iteration
             flag_e: True-new edges are removed for feature in {1} in this iteration
    """
    # assert adj.ndim == 2, 'dim error'
    n = adj.shape[0]
    K = adj.shape[1] // n
    # assert adj.shape[1] == K*n

    adj_ = copy.copy(adj)
    # mask_v = np.zeros(n, int)
    mask_new_v = []
    undermined = np.zeros(n, )
    flag_v = False
    flag_e = False
    path_best = ()

    p_f, c_f = dijkstra(adj_[:, :n], index_start, d=None, bound=cost_min, mask=mask)
    # Tests by optimality
    if c_f[index_end] >= cost_min:
        # mask_v = np.ones(n, int)
        # mask_v[[index_start, index_end]] = 0
        # mask_v = np.where(mask_v == 1)[0]
        mask[np.arange(n)] = 1
        mask[[index_start, index_end]] = 0
        adj_ = np.ones_like(adj_) * np.inf
        return adj_, cost_min, path_best, flag_v, flag_e
    else:
        seq_tmp = get_sequence(p_f, index_start, index_end)
        if is_elementary(seq_tmp) and is_resource_feasible(seq_tmp, adj_[:, n:K * n], resource_limits[:K - 1, :]):
            cost_min, path_best = c_f[index_end], np.copy(seq_tmp)
            mask[np.arange(n)] = 1
            mask[[index_start, index_end]] = 0
            adj_ = np.ones_like(adj_) * np.inf
            return adj_, cost_min, path_best, flag_v, flag_e

    p_b, c_b = dijkstra(adj_[:, :n].T, index_end, d=None, bound=cost_min, mask=mask)  # transposition !!

    # Mapping tables of forward paths, backward paths of each node
    dict_f, dict_b = {}, {}

    # 1.a Reductions by optimality through nodes. 找到需要去掉的节点
    for j in range(n):  # 对n-2个中间节点尝试拼接
        # if (j != index_start) and (c_f[j] != np.inf) and (c_b[j] != np.inf) and (j != index_end):  # 不包括起点、终点
        if (j != index_start) and (j != index_end) and not mask[j]:  # only for intermediate nodes
            cost_cur = c_f[j] + c_b[j]
            if cost_cur >= cost_min:
                mask_new_v.append(j)
            else:
                # undermined[j] = cost_cur
                if j not in dict_f.keys():
                    seq_f = get_sequence(p_f, index_start, j)
                    seq_b = get_sequence(p_b, index_end, j)
                    # dict_f[j], dict_b[j] = seq_f, copy.copy(seq_b)
                    dict_f[j], dict_b[j] = tuple(seq_f), tuple(seq_b)
                else:
                    # seq_f, seq_b = dict_f[j], copy.copy(dict_b[j])
                    seq_f, seq_b = list(dict_f[j]), list(dict_b[j])
                seq_b.reverse()
                seq_tmp = seq_f[:-1] + seq_b  # list
                if is_elementary(seq_tmp) and \
                        is_resource_feasible(seq_tmp, adj_[:, n:K * n], resource_limits[:K-1, :]):
                    mask_new_v.append(j)
                    path_best = np.copy(seq_tmp)
                    cost_min = cost_cur
                    # undermined[j] = 0.
                else:
                    undermined[j] = cost_cur

    # 1.b modify graph
    more_nodes = np.where(undermined >= cost_min)[0]  # Yu
    if len(more_nodes):
        mask_new_v += list(more_nodes)
    mask_new_v = np.array(mask_new_v)
    if len(mask_new_v):
        mask[mask_new_v] = 1
        flag_v = True
        adj_[mask_new_v, :] = np.inf
        for k in range(K):
            adj_[:, k * n + mask_new_v] = np.inf

    # 2.a Reductions by optimality through arcs，找到需要去掉的边
    """
    注意：邻接矩阵中节点自己到自己的边权设定为inf，即邻接矩阵中，主对角线元素全为inf
    下面第一行代码，对邻接矩阵的nxn条潜在边，去除边权为inf的边
    边权为inf的边包括：无边、自己到自己的边(其实也是无边)
    """
    nodes_i, nodes_j = np.where(adj_[:, :n] != np.inf)
    undermined1 = np.zeros(len(nodes_i), )
    mask_e_idx = []
    for i in range(len(nodes_i)):
        node_i, node_j = nodes_i[i], nodes_j[i]
        cost_cur = c_f[node_i] + adj_[node_i][node_j] + c_b[node_j]
        if cost_cur >= cost_min:
            mask_e_idx.append(i)
        else:
            # undermined1[i] = cost_cur
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
            if is_elementary(seq_tmp) and is_resource_feasible(seq_tmp, adj_[:, n:K * n], resource_limits[:K - 1, :]):
                mask_e_idx.append(i)
                path_best = np.copy(seq_tmp)
                cost_min = cost_cur
                # undermined1[i] = 0.
            else:
                undermined1[i] = cost_cur

    # 2.b modify graph
    more_edges = list(np.where(undermined1 >= cost_min)[0])
    mask_e_idx = mask_e_idx + more_edges
    if 0 != len(mask_e_idx):
        flag_e = True
        for i in range(K):
            adj_[nodes_i[mask_e_idx], i * n + nodes_j[mask_e_idx]] = np.inf

    return adj_, cost_min, path_best, flag_v, flag_e


if __name__ == '__main__':
    def TEST1():
        # test dock_tree1_k
        import random
        np.random.seed(2)
        random.seed(1)
        from environment.graph import generate_a_barabasi_albert_graph

        max_num_nodes, num_nodes, num_edges = 10, 10, 4
        index_end = num_nodes - 1

        delay_l, delay_u = 0.6, 0.6 * 3
        constrain = np.array([[delay_l, delay_u]])  # [1,2]

        adj = generate_a_barabasi_albert_graph(num_nodes, num_edges, max_num_nodes)
        adj = adj.squeeze(0)  # (n, n)
        # print(adj)
        c_min = np.inf
        mask = np.zeros(max_num_nodes, int)
        adj_, c_min, path, flag_v, flag_e = dock_tree1_k(adj, constrain, c_min, 0, index_end, mask)
        print(adj_)
        print(c_min, path)  # 0.4540485594252808 [0 6 9]
        print(np.where(mask)[0])  # [1 2 3 4 5 6 7 8]
        print(1)

    TEST1()
