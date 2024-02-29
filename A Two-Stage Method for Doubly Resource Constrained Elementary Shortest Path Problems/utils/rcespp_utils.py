#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time : 2022/8/20 13:54
# @Author : Jin Echo
# @File : rcespp_utils.py

"""
Components needed for preprocessing
"""

import numpy as np


def is_elementary(path):
    """
    :param path: list or tuple, sequence of nodes

    """
    return len(set(path)) == len(path)


def is_resource_feasible(path, adj, limits):
    """

    :param path: list or tuple
    :param adj: (n, (K-1)n)
    :param limits: (K-1, 2)
    """
    n = adj.shape[0]
    num_resource = limits.shape[0]
    assert num_resource == adj.shape[1] // n
    for i in range(num_resource):
        f_sum = np.add.reduce(adj[:, i * n:(i + 1) * n][path[:-1], path[1:]])
        if (f_sum < limits[i, 0]) or (f_sum > limits[i, 1]):
            return False
    return True
    # f_sum = [np.add.reduce(adj[:, i * n:(i + 1) * n][path[:-1], path[1:]]) for i in range(num_resource)]
    # f_sum = np.array(f_sum)
    # return np.all(f_sum >= limits[:, 0]) and np.all(f_sum <= limits[:, 1])


def aggregate_constraints(adj, resource_limits):
    """

    :param adj: (n,Kn) cost + resource
    :param limits: (K-1,2)
    :return (n,Kn+n), (K,2)
    """
    K = adj.shape[1] // adj.shape[0]
    if K >= 3:
        adj_agg = np.add.reduce(np.split(adj, K, axis=1)[1:])
        constraint_agg = np.add.reduce(resource_limits, axis=0)
        return np.hstack((adj, adj_agg)), np.vstack((resource_limits, constraint_agg))
    else:
        return adj, resource_limits



def remove(node_seq, masked_v):
    # masked_v is in ascending order
    node_seq = np.copy(node_seq)
    node_seq = np.array(node_seq)
    for v in masked_v[::-1]:
        node_seq[node_seq >= v] -= 1
    return node_seq


def recovery(node_seq, masked_v):
    # masked_v is in ascending order
    node_seq = np.copy(node_seq)
    node_seq = np.array(node_seq)
    for v in masked_v:
        node_seq[node_seq >= v] += 1
    return node_seq


if __name__ == '__main__':

    def test_recovery():
        a = np.arange(0, 50)
        np.random.shuffle(a)
        d = [3,5,8,4,40]
        print(a)
