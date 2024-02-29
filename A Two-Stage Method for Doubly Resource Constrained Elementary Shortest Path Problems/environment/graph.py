#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time : 2022/1/22 0:27
# @Author : Jin Echo
# @File : graph.py
import networkx as nx
import numpy as np


def generate_a_barabasi_albert_graph(n: int, m: int, max_n: int) -> np.ndarray:
    """
    Randomly generate a graph instance:
    K=2, The weight feature (cost or resource consumption) values are uniformly distributed random numbers in the range (0,1)

    :param n: Number of nodes
    :param m: Number of edges to attach from a new node to existing nodes
    :param max_n: Maximum number of nodes, n<=max_n
    :return: (2, num_nodes, num_nodes*K)
    """
    G = nx.barabasi_albert_graph(n, m)  # Generate the graph topology. undirected graph
    edges_idx = np.array(G.edges)
    num_edges = (n - m) * m
    # assert num_edges == edges_idx.shape[0]

    # cost
    adj_cost = np.ones([max_n] * 2) * np.inf
    adj_cost[edges_idx[:, 0], edges_idx[:, 1]] = np.random.rand(num_edges)
    adj_cost[edges_idx[:, 1], edges_idx[:, 0]] = adj_cost[edges_idx[:, 0], edges_idx[:, 1]]
    adj_cost = np.expand_dims(adj_cost, axis=0)
    # delay
    adj_delay = np.ones([max_n] * 2) * np.inf
    adj_delay[edges_idx[:, 0], edges_idx[:, 1]] = np.random.rand(num_edges)
    adj_delay[edges_idx[:, 1], edges_idx[:, 0]] = adj_delay[edges_idx[:, 0], edges_idx[:, 1]]
    adj_delay = np.expand_dims(adj_delay, axis=0)
    adj = np.concatenate((adj_cost, adj_delay), axis=-1)
    return adj

    # torch
    # G = nx.barabasi_albert_graph(self.num_nodes, self.num_edges)
    # edges_idx = np.array(G.edges)
    #
    # adj = torch.zeros([n] * 2)
    # num_edges = (n - m) * m
    # assert num_edges == edges_idx.shape[0]
    # adj[edges_idx[:, 0], edges_idx[:, 1]] = torch.rand(num_edges)  # np.random.rand(num_edges)
    # adj[edges_idx[:, 1], edges_idx[:, 0]] = adj[edges_idx[:, 0], edges_idx[:, 1]]
    # adj = torch.unsqueeze(adj, dim=0)
    # return adj


def generate_a_barabasi_albert_graph_with_no_constraints(n: int, m: int, max_n: int) -> np.ndarray:
    """
    Randomly generate a graph instance:
    K=1, The weight feature values are uniformly distributed random numbers in the range (0,1)

    :param n: Number of nodes
    :param m: Number of edges to attach from a new node to existing nodes
    :param max_n: Maximum number of nodes, n<=max_n
    :return: (1, num_nodes, num_nodes*K)
    """
    G = nx.barabasi_albert_graph(n, m)  # Generate the graph topology. undirected graph
    edges_idx = np.array(G.edges)
    num_edges = (n - m) * m
    # assert num_edges == edges_idx.shape[0]

    adj_cost = np.ones([max_n] * 2) * np.inf
    adj_cost[edges_idx[:, 0], edges_idx[:, 1]] = np.random.rand(num_edges)
    adj_cost[edges_idx[:, 1], edges_idx[:, 0]] = adj_cost[edges_idx[:, 0], edges_idx[:, 1]]
    adj_cost = np.expand_dims(adj_cost, axis=0)
    return adj_cost


def generate_a_barabasi_albert_graph_with_k_features(n: int, m: int, max_n: int, k: int) -> np.ndarray:
    """
    Randomly generate a graph instance:
    The weight feature values are uniformly distributed random numbers in the range (0,1)

    :param n: Number of nodes
    :param m: Number of edges to attach from a new node to existing nodes
    :param max_n: Maximum number of nodes, n<=max_n
    :param k: Number of edge features
    :return: (k, num_nodes, num_nodes*K)
    """
    G = nx.barabasi_albert_graph(n, m)  # Generate the graph topology. undirected graph
    edges_idx = np.array(G.edges)
    num_edges = (n - m) * m
    # assert num_edges == edges_idx.shape[0]

    adj = []
    for i in range(k):
        adj_ = np.ones([max_n] * 2) * np.inf
        adj_[edges_idx[:, 0], edges_idx[:, 1]] = np.random.rand(num_edges)
        adj_[edges_idx[:, 1], edges_idx[:, 0]] = adj_[edges_idx[:, 0], edges_idx[:, 1]]
        adj.append(adj_)

    adj = np.concatenate(adj, axis=-1)  # (max_n, max_n*k)
    adj = np.expand_dims(adj, axis=0)   # (k, max_n, max_n*k)
    return adj


class GraphGenerator:
    def __init__(self, model):
        """
        :param model: 'barabasi_albert_graph'
        """
        self.model = model
        raise NotImplementedError


    def generate_a_instance(self):
        if self.model == 'barabasi_albert_graph':
            pass
        else:
            print('暂未实现图生成模型{}'.format(self.model))
            return NotImplementedError



if __name__ == '__main__':
    import random
    random.seed(1)      # Fixed the random seed for the network topology.
    np.random.seed(1)   # Fixed the random seed for edge weights.
    # adj = generate_a_barabasi_albert_graph_withnoconstraint(4, 2, 4)
    adj1 = generate_a_barabasi_albert_graph_with_k_features(4, 2, 4, 2)
    random.seed(1)
    np.random.seed(1)
    adj2 = generate_a_barabasi_albert_graph(4, 2, 4)
    print((adj1 == adj2).all())


