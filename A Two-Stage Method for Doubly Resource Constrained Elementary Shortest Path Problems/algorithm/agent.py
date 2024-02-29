#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time : 2022/1/21 23:12
# @Author : Jin Echo
# @File : agent.py
import numpy as np
import copy
import time
import sys, os

class GreedyAgent():
    def __init__(self, name, model, obs_shape, act_space, args):
        self.name = name
        # self.agent_index = agent_index

    def greedy_action(self, obs):
        pass


class RandomAgent():
    def __init__(self, name, model, obs_shape, act_space, args):
        self.name = name
        # self.agent_index = agent_index

    def random_action(self, obs):
        pass


class DFSAgent:
    def __init__(self,):
        raise NotImplementedError


class BFSAgent:
    def __init__(self, ):
        raise NotImplementedError


class PulseAgent:
    def __init__(self, num_nodes, index_start=0, index_end=19):
        """

        :param num_nodes:
        :param index_start:
        :param index_end:
        """
        self.num_nodes = num_nodes
        self.index_start = index_start
        self.index_end = index_end

    def dfs(self, adj, index_start, index_end, mask):
        """

        :param adj: (num_nodes, K*num_nodes)
        :param mask: set
        :return:
        """
        if index_start == index_end:
            if (self.constraint_tmp >= self.constraint_boundary[:,
                                       0] - 1e-5).all():  # and (self.constraint_tmp <= self.constraint_boundary[:, 1] + 1e-5).all():
                self.c_min = self.cost_tmp
                self.path_best = copy.deepcopy(self.path_tmp)

            return
        # for i in range(self.num_nodes):  # DFS
        #     # 扩展未访问过的节点、并且邻接矩阵有边
        #     if (i != index_start) and (i not in mask) and (adj[index_start][i] != np.inf):  # 无边记为inf
        neighbors = np.where(adj[index_start, :self.num_nodes] != np.inf)[0]
        # neighbors = self.neighbors[index_start]
        for i in neighbors:
            if i not in mask:
                # 1. bound if self.cost_tmp + adj[index_start, i] >= self.c_min:
                if self.cost_tmp + adj[index_start, i] + self.back_min_cost[i][0] >= self.c_min:
                    continue
                # 2. infeasibility on resource upper limits. not use aggregate constraints
                constraint_next_step = np.empty(self.K - 1)
                flag = False
                for k in range(1, self.K):
                    adj_k = adj[:, k * self.num_nodes: (k + 1) * self.num_nodes]
                    constraint_next_step[k - 1] = adj_k[index_start, i]
                    if self.constraint_tmp[k - 1] + constraint_next_step[k - 1] + self.back_min_resource[i, k - 1] > \
                            self.constraint_boundary[k - 1, 1] + 1e-5:
                        flag = True
                        break
                if flag:
                    continue

                # constraint_next_step = adj[index_start, i + np.arange(self.num_nodes, self.num_nodes * self.K, self.num_nodes)]

                # # 3. infeasibility on resource lower limits，
                # # nodes_left = np.ones(self.num_nodes, int)
                # # nodes_left[list(mask)] = 0
                # # nodes_left[index_end] = 0
                # # nodes_left = np.where(nodes_left > 0.5)[0]
                #
                # flag = False
                # for k in range(1, self.K):
                #     adj_k_ = self.adj_[:, k * self.num_nodes: (k + 1) * self.num_nodes]
                #     # 剩余节点的bound_out
                #
                #     bound_out = np.add.reduce(np.max(adj_k_[:-1, :], axis=1)) - np.add.reduce(np.max(adj_k_[np.array(list(mask)), :], axis=1))
                #     # bound_out_ = np.add.reduce(np.max(adj_k_[nodes_left, :], axis=1))
                #     # print(bound_out,
                #     #       self.constraint_tmp[k - 1] + constraint_next_step[k - 1] + bound_out,
                #     #       self.constraint_boundary[k - 1, 0])
                #     if self.constraint_tmp[k - 1] + constraint_next_step[k - 1] + bound_out < \
                #             self.constraint_boundary[k - 1, 0] - 1e-5:
                #         flag = True
                #         break
                # if flag:
                #     print('pulse,资源下限生效')  # 未曾生效，因为计算的上界太不准确，偏大，无论是否经过预处理
                #     continue

                # q-paths

                self.path_tmp.append(i)
                mask.add(i)
                self.cost_tmp += adj[index_start, i]
                self.constraint_tmp += constraint_next_step
                self.dfs(adj, i, index_end, mask)

                self.path_tmp.pop()
                mask.remove(i)
                self.cost_tmp -= adj[index_start, i]
                self.constraint_tmp -= constraint_next_step

    def lb_first_dfs(self, adj, index_start, index_end, mask):
        """
        :param adj: (num_nodes, K*num_nodes)
        :param mask: set
        :return:
        """
        if index_start == index_end:
            if (self.cost_tmp < self.c_min) and (self.constraint_tmp >= self.constraint_boundary[:,
                                                                        0] - 1e-5).all():  # and (self.constraint_tmp <= self.constraint_boundary[:, 1] + 1e-5).all():
                self.c_min = self.cost_tmp
                self.path_best = copy.deepcopy(self.path_tmp)
            return

        neighbors = np.where(adj[index_start, :self.num_nodes] != np.inf)[0]
        # neighbors = self.neighbors[index_start]
        R = []
        for i in neighbors:
            if i not in mask:
                # 1. bound if self.cost_tmp + adj[index_start, i] >= self.c_min:
                lb = self.cost_tmp + adj[index_start, i] + self.back_min_cost[i][0]
                if lb >= self.c_min:
                    continue
                # 2. infeasibility on resource upper limits. not use aggregate constraints
                constraint_next_step = np.empty(self.K - 1)
                flag = False
                for k in range(1, self.K):
                    if self.constraint_tmp[k - 1] + adj[index_start, k * self.num_nodes + i] + self.back_min_resource[
                        i, k - 1] > \
                            self.constraint_boundary[k - 1, 1] + 1e-5:
                        flag = True
                        break
                if flag:
                    continue

                # constraint_next_step = adj[index_start, i + np.arange(self.num_nodes, self.num_nodes * self.K, self.num_nodes)]

                # # 3. infeasibility on resource lower limits，
                # # nodes_left = np.ones(self.num_nodes, int)
                # # nodes_left[list(mask)] = 0
                # # nodes_left[index_end] = 0
                # # nodes_left = np.where(nodes_left > 0.5)[0]
                # flag = False
                # for k in range(1, self.K):
                #     adj_k_ = self.adj_[:, k * self.num_nodes: (k + 1) * self.num_nodes]
                #
                #     bound_out = np.add.reduce(np.max(adj_k_[:-1, :], axis=1)) - np.add.reduce(np.max(adj_k_[np.array(list(mask)), :], axis=1))
                #     # bound_out_ = np.add.reduce(np.max(adj_k_[nodes_left, :], axis=1))
                #     # print(bound_out,
                #     #       self.constraint_tmp[k - 1] + constraint_next_step[k - 1] + bound_out,
                #     #       self.constraint_boundary[k - 1, 0])
                #     if self.constraint_tmp[k - 1] + constraint_next_step[k - 1] + bound_out < \
                #             self.constraint_boundary[k - 1, 0] - 1e-5:
                #         flag = True
                #         break
                # if flag:
                #     print('pulse,资源下限生效')  # 未曾生效，因为计算的上界太不准确，偏大，无论是否经过预处理
                #     continue

                R.append((i, lb))

        # neighbors_ordered = None
        # if len(R) == 2:
        #     if R[0][1] > R[1][1]:
        #         neighbors_ordered = [R[1][0], R[0][0]]
        #     else:
        #         neighbors_ordered = [R[0][0], R[1][0]]
        # elif len(R) == 0:
        #     return
        # elif len(R) == 1:
        #     neighbors_ordered = [R[0][0]]
        # else:
        #     neighbors_ordered = np.array([neighbor for neighbor, _ in sorted(R, key=lambda x: x[1])])

        neighbors_ordered = [neighbor for neighbor, _ in sorted(R, key=lambda x: x[1])]
        # # R_sorted = dict(sorted(R.items(), key=lambda x: x[1]))
        # # R_sorted = R
        # neighbors_ordered = np.array([neighbor for neighbor, _ in sorted(R.items(), key=lambda x: x[1])])
        # # neighbors_ordered = np.append(neighbors_ordered[0::2], neighbors_ordered[1::2])  # skip2
        # # center diffuse
        # mid = (len(neighbors_ordered) - 1) // 2
        # tmp = np.empty_like(neighbors_ordered)
        # tmp[0::2] = np.flip(neighbors_ordered[0:mid+1])
        # tmp[1::2] = neighbors_ordered[mid+1:]
        # neighbors_ordered = tmp

        # a = list(R.keys())
        # random.shuffle(a)

        for i in neighbors_ordered:  # R_sorted.keys():
            self.path_tmp.append(i)
            mask.add(i)

            self.cost_tmp += adj[index_start, i]
            constraint_next_step = adj[
                index_start, i + np.arange(self.num_nodes, self.num_nodes * self.K, self.num_nodes)]
            self.constraint_tmp += constraint_next_step
            self.lb_first_dfs(adj, i, index_end, mask)

            self.path_tmp.pop()
            mask.remove(i)
            self.cost_tmp -= adj[index_start, i]
            self.constraint_tmp -= constraint_next_step

    def get_best_solution(self, adj, constraint_boundary, back_min_cost, back_min_resource, c_min, mode):
        """

        :param adj: (n, K*n)
        :param constraint_boundary: (K-1, 2)
        :param back_min_cost: (num_nodes, 1)  终点到各节点的最小cost之和
        :param back_min_resource: (num_nodes, K-1)
        :param c_min: inf or c_min_initial
        :return: (cost_min  path_best:list) ; (np.inf, []) for infeasible path
        """
        # self.adj_ = np.copy(adj)
        # self.adj_[adj == np.inf] = 0
        self.K = adj.shape[1] // adj.shape[0]
        self.c_min = c_min  # 初始的c_min
        self.path_best = []  # c_min对应的路径
        self.path_tmp = [self.index_start]  # 临时存储构造的路径
        self.cost_tmp = 0.  # 临时构造的路径对应的累积cost
        self.constraint_tmp = np.zeros(self.K - 1, )  # 临时构造的路径对应的累积约束(K-1个)
        self.constraint_boundary = constraint_boundary
        self.back_min_cost = back_min_cost
        self.back_min_resource = back_min_resource

        # self.neighbors = {}
        # for i in range(self.num_nodes):
        #     self.neighbors[i] = np.where(adj[i, :self.num_nodes] != np.inf)[0]

        # mask = [self.index_start]
        mask = set()
        mask.add(self.index_start)
        # mask = np.zeros(self.num_nodes, np.int); mask[self.index_start] = 1
        if mode == 'dfs':
            self.dfs(adj, self.index_start, self.index_end, mask)
        else:
            self.lb_first_dfs(adj, self.index_start, self.index_end, mask)
        return self.c_min, self.path_best


class KSPAgent:
    def __init__(self, ):
        raise NotImplementedError


if __name__ == '__main__':
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    from utils.get_backward_min_features import get_backward_min_features
    from environment.graph import generate_a_barabasi_albert_graph
    import random

    np.random.seed(1)
    random.seed(1)

    def TEST4():
        max_num_nodes, num_nodes, num_edges = 15, 15, 4
        index_end = num_nodes - 1
        num_instances = 1
        delay_l, delay_u = 0.6, 0.6 * 5
        delay_constrain_all = [delay_l, delay_u]
        delay_constrain_all = np.reshape(delay_constrain_all * num_instances, (num_instances, 1, 2))

        adj_all = generate_a_barabasi_albert_graph(num_nodes, num_edges, max_num_nodes)
        back_min_resource = get_backward_min_features(adj_all[0, :, max_num_nodes:], index_end)
        back_min_cost = get_backward_min_features(adj_all[0, :, :max_num_nodes], index_end)

        # Pulse
        t0 = time.time()
        pulse_gent = PulseAgent(num_nodes, index_start=0, index_end=index_end)
        res = pulse_gent.get_best_solution(adj_all[0], delay_constrain_all.squeeze(0),
                                           back_min_cost, back_min_resource,
                                           c_min=np.inf, mode="dfs")
        print(res)  # 0.7310462720007012, [0, 7, 6, 14]
        print('Pulse Time: {:.2f}s'.format((time.time() - t0)))


    TEST4()
