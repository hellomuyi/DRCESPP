#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time : 2022/8/14 12:31
# @Author : Jin Echo
# @File : exp_preprocessing.py

import os, sys
import time
from itertools import product
import multiprocessing as mp
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.common_utils import Logger, set_seed, write_xls, tell_equal
from utils.get_backward_min_features import get_backward_min_features
from environment.graph import generate_a_barabasi_albert_graph_with_k_features
from utils.rcespp_utils import aggregate_constraints
from environment.get_constraint import get_constraint
from algorithm.normalize_dock import normalize_dock
from algorithm.agent import PulseAgent


def parse_args():
    parser = argparse.ArgumentParser("Preprocessing and two-stage algorithm experiments for network optimization")

    # parser.add_argument("--epsilon", type=float, default=1e-5, help="")
    parser.add_argument("--seed", type=int, default=138001, help="initial seed of the random number")  # 138001
    parser.add_argument("--exp-name", type=str, default="ts_again_",
                        help="name of the experiment")  # pre_qhat3, ts_huawei_middle_all_neighbors2dict
    parser.add_argument("--xls-name", type=str, default="ts_again_")  # excel table name: pre_qhat1
    parser.add_argument("--save-dir", type=str, default="./result/paper3/",
                        help="directory in which results should be saved")
    parser.add_argument("--log-dir", type=str, default="./result/paper3/",
                        help="directory in which log files should be saved")
    parser.add_argument("--log-flag", type=str, default=True, help="whether to log")

    return parser.parse_args()


def one_cpu(seed, param_graph, node_sd, **kwargs):
    """
    generate an instance with a fixed random seed, then solve it.
    :param seed:
    :param param_graph:  (num_nodes, num_edges, max_num_nodes, K)
    :param node_sd: (idx_start, idx_end)
    :param kwargs: query: float [SourceID, DestinationID, MinDelay, MaxDelay, WorkOptCost]
    """
    # pid = os.getpid()
    # print("seed{}, pid{}  start...".format(seed, pid))
    # t_start = time.time()
    set_seed(seed)
    num_nodes = param_graph[0]
    if 'adj' in kwargs and 'query' in kwargs:
        adj = kwargs['adj']
        query = kwargs['query']
        constraint_boundary = np.expand_dims(query[2: 4], axis=0)   # [k-1, 2]
        t_generate = 0
    else:
        t0 = time.time()
        adj = generate_a_barabasi_albert_graph_with_k_features(*param_graph)
        adj = adj.squeeze(0)
        constraint_boundary = get_constraint(adj, 0, num_nodes - 1, **kwargs)
        t_generate = time.time() - t0

    # pure pulse algorithm
    adj_, constraint_boundary_ = aggregate_constraints(adj, constraint_boundary)
    t0 = time.time()
    index_end = node_sd[1]
    back_min_cost = get_backward_min_features(adj_[:, :num_nodes], index_end)
    back_min_resource = get_backward_min_features(adj_[:, num_nodes:], index_end)
    # back_max_resource = get_backward_max_features(adj_[:, num_nodes:], index_end, num_nodes-1)
    pulse_agent = PulseAgent(num_nodes, *node_sd)
    cost_pulse_c, _ = pulse_agent.get_best_solution(adj_,
                                                    constraint_boundary_,
                                                    back_min_cost, back_min_resource,
                                                    # back_max_resource,
                                                    c_min=np.inf, mode='dfs')
    # cost_pulse_c = np.inf
    t_pulse_c = time.time() - t0

    # LB-first pulse
    adj_, constraint_boundary_ = aggregate_constraints(adj, constraint_boundary)
    t0 = time.time()
    index_end = node_sd[1]
    back_min_cost = get_backward_min_features(adj_[:, :num_nodes], index_end)
    back_min_resource = get_backward_min_features(adj_[:, num_nodes:], index_end)
    # back_max_resource = get_backward_max_features(adj_[:, num_nodes:], index_end, num_nodes - 1)
    pulse_agent = PulseAgent(num_nodes, *node_sd)
    cost_lbpulse_c, _ = pulse_agent.get_best_solution(adj_,
                                                      constraint_boundary_,
                                                      back_min_cost, back_min_resource,
                                                      # back_max_resource,
                                                      c_min=np.inf, mode='lb')
    # cost_lbpulse_c = np.inf
    t_lbpulse_c = time.time() - t0

    # preprocessing
    t0 = time.time()
    adj_normalized, cost_min, path_best, mask = normalize_dock(adj, constraint_boundary, *node_sd)
    # adj_normalized, cost_min, mask = adj, np.inf, []
    t_preprocess = time.time() - t0

    isdone = 1 if len(mask) == num_nodes - 2 else 0

    if not isdone:
        # number of nodes / edges in the reduced network
        n_edges = np.sum(adj_normalized[:, :num_nodes] != np.inf)
        n_nodes = num_nodes - len(mask)

        # pulse or lb-first pulse solve the reduced network
        adj_normalized = np.delete(adj_normalized, mask, axis=0)  # delete rows
        for i in range(param_graph[3]-1, -1, -1):  # 4,3,2,1,0
            adj_normalized = np.delete(adj_normalized, mask + i * num_nodes, axis=1)  # delete columns
        index_end_ = node_sd[1] - sum(mask < index_end)
        index_start_ = node_sd[0] - sum(mask < node_sd[0])

        adj_normalized, constraint_boundary = aggregate_constraints(adj_normalized, constraint_boundary)
        back_min_cost = get_backward_min_features(adj_normalized[:, :n_nodes], index_end_)
        back_min_resource = get_backward_min_features(adj_normalized[:, n_nodes:], index_end_)
        # back_max_resource = get_backward_max_features(adj_normalized[:, n_nodes:], index_end_, n_nodes - 1)
        pulse_agent = PulseAgent(n_nodes, index_start=index_start_, index_end=index_end_)
        t0 = time.time()
        cost_lbpulse_l, _ = pulse_agent.get_best_solution(adj_normalized,
                                                          constraint_boundary,
                                                          back_min_cost, back_min_resource,
                                                          # back_max_resource,
                                                          c_min=cost_min, mode='lb')
        # cost_lbpulse_l = np.inf
        t_lbpulse_l = time.time() - t0
    else:
        n_edges, n_nodes = -np.inf, -np.inf  # redundant
        cost_lbpulse_l = np.inf
        t_lbpulse_l = 0

    # assert tell_equal(cost_pulse_c, np.min((cost_lbpulse_l, cost_min))), \
    #     "idx:{}, cost_pulse_c:{}, cost_lbpulse_l:{}, cost_min:{}".format(seed, cost_pulse_c, cost_lbpulse_l, cost_min)
    # assert tell_equal(cost_lbpulse_c, cost_pulse_c), \
    #     "idx:{}, cost_lbpulse_c:{}, cost_pulse_c:{}".format(seed, cost_lbpulse_c, cost_pulse_c)

    return (
        t_generate, t_pulse_c, cost_pulse_c,
        t_lbpulse_c, cost_lbpulse_c,
        t_preprocess, t_lbpulse_l, isdone, cost_min, cost_lbpulse_l, n_nodes, n_edges)


def error_call(e):
    print(e)
    print('callback_error!')


def test_simulator_mp(arglist):
    """
    simulation data
    :param arglist:
    :return:
    """
    print("pid: ", os.getpid())
    t_start = time.time()
    ratio_edges_batch = [0.6, 0.7, 0.8]  # [0.4, 0.6, 0.8]  [0.6, 0.7, 0.8]
    num_nodes_batch = [100, 500]  # [100, 500, 1000]   [100, 500]
    # q_batch = [0.8, 0.6, 0.4, 0.2, 0.1]
    q_hat_batch = [-0.2, -0.1, 0., 0.1]

    K_batch = [2, 5]
    num_instances = 100
    num_processes = 18  # 18
    print('K_batch:', K_batch)
    print('num_nodes_batch:', num_nodes_batch)
    print('ratio_edges_batch:', ratio_edges_batch)
    # print('q_batch:', q_batch)
    print('q_hat_batch:', q_hat_batch)
    print('num_instances: ', num_instances)
    print('num_processes: ', num_processes)
    cnt = arglist.seed
    results_info = []
    for K, num_nodes, ratio_edges, q_hat in product(K_batch, num_nodes_batch, ratio_edges_batch, q_hat_batch):
        # for K, num_nodes, ratio_edges, q in list(product(K_batch, num_nodes_batch1, ratio_edges_batch1, q_batch)) + \
        #     list(product(K_batch, num_nodes_batch2, ratio_edges_batch2, q_batch)):
        q = q_hat
        t0 = time.time()
        num_edges = int(num_nodes ** ratio_edges)

        node_sd = (0, num_nodes - 1)
        param_graph = (num_nodes, num_edges, num_nodes, K)

        data_wrapped = [[]] * num_instances

        # pool = mp.Pool(mp.cpu_count() // 2)  # 36/2  48/2
        pool = mp.Pool(num_processes)
        for i in range(num_instances):
            # q or q_hat
            data_wrapped[i] = pool.apply_async(one_cpu, (cnt, param_graph, node_sd,), {'q_hat': q, },
                                               error_callback=error_call)
            cnt += 1
        pool.close()
        pool.join()

        # ********************* Summarize the data after multi-process solution and calculate indicators for a parameter combination.
        data = [item.get() for item in data_wrapped]
        data = np.vstack(data)
        assert data.shape == (num_instances, 12)
        t_generate_batch, t_pulse_c_batch, cost_pulse_c_batch, t_lbpulse_c_batch, cost_lbpulse_c_batch,\
        t_preprocessing_batch, t_lbpulse_l_batch, isdone_batch, \
        cost_min_batch, cost_pulse_l_batch, n_nodes_batch, n_edges_batch = data.T

        idx_undone = np.where(isdone_batch < 0.5)[0]
        idx_done = np.where(isdone_batch > 0.5)[0]
        num_done_inf = np.sum(cost_min_batch[idx_done] == np.inf)
        num_done_notinf = len(idx_done) - num_done_inf
        num_init_cost = np.sum(cost_min_batch[idx_undone] != np.inf)

        t_generate, t_preprocessing = np.sum(t_generate_batch), np.sum(t_preprocessing_batch)
        t_lbpulse_l = np.sum(t_lbpulse_l_batch[idx_undone])  # if index_done is empty, then t_pulse_l=0
        t_pulse_c = np.sum(t_pulse_c_batch)
        t_lbpulse_c = np.sum(t_lbpulse_c_batch)
        # 几何平均geometric mean 与 算术平均arithmetic mean
        # 几何平均加速比会有问题，平均时间two-stage更小，但几何平均却小于1
        # 1和2的计算方式是不对的，明显偏大，因为对每个实例加速比最小为0，最大无穷，若有一个为无穷
        # 3的计算方法和pulse原文一致，直接就是平均时间之比。但pulse原文也是不得已而为之,它没有对比算法每个实例的时间
        # 其双向pulse算法的论文，是用来几何和算术
        # 2008Carlyle的table4 所用的加速比也是基于所有实例的平均时间而不是每个实例都算
        speedup1 = np.cumprod((t_pulse_c_batch / (t_preprocessing_batch + t_lbpulse_l_batch)) ** (1 / num_instances))[-1]
        speedup2 = np.mean(t_pulse_c_batch / (t_preprocessing_batch + t_lbpulse_l_batch))
        speedup3 = t_pulse_c / (t_preprocessing + t_lbpulse_l)
        speedup4 = t_pulse_c / t_lbpulse_c
        # **For the reduced networks**, calculate the percentage of nodes and edges that are reduced by preprocessing.
        if len(idx_undone):
            ratio_nodes_removed_avg = (num_nodes - np.mean(n_nodes_batch[idx_undone])) / (num_nodes - 2)
            n_edges_before = (num_nodes - num_edges) * num_edges * 2
            ratio_edges_removed_avg = (n_edges_before - np.mean(n_edges_batch[idx_undone])) / n_edges_before

        # print info
        res = []
        print('*' * 100)
        print('num_instances:{}  K:{}  num_nodes:{}  num_edges:{}({:.2f})  q_hat:{:.2f}  seed:{}-{}'
              .format(num_instances, K, num_nodes, num_edges, ratio_edges, q, cnt - num_instances, cnt - 1))
        # res.append(str(K) + '-' + str(ratio_edges) + '-' + str(num_nodes) + '-' + str(q))  # 0. parameters
        res.append(K)
        res.append(num_nodes)
        res.append(ratio_edges)
        res.append((num_nodes-num_edges)*num_edges*2)
        res.append(q)  # 0,1,2,3,4
        print('预处理提前终止算法的实例数: {}(inf) + {}(not inf) = {}/{} = {:.4f}'
              .format(num_done_inf, num_done_notinf, len(idx_done), num_instances,
                      len(idx_done) / num_instances))
        res.append(len(idx_done))  # 5. 结束的拓扑数

        if len(idx_undone):
            print('节点数: {} -> {:.2f}  {:.4f}'.format(num_nodes, np.mean(n_nodes_batch[idx_undone]),
                                                     ratio_nodes_removed_avg))
            print('边数: {} -> {:.2f}  {:.4f}'.format(n_edges_before, np.mean(n_edges_batch[idx_undone]),
                                                    ratio_edges_removed_avg))

            num_remain_better = np.sum(cost_pulse_l_batch[idx_undone] < cost_min_batch[idx_undone] - 1e-5)
            print('第一阶段未结束的拓扑，精确算法比cost_min更优：{}/{}'.
                  format(num_remain_better, len(idx_undone)))
            num_remain_get = np.sum(cost_pulse_l_batch[idx_undone] != np.inf)
            print('剩余拓扑存在可行解的比例: {}/{}={:.4f}'
                  .format(num_remain_get, len(idx_undone), num_remain_get / len(idx_undone)))
            # 预处理为剩余拓扑提供了初始可行解
            print('预处理为剩余拓扑提供了初始解: {}/{}'.format(num_init_cost, len(idx_undone)))
            print('generate data: {:.2f}m  pulse_c: {:.2f}m  lb_pulse_c: {:.2f}m  preprocessing: {:.2f}m  lb_pulse_l: {:.2f}m'
                  .format(t_generate / 60, t_pulse_c / 60,  t_lbpulse_c / 60, t_preprocessing / 60, t_lbpulse_l / 60))
            print('                 avg  pulse_c: {:.4f}s  lb_pulse_c: {:.4f}s  preprocessing: {:.4f}s  lbpulse_l: {:.4f}s  total: {:.4f}s'
                  .format(t_pulse_c / num_instances, t_lbpulse_c / num_instances, t_preprocessing / num_instances, t_lbpulse_l / len(idx_undone),
                          (t_preprocessing + t_lbpulse_l) / num_instances))
            print('speed up: geometric:{:.4f}  arithmetic:{:.4f}  true: {:.4f}\nspeed up2: {:.4f}'
                  .format(speedup1, speedup2, speedup3, speedup4))

            res.append(str(num_remain_get) + '/' + str(len(idx_undone)))  # 6. 剩余拓扑中存在可行解的实例数
            res.append(str(num_remain_better) + '/' + str(len(idx_undone)))  # 7. 剩余拓扑中存在更优解的实例数
            res.append(format(ratio_nodes_removed_avg * 100, '.2f'))  # 8. 剩余拓扑中被去除的点的比例
            # res.append(format(ratio_edges_removed_avg*100, '.2f'))  # 9. 剩余拓扑中被去除的点的比例
            res.append(format(ratio_nodes_removed_avg * 100, '.2f') + ' (' + format(ratio_edges_removed_avg * 100,
                                                                                    '.2f') + ')')
            res.append(format(str(num_init_cost) + '/' + str(len(idx_undone))))  # 10. 预处理为剩余拓扑提供初始可行解
            res.append(format(t_pulse_c / num_instances, '.2f'))  # 11. 纯pulse
            res.append(format(t_lbpulse_c / num_instances, '.2f'))  # 12. 纯 lb-first pulse
            res.append(format(t_preprocessing / num_instances, '.2f'))  # 13. 预处理平均时间
            res.append(format(t_lbpulse_l / len(idx_undone), '.3f'))  # len(idx_undone), '.4f'))  # 14. pulse求剩余拓扑的平均时间 ×
            res.append(format((t_preprocessing + t_lbpulse_l) / num_instances, '.2f'))  # 15. 两阶段平均时间
        else:
            print(
                'generate data: {:.2f}m  pulse_c: {:.2f}m  lb_pulse_c: {:.2f}m  preprocessing: {:.2f}m  pulse_l: -m'.format(t_generate / 60,
                                                                                                       t_pulse_c / 60, t_lbpulse_c / 60,
                                                                                                       t_preprocessing / 60))
            print('                 avg  pulse_c: {:.4f}s  lb_pulse_c: {:.4f}s  preprocessing: {:.4f}s  pulse_l: -s  total: {:.4f}s'.format(
                t_pulse_c / num_instances, t_lbpulse_c / num_instances, t_preprocessing / num_instances, t_preprocessing / num_instances))
            print('speed up1: geometric:{:.4f}  arithmetic:{:.4f}  true: {:.4f}\nspeed up2: {:.4f}'
                  .format(speedup1, speedup2, speedup3, speedup4))
            res += ['-'] * 10
            res[11] = format(t_pulse_c / num_instances, '.2f')
            res[12] = format(t_lbpulse_c / num_instances, '.2f')
            res[13] = format(t_preprocessing / num_instances, '.2f')  # 13. 预处理平均时间
            res[15] = res[13]
        # res.append(format(speedup1, '.2f'))
        # res.append(format(speedup2, '.2f'))
        res.append(format(speedup3, '.2f'))  # 16 two-stage
        res.append(format(speedup4, '.2f'))  # 17 改进pulse的加速比
        results_info.append(res)
        print('actual time: {:.2f}m'.format((time.time() - t0) / 60))
    print('exp time: {:.2f}m'.format((time.time() - t_start) / 60))
    write_xls(arglist.save_dir + arglist.xls_name + '.xls', results_info)


if __name__ == '__main__':
    arglist = parse_args()

    if arglist.log_flag:
        log_dir = arglist.log_dir + arglist.exp_name + '.log'
        sys.stdout = Logger(log_dir, sys.stdout)
        sys.stderr = Logger(log_dir, sys.stderr)

    print('*' * 100)
    print('*' * 100)
    print('pid: ', os.getpid())
    print("init seed: ", arglist.seed)
    test_simulator_mp(arglist)