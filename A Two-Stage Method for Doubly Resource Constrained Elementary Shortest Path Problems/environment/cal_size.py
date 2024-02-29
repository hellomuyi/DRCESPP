#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time : 2022/9/4 0:31
# @Author : Jin Echo
# @File : cal_size.py

n = [50, 100, 500, 1000, 2000, 5000]
m = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for n_ in n:
    for ratio_edges in m:
        num_e = int(n_**ratio_edges)
        num_edges = (n_-num_e) * num_e
        print('{} {} {} {} {}'.format(n_, ratio_edges, num_e, num_edges, num_edges*2))
    print('*'*20)

