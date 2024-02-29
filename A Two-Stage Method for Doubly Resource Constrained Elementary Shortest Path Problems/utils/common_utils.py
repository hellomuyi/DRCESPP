#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# @Time : 2022/1/21 22:06
# @Author : Jin Echo
# @File : common_utils.py
import os, sys

import numpy as np
import random
# import torch
import xlwt


__all__ = [
    'Logger',
    'set_seed',
    'tell_equal',
    'write_xls',
    'get_trainable_para_num',
]


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()
        self.log.flush()

    def flush(self):
        pass


def set_seed(seed=1, is_scatter=False):
    """
    当使用PyTorch底层的scatter时，(例如, PyG底层基于 PyTorch scatter_add实现)，是无法通过固定随机种子复现的，
    scatter operations are non-deterministic。此时固定种子(y也即调用此函数没有意义)，而我们的图神经网络的实现基于PyG(PyTorch Geometric)
    其原文(https://arxiv.org/abs/1903.02428)明确指出即使固定了种子训练时也无法复现，但推理inference时可以：
    In addition, it should be noted that scatter operations are non-deterministic by nature on the GPU.
    Although we did not observe any deviations for inference, training results can vary across the same manual seeds
    ---
    torch_scatter.scatter的描述
    This operation is implemented via atomic operations on the GPU and is
    therefore **non-deterministic** since the order of parallel operations
    to the same value is undetermined.
    For floating-point variables, this results in a source of variance in
    the result.
    ---
    :param seed:
    :param is_scatter: 故当is_scatter=True时，不设置会影响时间性能的torch.backends.cudnn.deterministic和torch.backends.cudnn.benchmark，
                       (事实上，本应不设置任何事情，种子没有意义)
    """
    random.seed(seed)
    np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)    # 为了禁止hash随机化
    # torch.manual_seed(seed)                     # 为CPU
    # torch.cuda.manual_seed(seed)                # 为当前GPU
    # torch.cuda.manual_seed_all(seed)            # 为所有GPU
    # # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"       # for CUDA 10.1
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # for CUDA >= 10.2
    # if not is_scatter:
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False


def tell_equal(x, y, epsilon=1e-5):
    """
    等价于np.allclose(x, y, rtol=0, atol=epsilon, equal_nan=False)
    判断两浮点数数组元素是否都相等，含inf
    :param x: ndarray (n,)
    :param y: ndarray (n,)
    :param epsilon: float 判断浮点数相等的误差
    :return: bool
    """
    # delta_xy = x - y  # np.inf - np.inf == nan  会出现warning
    # delta_xy = delta_xy[~np.isnan(delta_xy)]
    # return (np.abs(delta_xy) < epsilon).all()
    return np.allclose(x, y, rtol=0, atol=epsilon, equal_nan=False)


# def cal_number_equal(x, y):
#     """
#     计算相等元素的数量
#     :param x: (n, )
#     :param y: (n, )
#     :return:
#     """
#     a = copy.copy(x)
#     b = copy.copy(y)
#     assert len(a) == len(b)
#     # a_b = a - b
#     # a_b[np.isnan(a_b)] = 0
#     # return np.sum(abs(a_b) < 1e-5)
#     MAX = 2 ** 31 - 1
#     a[a == np.inf] = MAX
#     a[a == -np.inf] = -MAX
#     b[b == np.inf] = MAX
#     b[b == -np.inf] = -MAX
#     idx_equal = np.where(abs(a - b) <= 1e-5)[0]
#     return len(idx_equal), idx_equal   # np.sum(abs(a - b) <= 1e-5)


# 获得可训练参数的数量
def get_trainable_para_num(model):
    num = 0
    for para in model.parameters():
        if para.requires_grad == True:
            num += para.nelement()
    # print(f"trainable paras number: {num}")
    # num = sum(p.numel() for p in model.parameters())
    return num


def timer(name, end='\n'):
    print()
    pass


def write_xls(filename, data, column_name=None):
    """

    :param filename:
    :param data: Two levels of nested lists (or ndarray with ndim=2)
    :param column_name:
    """
    assert filename[-4:] == '.xls', 'the file extension must be .xls'
    assert isinstance(data[0], list)  # or isinstance(data[0], np.ndarray)
    if os.path.exists(filename):
        os.remove(filename)

    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('Sheet 1', cell_overwrite_ok=True)

    idx_row = 0
    if column_name is not None:
        for i in range(len(column_name)):
            sheet.write(idx_row, i, column_name[i])
        idx_row += 1

    for i in range(len(data)):
        for j in range(len(data[i])):
            sheet.write(idx_row, j, data[i][j])
        idx_row += 1

    book.save(filename)


