# coding=utf8
# collect some functions and classes which may useful
#####################################################
## import part

# import numpy as np
# import torch

#####################################################
# 将数据集标签按已知未知分别赋予0 and 1, v2 版，相比原版
# 舍弃了for循环
# 变量说明：
# dataset_Y: label 对应的Y数组，这里需要是一维的
# numclass: 已知类类别数，如 =10，表示有10类已知，则
# 大于此的视作未知（要看具体标签从0还是1起数）
import copy
def emgdata_label_tsp_01_v2(dataset_Y,num_knclass):
    emg_label = copy.deepcopy(dataset_Y[:])
    kn_idx = (emg_label < num_knclass)
    kn_idx = kn_idx.squeeze()
    un_idx = (emg_label >= num_knclass)
    un_idx = un_idx.squeeze()
    emg_label[kn_idx] = 1
    emg_label[un_idx] = 0
    return emg_label