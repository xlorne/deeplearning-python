#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File Name: cnn_ssd_test.py
# Description:
# Create Time: 2020-11-03 09:37
# Author: lorne

import d2lzh as d2l
from mxnet import autograd, contrib, gluon, image, init, nd
from mxnet.gluon import loss as gloss, nn
import time
import os
from matplotlib import pyplot as plt
from mxnet.gluon import data as gdata
from IPython import display

from mxnet.gluon import utils as gutils

def load_data_pikachu(batch_size, edge_size=64):  # edge_size：输出图像的宽和高
    data_dir = '/mnt/data/pikachu/'
    train_iter = image.ImageDetIter(
        path_imgrec=os.path.join(data_dir, 'train.rec'),
        path_imgidx=os.path.join(data_dir, 'train.idx'),
        batch_size=batch_size,
        data_shape=(3, edge_size, edge_size),  # 输出图像的形状
        shuffle=False,  # 以随机顺序读取数据集
        rand_crop=1,    # 随机裁剪的概率为100%
        min_object_covered=0.9, #物体出现的最小区域
        max_attempts=200    #最大的尝试次数
    )
    return train_iter

batch_size, edge_size = 1, 64
train_iter = load_data_pikachu(batch_size, edge_size)
print(train_iter.data_shape,train_iter.batch_size)

# 本函数已保存在d2lzh包中方便以后使用
def show_image(images):
    display.set_matplotlib_formats('png')
    # 这里的_表示我们忽略（不使用）的变量
    # _, figs = plt.subplots(1, len(images), figsize=(32, 32))
    # for f, img in zip(figs, images):
    plt.imshow(images.reshape(edge_size, edge_size).asnumpy())
    # plt.axes.get_xaxis().set_visible(False)
    # plt.axes.get_yaxis().set_visible(False)
    plt.show()

ctx = d2l.try_gpu()

# 从头读取数据
# train_iter.reset()
batch = train_iter.next()
X = batch.data[0].as_in_context(ctx)
show_image(X)
# for batch in train_iter:
#     X = batch.data[0].as_in_context(ctx)
#     show_image(X)

