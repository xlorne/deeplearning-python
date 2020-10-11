#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File Name: cnn_base.py
# Description: 卷积神经网络练习
# 卷积神经网络参数介绍：
# 输入数据 : ci x nh x nw | ci=>channel数,即通道数. nh=>高度 nw=>高度
# 卷积层超参数 kernel bias padding stride, 参与反向传播的是 kernel bias, padding stride 是固定值
# kernel 卷积核，在多输出时形状为 co×ci×kh×kw co=>输出大小 ci=>卷积层的输入channel,kh=>kernel的height,hw=>kernel的weight
# bias DNN中的bias一致
# padding 填充的大小 (1,1) 在高和宽两侧的填充数分别为1和1
# stride 卷积移动的方向(1,1) 左移动一格右移动一格
# http://zh.d2l.ai/chapter_convolutional-neural-networks/conv-layer.html
# http://zh.d2l.ai/chapter_convolutional-neural-networks/padding-and-strides.html
# http://zh.d2l.ai/chapter_convolutional-neural-networks/channels.html
# http://zh.d2l.ai/chapter_convolutional-neural-networks/pooling.html
# Create Time: 2020-10-11 14:16
# Author: lorne

from mxnet import nd
from mxnet.gluon import nn


# 卷积运算过程
def corr2d(X, K):
    h, w = K.shape
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

# 卷积运算测试
X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = nd.array([[0, 1], [2, 3]])
res = corr2d(X, K)
print("corr2d",res)

# 定义卷积运算对象Conv2D
class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=kernel_size)
        self.bias = self.params.get('bias', shape=(1,))


    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()


# 多通道计算
def corr2d_multi_in(X, K):
    # 首先沿着X和K的第0维（通道维）遍历。然后使用*将结果列表变成add_n函数的位置参数
    # （positional argument）来进行相加
    return nd.add_n(*[corr2d(x, k) for x, k in zip(X, K)])


X = nd.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
              [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = nd.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

res = corr2d_multi_in(X, K)
print("corr2d_multi_in",res)

# 多输出通道计算
def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历，每次同输入X做互相关计算。所有结果使用stack函数合并在一起
    return nd.stack(*[corr2d_multi_in(X, k) for k in K])

# Kernel 也是一个多维的数组 co×ci×kh×kw co=>输出大小 ci=>卷积层的输入channel,kh=>kernel的height,hw=>kernel的weight
print("before stack K:",K)
K = nd.stack(K, K + 1, K + 2)
print("after stack K:",K)
print("corr2d_multi_in_out k shape:",K.shape)

res = corr2d_multi_in_out(X, K)
print("corr2d_multi_in_out",res)

# 全连接层的矩阵乘法
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = nd.dot(K, X)  # 全连接层的矩阵乘法
    return Y.reshape((c_o, h, w))


X = nd.random.uniform(shape=(3, 3, 3))
K = nd.random.uniform(shape=(2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

res=(Y1 - Y2).norm().asscalar() < 1e-6
print("1x1卷积与全链接一致:",res)


# 池化层计算
def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = nd.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y


X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
res = pool2d(X, (2, 2))
print("pool2d res:",res)


# 我们先构造一个形状为(1, 1, 4, 4)的输入数据，前两个维度分别是批量和通道。
X = nd.arange(16).reshape((1, 1, 4, 4))
# 默认没有指定stride的时候不移动，第一个框内的最大值即为10
pool2d = nn.MaxPool2D(3)
res = pool2d(X)
print("pool2d test res:",res)

pool2d = nn.MaxPool2D(3, padding=1, strides=2)
res = pool2d(X)
print("pool2d test padding(1,1) strides(2,2) res:",res)

# 多通道
# 在处理多通道输入数据时，池化层对每个输入通道分别池化，而不是像卷积层那样将各通道的输入按通道相加。这意味着池化层的输出通道数与输入通道数相等。
X = nd.concat(X, X + 1, dim=1)
pool2d = nn.MaxPool2D(3, padding=1, strides=2)
res = pool2d(X)
print("mlp pool2d test padding(1,1) strides(2,2) res:",res)