#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File Name: lenet_test.py
# Description: LeNet Learning
# http://zh.d2l.ai/chapter_convolutional-neural-networks/lenet.html
# Create Time: 2020-10-11 16:04
# Author: lorne

import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import time


net = nn.Sequential()
# LeNet网络的定义
net.add(nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        # Dense会默认将(批量大小, 通道, 高, 宽)形状的输入转换成
        # (批量大小, 通道 * 高 * 宽)形状的输入
        nn.Dense(120, activation='sigmoid'),
        nn.Dense(84, activation='sigmoid'),
        nn.Dense(10))

# 打印每层网络结构数据
X = nd.random.uniform(shape=(1, 1, 28, 28))
net.initialize()
for layer in net:
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)

# 加载fashion_mnist数据
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

# 判断是否支持GPU计算，不能则使用CPU。GPU需要N卡安装CUDU
def try_gpu():
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx

ctx = try_gpu()

# 函数训练过程与mlp_neuralnet_test一致
def train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx,num_epochs):
    print('training on', ctx)
    loss = gloss.SoftmaxCrossEntropyLoss()
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            # 参数更新
            trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
                 time.time() - start))


# 测试数据
def evaluate_accuracy(data_iter, net, ctx):
    acc_sum, n = nd.array([0], ctx=ctx), 0
    for X, y in data_iter:
        # 如果ctx代表GPU及相应的显存，将数据复制到显存上
        X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum()
        n += y.size
    return acc_sum.asscalar() / n

lr, num_epochs = 0.9, 5
# 初始化函数
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
# 梯度下降算法
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})

# 开始训练
train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)


# 数据预测（随机测试）
X = nd.random.uniform(shape=(1, 1, 28, 28))
prediction = net(X).argmax(axis=1)
print("prediction:",prediction)