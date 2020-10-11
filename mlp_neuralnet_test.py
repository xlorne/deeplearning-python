#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File Name: mlp_neuralnet_test.py
# Description: 深度神经网络 fashion数据测试
# Create Time: 2020-10-11 11:44
# Author: lorne

import d2lzh as d2l
from mxnet import autograd
from mxnet import nd
from mxnet.gluon import loss as gloss

# 批量数据集大小
batch_size = 256
# 训练次数
num_epochs=15
# 学习率
lr=0.5
# 输入参数 fashion图片的大小为:28x28
num_inputs= 784
# 输出参数
num_outputs= 10
# 隐藏层参数
num_hiddens =256

# 数据集
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 训练参数
W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
b1 = nd.ones(num_hiddens)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
b2 = nd.ones(num_outputs)

# 组装成集合
params = [W1, b1, W2, b2]

# 训练函数
def train( train_iter,test_iter,forword, loss, num_epochs, batch_size,
              params=None, lr=None):
    """Train and evaluate a model with CPU."""
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        # 每一批次的数据
        for X, y in train_iter:
            # 记录forward函数的计算过程，然后对params求导
            with autograd.record():
                y_hat = forword(X)
                # 损失函数计算
                l = loss(y_hat, y)
            # 反向传播通过自动求导完成各层参数的导数
            l.backward()
            # 更新参数
            sgd(params, lr, batch_size)

            y = y.astype('float32')
            # 统计损失函数值
            train_l_sum += l.asscalar()
            # 统计准确预测的数据
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            # 统计记录条数
            n += y.size

        test_acc = evaluate_accuracy(test_iter, forword)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

# 测试训练并记录得分
def evaluate_accuracy(data_iter, forward):
    """Evaluate accuracy of a model on the given data set."""
    acc_sum, n = 0.0, 0
    for features, labels in data_iter:
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (forward(X).argmax(axis=1) == y).sum().asscalar()
            n += y.size
    return acc_sum / n

# 向前传播
def forword(X):
    # reshape 以后X为 256 x 784 ，一张图片的大小为28 x 28 = 784，256是batch_size
    X = X.reshape((-1, num_inputs))
    z1 = nd.dot(X,W1)+b1
    a1 = nd.relu(z1)
    z2 = nd.dot(a1,W2)+b2
    a2 = nd.softmax(z2)
    return a2

# 损失函数
def loss(y_hat,y):
    loss = gloss.SoftmaxCrossEntropyLoss()
    return loss(y_hat,y).sum()

# 更新参数
def sgd(params, lr, batch_size):
    """Mini-batch stochastic gradient descent."""
    for param in params:
        param[:] = param - (lr * param.grad / batch_size)


# 记录参数变量，用于自动求导
for param in params:
    param.attach_grad()

# 开始训练
train(train_iter,test_iter,forword,loss,num_epochs,batch_size,params,lr)