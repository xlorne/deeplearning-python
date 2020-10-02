#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File Name: fashion_mnist.py
# Description: http://zh.d2l.ai/chapter_deep-learning-basics/fashion-mnist.html
# Create Time: 2020-10-02 19:26
# Author: lorne

from matplotlib import pyplot as plt
from mxnet.gluon import data as gdata
from IPython import display

mnist_train = gdata.vision.FashionMNIST(train=True)
mnist_test = gdata.vision.FashionMNIST(train=False)
print(len(mnist_train),len(mnist_test))


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


# 本函数已保存在d2lzh包中方便以后使用
def show_fashion_mnist(images, labels):
    display.set_matplotlib_formats('svg')
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

if __name__ == '__main__':
    X, y = mnist_train[0:9]
    show_fashion_mnist(X, get_fashion_mnist_labels(y))
