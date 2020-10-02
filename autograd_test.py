#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File Name: autograd_test.py
# Description: http://zh.d2l.ai/chapter_prerequisite/autograd.html
# Create Time: 2020-10-02 17:18
# Author: lorne


from mxnet import autograd, nd


w = nd.random.normal(scale=0.01, shape=(2, 1))
b = nd.random.normal(scale=0.05,shape=(1,1))

print(w)
print(b)

b.attach_grad()
w.attach_grad()

def fx(b,w):
    return 3*b+4*w


with autograd.record():
    z=fx(b,w)
z.backward()

print(b.grad)
print(w.grad)


# b\prime = 6 因为bshape为1，1 在做运算时加了两次固是2x3=6