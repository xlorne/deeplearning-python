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


from gluoncv import data, utils
from matplotlib import pyplot as plt


train_dataset = data.VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
val_dataset = data.VOCDetection(splits=[(2007, 'test')])

