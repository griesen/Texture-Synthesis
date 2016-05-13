import thing4
import dldata.stimulus_sets.hvm as hvm
from collections import OrderedDict


pf = '/home/ubuntu/new/caffe/examples/cifar10/cifar10_quick_train_test_rand.prototxt'
sf  = '/home/ubuntu/new/caffe/examples/cifar10/cifar10_quick_iter_5000.caffemodel.h5'

import numpy as np
v = np.load('/home/ubuntu/new/caffe/data_target.npy')

im = v[2]

#r, targets = thing4.main(pf, sf, '/home/ubuntu/test_xc.npy',  [(im, [('conv1', 200, 'corr'), ('conv2', 200, 'corr'), ('c#onv3', 200, 'corr')])], 'data', mean=np.zeros_like(v[0]))

r, targets = thing4.main(pf, sf, '/home/ubuntu/test_yc1.npy',  [(im, [('pool3', 100, 'corr')])], 'data', mean=np.zeros_like(v[0]))
