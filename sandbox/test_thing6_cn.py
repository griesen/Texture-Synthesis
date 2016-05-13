import numpy as np

import thing4
reload(thing4)
import dldata.stimulus_sets.hvm as hvm
from collections import OrderedDict



pf = '/home/ubuntu/new/caffe/examples/cifar10/caffenet_rand.prototxt'
sf = '/home/ubuntu/new/caffe/examples/cifar10/bvlc_reference_caffenet.caffemodel'

#im0 = 255*imgs[0].transpose((2, 0, 1)).reshape((1, 3, 256, 256))
#im1 = 255*imgs[-1].transpose((2, 0, 1)).reshape((1, 3, 256, 256))


var = np.array([-1] + [0]*4095).astype(np.float32)

r = thing4.main(pf, sf, '/home/ubuntu/test_O3_bounds_s.npy', targets0=[(None,[('fc6', 1, ('max', var))])], crop=(14, -15, 14, -15), use_bounds = True,seed=1)
