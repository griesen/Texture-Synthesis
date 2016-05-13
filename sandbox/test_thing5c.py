import thing5
import dldata.stimulus_sets.hvm as hvm
from collections import OrderedDict


preproc = OrderedDict([(u'normalize', False), (u'dtype', u'float32'), (u'resize_to', [256, 256, 3]), (u'mode', u'RGB'), (u'crop', None), (u'mask', None)])

pf = '/home/ubuntu/new/caffe/examples/cifar10/roschinet_larger_rand_test.prototxt'
sf = '/home/ubuntu/new/caffe/examples/cifar10/roschinet_larger.caffemodel.h5'
import numpy as np
from PIL import Image
im = np.asarray(Image.open('/home/ubuntu/imgres.jpg').resize((256, 256), resample=Image.ANTIALIAS)).transpose(2, 0, 1).reshape((1, 3, 256, 256)) 
r =  thing5.main(pf, sf, '/home/ubuntu/test_y5.npy', im, None, {'conv1': 1, 'conv2': 1, 'conv3': 1, 'conv4': 1, 'conv5': 1}, 'data')
