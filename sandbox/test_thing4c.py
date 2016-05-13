import thing4b
import dldata.stimulus_sets.hvm as hvm
from collections import OrderedDict


preproc = OrderedDict([(u'normalize', False), (u'dtype', u'float32'), (u'resize_to', [256, 256, 3]), (u'mode', u'RGB'), (u'crop', None), (u'mask', None)])

pf = '/home/ubuntu/new/caffe/examples/cifar10/roschinet_larger_rand_test_mean.prototxt'
sf = '/home/ubuntu/new/caffe/examples/cifar10/roschinet_larger.caffemodel.h5'
import numpy as np
from PIL import Image
im = np.asarray(Image.open('/home/ubuntu/imgres.jpg').resize((256, 256), resample=Image.ANTIALIAS)).transpose(2, 0, 1).reshape((1, 3, 256, 256)) 

#im1 = np.asarray(Image.open('/home/ubuntu/imgres-1.jpg').resize((256, 256), resample=Image.ANTIALIAS)).transpose(2, 0, 1).reshape((1, 3, 256, 256)) 

im1 = np.asarray(Image.open('/home/ubuntu/imgres-2.png').convert('RGB').resize((256, 256), resample=Image.ANTIALIAS)).transpose(2, 0, 1).reshape((1, 3, 256, 256)) 


r, targets = thing4b.main(pf, sf, '/home/ubuntu/test_fruits_bounds.npy', [(im1, [('conv1', 1, 'corr'), ('pool1', 1, 'corr'), ('conv2', 1, 'corr'), ('pool2', 1, 'corr'), ('conv3', 1, 'corr'), ('conv4', 1, 'corr'),  ('conv5', 1, 'corr'), ('pool3', 1, 'corr')])], 'data', seed=2, use_bounds=True)


#r, targets = thing4.main(pf, sf, '/home/ubuntu/test_y2a1_bounds.npy', [(im, [('conv1', 100, 'corr'), ('conv2', 100, 'corr'), ('conv3', 100, 'corr'), ('conv4', 100, 'corr'), ('conv5', 100, 'corr')])], 'data', seed=0, use_bounds=True, start_normal=False)

#r, targets = thing4.main(pf, sf, '/home/ubuntu/test_y2a1_bounds.npy', [(im, [('conv1', 100, 'ss')])], 'data', seed=0, use_bounds=True)
