import thing4
import dldata.stimulus_sets.hvm as hvm
from collections import OrderedDict


preproc = OrderedDict([(u'normalize', False), (u'dtype', u'float32'), (u'resize_to', [256, 256, 3]), (u'mode', u'RGB'), (u'crop', None), (u'mask', None)])

dataset = hvm.HvMWithDiscfade()
imgs = dataset.get_images(preproc=preproc)
pf = '/home/ubuntu/new/caffe/examples/cifar10/caffenet_rand.prototxt'
#pf = '/home/ubuntu/new/caffe/examples/cifar10/caffenet_mean.prototxt'
sf = '/home/ubuntu/new/caffe/examples/cifar10/bvlc_reference_caffenet.caffemodel'

#im0 = 255*imgs[0].transpose((2, 0, 1)).reshape((1, 3, 256, 256))
#im1 = 255*imgs[-1].transpose((2, 0, 1)).reshape((1, 3, 256, 256))
meta = dataset.meta
inds = (meta['category'] == 'Cars').nonzero()[0]
ims = [255*imgs[i][14:-15][:, 14:-15].transpose((2, 0, 1)).reshape((1, 3, 227, 227)) for i in inds]
import numpy as np
from PIL import Image
im = np.asarray(Image.open('/home/ubuntu/imgres.jpg').resize((256, 256), resample=Image.ANTIALIAS)).transpose(2, 0, 1).reshape((1, 3, 256, 256))[:, :, 14:-15][:, :, :, 14:-15]

#r, targets = thing4.main(pf, sf, '/home/ubuntu/test_y4a.npy',  [([ims[i] for i in range(640)], [('fc7', 100, 'ss'), ('fc6', 100, 'ss'), ('fc8', 100, 'ss')])], 'data', crop=(14, -15, 14, -15))

#r, targets = thing4.main(pf, sf, '/home/ubuntu/test_y3b.npy',  [(ims[-1], [('conv1', 100, 'corr'), ('conv2', 100, 'corr', ), ('conv5', 100, 'corr'), ('pool5', 100, 'ss')])], 'data', crop=(14, -15, 14, -15))

#r, targets = thing4.main(pf, sf, '/home/ubuntu/test_X.npy',  [([ims[i] for i in range(1)], [('conv5', 100, 'corr'), ('pool5', 100, 'ss'), ('fc7', 100, 'ss'), ('fc6', 100, 'ss'), ('fc8', 100, 'ss')])], 'data', crop=(14, -15, 14, -15))

#r, targets = thing4.main(pf, sf, '/home/ubuntu/test_X.npy',  [([ims[i] for i in range(640)], [('fc8', 10000, 'ss')])], 'data', crop=(14, -15, 14, -15), use_bounds=True)

#inds = (meta['category'] == 'Faces').nonzero()[0]
#ims = [255*imgs[i][14:-15][:, 14:-15].transpose((2, 0, 1)).reshape((1, 3, 227, 227)) for i in inds]
im = np.asarray(Image.open('/home/ubuntu/imgres.jpg').resize((256, 256), resample=Image.ANTIALIAS)).transpose(2, 0, 1).reshape((1, 3, 256, 256))[:, :, 14:-15][:, :, :, 14:-15]

im1 = np.asarray(Image.open('/home/ubuntu/imgres-1.jpg').resize((256, 256), resample=Image.ANTIALIAS)).transpose(2, 0, 1).reshape((1, 3, 256, 256))[:, :, 14:-15][:, :,:, 14:-15]

var = np.zeros((1000,)).astype(np.float32)
#var = np.zeros((4096,)).astype(np.float32)
var[0] = 1

#r, targets = thing4.main(pf, sf, '/home/ubuntu/test_something4.npy',  [([im], [('pool1', 100, 'corr')]), (None, [('fc8', 1, ('max', var))])], 'data', use_bounds=True, crop=(14, -15, 14, -15))

r, targets = thing4.main(pf, sf, '/home/ubuntu/genstim10/genstim.npy', 
                         [(None, [('data', .01, 'smooth'), 
                                  ('fc8', (10000, var), 'softmax')]),
                           ([im], [('pool1', 20, 'corr')])
                        ],
                     data_layer = 'data', use_bounds=True, crop=(14, -15, 14, -15), 
                     start_normal=True, seed=1, 
                     start_im = im,
                     save_dir='/home/ubuntu/genstim10/things', save_freq=100)
