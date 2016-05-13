import thing4
import dldata.stimulus_sets.hvm as hvm
from collections import OrderedDict


preproc = OrderedDict([(u'normalize', False), (u'dtype', u'float32'), (u'resize_to', [256, 256, 3]), (u'mode', u'RGB'), (u'crop', None), (u'mask', None)])

dataset = hvm.HvMWithDiscfade()
imgs = dataset.get_images(preproc=preproc)
pf = '/home/ubuntu/new/caffe/examples/cifar10/caffenet_rand.prototxt'
sf = '/home/ubuntu/new/caffe/examples/cifar10/bvlc_reference_caffenet.caffemodel'

#im0 = 255*imgs[0].transpose((2, 0, 1)).reshape((1, 3, 256, 256))
#im1 = 255*imgs[-1].transpose((2, 0, 1)).reshape((1, 3, 256, 256))
meta = dataset.meta
inds = (meta['category'] == 'Cars').nonzero()[0]
ims = [255*imgs[i][14:-15][:, 14:-15].transpose((2, 0, 1)).reshape((1, 3, 227, 227)) for i in inds]
import numpy as np
from PIL import Image
im = np.asarray(Image.open('/home/ubuntu/imgres.jpg').resize((256, 256), resample=Image.ANTIALIAS)).transpose(2, 0, 1).reshape((1, 3, 256, 256))[:, :, 14:-15][:, :, :, 14:-15]

im1 = np.asarray(Image.open('/home/ubuntu/test.png').convert('RGB').resize((256, 256), resample=Image.ANTIALIAS)).transpose(2, 0, 1).reshape((1, 3, 256, 256))[:, :, 14:-15][:, :, :, 14:-15]


im2 = np.asarray(Image.open('/home/ubuntu/imgres-2.png').convert('RGB').resize((256, 256), resample=Image.ANTIALIAS)).transpose(2, 0, 1).reshape((1, 3, 256, 256))[:, :, 14:-15][:, :, :, 14:-15]


#r, targets = thing4.main(pf, sf, '/home/ubuntu/test_y4a.npy',  [([ims[i] for i in range(640)], [('fc7', 100, 'ss'), ('fc6', 100, 'ss'), ('fc8', 100, 'ss')])], 'data', crop=(14, -15, 14, -15))

#r, targets = thing4.main(pf, sf, '/home/ubuntu/test_rocks_cn_pool.npy',  [(im1, [('conv1', 100, 'corr'), ('pool1', 100, 'corr'), ('conv2', 100, 'corr', ), ('pool2', 100, 'corr'),  ('conv3', 100, 'corr'), ('conv4', 100, 'corr'), ('conv5', 100, 'corr'), ('pool5', 100, 'corr')])], 'data', crop=(14, -15, 14, -15))


r, targets = thing4.main(pf, sf, '/home/ubuntu/test_fruits_cn_all.npy',  [(im2, [('conv1', 1000, 'corr'), ('pool1', 1000, 'corr'), ('conv2', 1000, 'corr'), ('pool2', 1000, 'corr'), ('conv5', 1000, 'corr'), ('pool5', 1000, 'corr'), ('fc6', 1000, 'ss'), ('fc7', 1000, 'ss')])], use_bounds=True, data_layer='data', crop=(14, -15, 14, -15))
 

#r, targets = thing4.main(pf, sf, '/home/ubuntu/test_y3d.npy',  [(ims[0], [('conv1', 100, 'corr'), ('conv2', 100, 'corr', ), ('conv3', 100, 'corr'), ('conv4', 100, 'corr'), ('conv5', 100, 'corr')])], 'data', crop=(14, -15, 14, -15), use_bounds=True, start_normal=False)
