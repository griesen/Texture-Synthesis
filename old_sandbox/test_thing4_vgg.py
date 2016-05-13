import thing4a
import dldata.stimulus_sets.hvm as hvm
from collections import OrderedDict


preproc = OrderedDict([(u'normalize', False), (u'dtype', u'float32'), (u'resize_to', [256, 256, 3]), (u'mode', u'RGB'), (u'crop', None), (u'mask', None)])

dataset = hvm.HvMWithDiscfade()
imgs = dataset.get_images(preproc=preproc)
pf = '/home/ubuntu/new/caffe/vgg_avg2.prototxt'
sf = '/home/ubuntu/new/caffe/vgg_normalised.caffemodel'

#im0 = 255*imgs[0].transpose((2, 0, 1)).reshape((1, 3, 256, 256))
#im1 = 255*imgs[-1].transpose((2, 0, 1)).reshape((1, 3, 256, 256))
meta = dataset.meta
inds = (meta['category'] == 'Cars').nonzero()[0]
ims = [255*imgs[i][14:-15][:, 14:-15].transpose((2, 0, 1)).reshape((1, 3, 227, 227)) for i in inds]
import numpy as np
from PIL import Image
im = np.asarray(Image.open('/home/ubuntu/imgres.jpg').resize((256, 256), resample=Image.ANTIALIAS)).transpose(2, 0, 1).reshape((1, 3, 256, 256))[:, :, 16:-16][:, :, :, 16:-16]

im1 = np.asarray(Image.open('/home/ubuntu/test.png').convert('RGB').resize((256, 256), resample=Image.ANTIALIAS)).transpose(2, 0, 1).reshape((1, 3, 256, 256))[:, :, 16:-16][:, :, :, 16:-16]

im2 = np.asarray(Image.open('/home/ubuntu/imgres-2.png').convert('RGB').resize((256, 256), resample=Image.ANTIALIAS)).transpose(2, 0, 1).reshape((1, 3, 256, 256))[:, :, 16:-16][:, :, :, 16:-16]


print (im1.shape)

#r, targets = thing4.main(pf, sf, '/home/ubuntu/test_y4a.npy',  [([ims[i] for i in range(640)], [('fc7', 100, 'ss'), ('fc6', 100, 'ss'), ('fc8', 100, 'ss')])], 'data', crop=(14, -15, 14, -15))

#r, targets = thing4.main(pf, sf, '/home/ubuntu/test_rocks_cn_pool.npy',  [(im1, [('conv1', 100, 'corr'), ('pool1', 100, 'corr'), ('conv2', 100, 'corr', ), ('pool2', 100, 'corr'),  ('conv3', 100, 'corr'), ('conv4', 100, 'corr'), ('conv5', 100, 'corr'), ('pool5', 100, 'corr')])], 'data', crop=(14, -15, 14, -15))

r, targets = thing4a.main(pf, sf, '/home/ubuntu/test_vgg_pool4_fruits.npy',  
                                                 [(im2, [('conv1_1', 10, 'corr'), 
                                                         ('conv1_2', 10, 'corr'),
                                                         ('pool1', 10, 'corr'),
                                                         ('conv2_1', 1, 'corr'),
                                                         ('conv2_2', 1, 'corr'),
                                                         ('pool2', 1, 'corr'),
                                                         ('conv3_1', 1, 'corr'),
                                                         ('conv3_2', 1, 'corr'),
                                                         ('conv3_3', 1, 'corr'),
                                                         ('conv3_4', 1, 'corr'),
                                                         ('pool3', 1, 'corr'),
                                                         ('conv4_1', 1, 'corr'),
                                                         ('conv4_2', 1, 'corr'),
                                                         ('conv4_3', 1, 'corr'),
                                                         ('conv4_4', 1, 'corr'),
                                                         ('pool4', 1, 'corr')
                                                     ])],
                                                    data_layer='data', crop=(16, -16, 16, -16), start_layer='conv1_1', use_bounds=True)

