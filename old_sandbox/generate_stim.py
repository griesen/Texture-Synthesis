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

var = np.zeros((1000,)).astype(np.float32)
var[23] = -1

r, targets = thing4.main(pf, sf, '/home/ubuntu/genstim1.npy',  [(None, [('fc8', (100, var), 'max')])], 'data', use_bounds=False, crop=(14, -15, 14, -15), start_normal=False, start_im=im)
