import thing4
import dldata.stimulus_sets.hvm as hvm
from collections import OrderedDict


preproc = OrderedDict([(u'normalize', False), (u'dtype', u'float32'), (u'resize_to', [256, 256, 3]), (u'mode', u'RGB'), (u'crop', None), (u'mask', None)])

dataset = hvm.HvMWithDiscfade()
imgs = dataset.get_images(preproc=preproc)
pf = '/home/ubuntu/new/caffe/examples/cifar10/roschinet_larger_rand_test.prototxt'
sf = '/home/ubuntu/new/caffe/examples/cifar10/roschinet_larger.caffemodel.h5'

im0 = imgs[0].transpose((2, 0, 1)).reshape((1, 3, 256, 256))
im1 = imgs[-1].transpose((2, 0, 1)).reshape((1, 3, 256, 256))
meta = dataset.meta
inds = meta['category'] == 'Cars'
ims = [imgs[i].transpose((2, 0, 1)).reshape((1, 3, 256, 256)) for i in inds[::20][:10]]
r, targets = thing4.main(pf, sf, '/home/ubuntu/test_ig.npy', [(ims[i], [('conv4', 100, 'ss')]) for i in range(10) ], 'data')
