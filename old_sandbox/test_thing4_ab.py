import thing4_a
import dldata.stimulus_sets.hvm as hvm
from collections import OrderedDict


preproc = OrderedDict([(u'normalize', False), (u'dtype', u'float32'), (u'resize_to', [256, 256, 3]), (u'mode', u'RGB'), (u'crop', None), (u'mask', None)])

dataset = hvm.HvMWithDiscfade()
imgs = dataset.get_images(preproc=preproc)
pf = '/home/ubuntu/new/caffe/examples/cifar10/roschinet_larger_rand_test.prototxt'
sf = '/home/ubuntu/new/caffe/examples/cifar10/roschinet_larger.caffemodel.h5'

meta = dataset.meta
inds = (meta['category'] == 'Cars').nonzero()[0]
ims = [255*imgs[i].transpose((2, 0, 1)).reshape((1, 3, 256, 256)) for i in inds[::20][:20]]
r, targets = thing4_a.main(pf, sf, '/home/ubuntu/test_blix2.npy', [([ims[i] for i in range(20)], [('pool3', 10, 'corr'), ('conv1', 1, 'ss')])], 'data')
