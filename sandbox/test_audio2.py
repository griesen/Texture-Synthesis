import thing4
import dldata.stimulus_sets.hvm as hvm
from collections import OrderedDict


preproc = OrderedDict([(u'normalize', False), (u'dtype', u'float32'), (u'resize_to', [256, 256, 3]), (u'mode', u'RGB'), (u'crop', None), (u'mask', None)])

pf = '/home/ubuntu/new/caffe/examples/cifar10/audio2.prototxt'
sf = '/home/ubuntu/new/caffe/examples/cifar10/audio.caffemodel.h5'
import numpy as np
from PIL import Image
im = np.asarray(Image.open('/home/ubuntu/imgres.jpg').resize((256, 256), resample=Image.ANTIALIAS)).transpose(2, 0, 1).reshape((1, 3, 256, 256)) 

im1 = np.asarray(Image.open('/home/ubuntu/imgres-2.png').convert('RGB').resize((225, 225), resample=Image.ANTIALIAS)).transpose(2, 0, 1).reshape((1, 3, 225, 225)) 

mean = np.load('/home/ubuntu/new/caffe/audio_batches_mean.npy')

r, targets = thing4.main(pf, sf, '/home/ubuntu/test_audio2.npy', [(im1, [('pool1', 1, 'ss')])], 'data', seed=2, use_bounds=True, mean=mean)

