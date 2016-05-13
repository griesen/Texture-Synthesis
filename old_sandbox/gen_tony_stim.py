import thing4
# import dldata.stimulus_sets.hvm as hvm
from collections import OrderedDict


preproc = OrderedDict([(u'normalize', False), (u'dtype', u'float32'), (u'resize_to', [256, 256, 3]), (u'mode', u'RGB'), (u'crop', None), (u'mask', None)])

# dataset = hvm.HvMWithDiscfade()
# imgs = dataset.get_images(preproc=preproc)
pf = '/Users/babylab/Desktop/sandbox/model.prototxt'
sf = '/Users/babylab/Desktop/sandbox/model_parameters.caffemodel'

#im0 = 255*imgs[0].transpose((2, 0, 1)).reshape((1, 3, 256, 256))
#im1 = 255*imgs[-1].transpose((2, 0, 1)).reshape((1, 3, 256, 256))
# meta = dataset.meta
# inds = (meta['category'] == 'Cars').nonzero()[0]
# ims = [255*imgs[i][14:-15][:, 14:-15].transpose((2, 0, 1)).reshape((1, 3, 227, 227)) for i in inds]
import numpy as np
from PIL import Image

im2 = np.asarray(Image.open('/Users/babylab/Desktop/sandbox/textures/texture_jpgs/campbell256.o.jpg').convert('RGB').resize((256, 256), resample=Image.ANTIALIAS)).transpose(2, 0, 1).reshape((1, 3, 256, 256))[:, :, 16:-16][:, :, :, 16:-16]

import os
dirname = '/Users/babylab/Desktop/sandbox/textures/generated/campbell256_p2'
r, targets = thing4.main(pf, sf, os.path.join(dirname, 'genstim.npy'),  
                         [(im2, [('conv1', 100, 'corr'), 
                                 ('norm1', 100, 'corr'), 
                                 ('pool1', 100, 'corr'), 
                                 ('conv2', 10000, 'corr', ), 
                                 ('norm2', 100, 'corr'), 
                                 ('pool2', 100, 'corr'),  
                                 #('conv3', 10000, 'corr'), 
                                 #('conv4', 100000, 'corr'), 
                                 #('conv5', 1000000 , 'corr'), 
                                 #('pool5', 1000000, 'corr'), 
                                 #('fc6', 10000, 'ss'), 
                                 #('fc7', 10000, 'ss'),
                                 #('fc8', 10000, 'ss')
                             ])], 
                          use_bounds=True, 
                         data_layer='data',
                         # start_layer='conv1',
                         start_normal=False,
                         crop=(16, -16, 16, -16),
                         save_dir = os.path.join(dirname, 'history'),
                         save_freq=100,
                         seed=0)
 

