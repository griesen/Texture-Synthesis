import thing4
import os
import numpy as np
import time
import sys
from PIL import Image
from collections import OrderedDict

STAT_LIST = [('conv1_1', 1000, 'corr_rs'),
             ('conv1_2', 100, 'corr_rs'),
             ('pool1', 100, 'corr_rs'),
             ('conv2_1', 100, 'corr_rs'),
             ('conv2_2', 100, 'corr_rs'),
             ('pool2', 100, 'corr_rs'),
             ('conv3_1', 100, 'corr_rs'),
             ('conv3_2', 100, 'corr_rs'),
             ('conv3_3', 100, 'corr_rs'),
             ('conv3_4', 100, 'corr_rs'),
             ('pool3', 10, 'corr_rs'),
             ('conv4_1', 100, 'corr_rs'),
             ('conv4_2', 100, 'corr_rs'),
             ('conv4_3', 100, 'corr_rs'),
             ('conv4_4', 1000, 'corr_rs'),
             ('pool4', 100, 'corr_rs')]

pf = 'model.prototxt'
sf = 'model_parameters.caffemodel'

layers = [1, 3, 6, 11, None]
for i, layer in enumerate(layers):
    print i
    sys.stdout.flush()
    ts = time.time()
    im = 'campbell256.o.jpg'
    im2 = np.asarray(Image.open('textures/texture_jpgs/'+im).convert('RGB').resize((256, 256), resample=Image.ANTIALIAS)).transpose(2, 0, 1).reshape((1, 3, 256, 256))[:, :, 16:-16][:, :, :, 16:-16]
    im_name = im[:im.find('.')]+'_p2_layer'+str(i)
    dirname = 'textures/generated/'+im_name
    if layer is not None:
        STATS = STAT_LIST[:layer]
    else:
        STATS = STAT_LIST

    r, targets = thing4.main(pf, sf, os.path.join(dirname, 'genstim.npy'),
                             [(im2, STATS)],
                             use_bounds=True,
                             data_layer='data',
                             start_layer='conv1_1',
                             start_normal=False,
                             crop=(16, -16, 16, -16),
                             save_dir = os.path.join(dirname, 'history'),
                             save_freq=100,
                             seed=0)
    print time.time()-ts
    sys.stdout.flush()