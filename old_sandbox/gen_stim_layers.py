import thing4
import os
import numpy as np
import time
import sys
from PIL import Image
from scipy.io import savemat

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


def save_stim(stim_name):
    stim = np.load(stim_name)
    stim = stim.squeeze().transpose((1,2,0)).astype('uint8')
    save_name = stim_name[:stim_name.find('.npy')]
    savemat(save_name, {save_name: stim})


def generate_layers(path, filename, out_path):
    for i, layer in enumerate(layers):
        print i
        sys.stdout.flush()
        ts = time.time()
        im = np.asarray(Image.open(os.path.join(path, filename)).convert('RGB').resize((256, 256), resample=Image.ANTIALIAS)) \
                        .transpose(2, 0, 1).reshape((1, 3, 256, 256))[:, :, 16:-16][:, :, :, 16:-16]
        im_name = filename[:filename.find('.')]
        dirname = os.path.join(out_path,im_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        if layer is not None:
            STATS = STAT_LIST[:layer]
        else:
            STATS = STAT_LIST

        stim_name = os.path.join(dirname, im_name+'_layer'+str(i)+'.npy')
        r, targets = thing4.main(pf, sf, stim_name,
                                 [(im, STATS)],
                                 use_bounds=True,
                                 data_layer='data',
                                 start_layer='conv1_1',
                                 start_normal=False,
                                 crop=(16, -16, 16, -16),
                                 save_dir = os.path.join(dirname, im_name+'_layer'+str(i)+'_history'),
                                 save_freq=100,
                                 seed=0)
        save_stim(stim_name)
        print time.time()-ts
        sys.stdout.flush()


if __name__ == '__main__':
    assert len(sys.argv) == 4, 'Not enough arguments'
    path = sys.argv[1]
    filename = sys.argv[2]
    out_path = sys.argv[3]
    generate_layers(path, filename, out_path)