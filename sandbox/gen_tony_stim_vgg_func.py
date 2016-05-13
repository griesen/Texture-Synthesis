from PIL import Image
import os
import numpy as np
import thing4


# pf = '/home/render/synthesis/sandbox/vgg_avg.prototxt'
# sf = '/home/render/synthesis/sandbox/vgg_normalised.caffemodel'
pf = '/Users/babylab/Desktop/sandbox/model.prototxt'
sf = '/Users/babylab/Desktop/sandbox/model_parameters.caffemodel'

STAT_LIST = [('conv1_1', 1000, 'corr_rs'), 
             ('conv1_2', 1000, 'corr_rs'),
             ('pool1', 1000, 'corr_rs'),
             ('conv2_1', 1000, 'corr_rs'),
             ('conv2_2', 1000, 'corr_rs'),
             ('pool2', 1000, 'corr_rs'),
             ('conv3_1', 1000, 'corr_rs'),
             ('conv3_2', 1000, 'corr_rs'),
             ('conv3_3', 1000, 'corr_rs'),
             ('conv3_4', 1000, 'corr_rs'),
             ('pool3', 100, 'corr_rs'),
             ('conv4_1', 1000, 'corr_rs'),
             ('conv4_2', 1000, 'corr_rs'),
             ('conv4_3', 1000, 'corr_rs'),
             ('conv4_4', 10000, 'corr_rs'),
             ('pool4', 1000, 'corr_rs')]

STAT_LIST_2 = [('pool4', 10, 'ss'),
               ('pool5', 10, 'ss')]


def get_stim_helper(args):
    fname = args['fname']
    stat_n = args.get('stat_n', None)
    stat_n2 = args.get('stat_n2', None)
    seed = args.get('seed', 0)
    out_fname = args['out_fname']
    save_freq = args.get('save_freq', 100)
    maxfun = args.get('maxfun', 3000)
    return get_stim(fname=fname, stat_n=stat_n, stat_n2=stat_n2, seed=seed, out_fname=out_fname, save_freq=save_freq, maxfun=maxfun)


def get_stim(fname, stat_n, stat_n2, seed, out_fname, save_freq, maxfun):
    impath = os.path.join('/Users/babylab/Desktop/sandbox/textures/texture_jpgs/', fname)
    im_arr = np.asarray(Image.open(impath).convert('RGB').resize((224, 224), resample=Image.ANTIALIAS)).transpose(2, 0, 1).reshape((1, 3, 224, 224))

    if stat_n is not None:
        stat_list = STAT_LIST[: stat_n]
    else:
        stat_list = STAT_LIST[:]

    if stat_n2 is not None:
        stat_list += STAT_LIST_2[: stat_n2]

    dirname = os.path.join('/Users/babylab/Desktop/sandbox/textures/generated', out_fname)
    final_path = os.path.join(dirname, 'genstim.npy')
    if os.path.exists(final_path):
        print('%s already finished, exiting' % dirname)
        return 
    else:
        hdir = os.path.join(dirname, 'history')
        if os.path.isdir(hdir):
            hist = filter(lambda x: x.startswith('im_'), os.listdir(hdir))
        else:
            hist = []
        if len(hist) > 0:
            histints = [int(x.split('_')[-1].split('.')[0]) for x in hist]
            mhist = max(histints)
            start_path = os.path.join(hdir, 'im_%d.npy' % mhist)
            print('Starting with %s' % start_path)
            x0 = np.load(start_path)
            r, targets = thing4.main(pf, sf, final_path,  
                             [(im_arr, stat_list)], 
                             use_bounds=True, 
                             data_layer='data', 
                             start_layer='conv1_1',
                             start_normal=False,
                             start_im = x0,
                             count_start = mhist,
                             crop=(16, -16, 16, -16),
                             save_dir = hdir,
                             save_freq=save_freq,
                             seed=seed, 
                             opt_kwargs={'maxfun': maxfun})

        else:
            print('Starting %s from scratch' % dirname)
            r, targets = thing4.main(pf, sf, final_path,  
                             [(im_arr, stat_list)], 
                             use_bounds=True, 
                             data_layer='data', 
                             start_layer='conv1_1',
                             start_normal=False,
                             crop=(16, -16, 16, -16),
                             save_dir = hdir,
                             save_freq=save_freq,
                             seed=seed, 
                             opt_kwargs={'maxfun': maxfun})
        return r, targets

