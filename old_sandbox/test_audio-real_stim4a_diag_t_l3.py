import cPickle
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

x = cPickle.load(open('/home/ubuntu/new/caffe/cgrams_jmesFailures.cpy'))
im = Image.fromarray((np.tile(x['all_cgrams'][:, :, 8], (3, 1, 1))*255).astype(np.uint8).transpose((1, 2, 0))).resize((225, 225), resample=Image.ANTIALIAS)
im = np.asarray(im).transpose((2, 0, 1)).reshape((1, 3, 225, 225))
print(im.shape)
r, targets = thing4.main(pf, sf, '/home/ubuntu/test_audio_stim4a_diag_t_l3/test_real.npy', 
                         [(im, [('conv1', 1, 'corr_diag_t'), 
                                ('pool1', 1, 'corr_diag'), 
                                ('conv2', 1, 'corr_diag'), 
                                ('pool2', 1, 'corr_diag'), 
                                #('conv3', 1000, 'corr_diag'), 
                                #('conv4', 100000, 'corr_diag'),  
                                #('conv5', 100000, 'corr_diag'), 
                                #('pool5', 100000, 'corr_diag'),
                                #('fc6', 100000, 'ss')
                            ])], 'data', seed=2, use_bounds=True, 
                         #start_normal=False, start_im = im,
                         mean=mean, save_dir = '/home/ubuntu/test_audio_stim4a_diag_t_l3/things/', save_freq=100)

