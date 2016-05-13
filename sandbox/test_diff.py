import cPickle
import numpy as np
import caffe
import dldata.stimulus_sets.synthetic.transform_training as t
import scipy.optimize as opt

bsize = 200
minibsize = 5

dataset = t.RoschDatasetTransform()
meta = dataset.meta
tmeta = dataset.tmeta
data = dataset.get_images(preproc=t.DEFAULT_PREPROC)

N = caffe.Net('/home/ubuntu/new/caffe/examples/cifar10/caffenet_mean_diff.prototxt', '/home/ubuntu/new/caffe/examples/cifar10/bvlc_reference_caffenet.caffemodel', caffe.TRAIN)

N.transformer = caffe.io.Transformer({'data': N.blobs['data'].data.shape})
N.transformer.set_transpose('data', (2, 0, 1))

mean = np.load('/home/ubuntu/new/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy')
crop = (14, -15, 14, -15)
mean = mean[:, crop[0]: crop[1]][:, :, crop[2]: crop[3]]
N.transformer.set_mean('data', mean)


def stuff(inds, trans_vals=None):
    assert inds.max() < len(tmeta), (inds.max(), len(meta))
    assert len(inds) == N.blobs['data'].data.shape[0], (len(inds), N.blobs['data'].data.shape)
    tinds = len(tmeta) + inds
    N.blobs['data'].data[:] = 255 * data[tinds].transpose((0, 3, 1, 2))[:, :, crop[0]: crop[1]][:, :, :, crop[2]: crop[3]]

    res = N.forward(end='fc6')['fc6'].copy()

    N.blobs['data'].data[:] = 255 * data[inds].transpose((0, 3, 1, 2))[:, :, crop[0]: crop[1]][:, :, :, crop[2]: crop[3]]

    N.forward(end='fc6')['fc6'].copy()

    ma = np.array([[_m[f]  for f in ['ty', 'tz', 's', 'rxy', 'rxz', 'ryz']] for _m in tmeta[inds]])

    N.blobs['transform'].data[:] = ma
    N.forward(start='fc6', end='concat')
    N.blobs['diff_data'].data[:] = res
    if trans_vals is not None:
        N.layers[-3].blobs[0].data[:] = trans_vals[0]
        N.layers[-3].blobs[1].data[:] = trans_vals[1]
    N.forward(start='concat', end='loss')
    N.backward(end='concat')

    loss = N.blobs['loss'].data
    diff = list(N.layers[-3].blobs)
    diff_w = diff[0].diff
    diff_b = diff[1].diff

    nshp = np.prod(N.blobs['fc6'].data.shape)
    print('nshp', nshp)
    loss *= (1. / nshp)
    diff_w *= (1. / nshp)
    diff_b *= (1. / nshp)

    return loss, diff_w, diff_b

COUNT = 0

def get_val_and_diff(x, wshp, bshp):
    global COUNT
    print('count %d' % COUNT)

    wlen = np.prod(wshp)
    blen = np.prod(bshp)
    assert len(x) == wlen + blen, (len(x), wlen, blen)
    w = x[:wlen].reshape(wshp)
    b = x[wlen: ].reshape(bshp)
    
    at_num = COUNT
    enum = at_num / (len(data) / 2)
    bnum = at_num % (len(data) / 2)
    perm = np.random.RandomState(seed=enum).permutation(len(data) / 2)
    
    print('enum', enum, bnum) 

    inds = perm[bnum: bnum + bsize]
    ncases = len(inds)
    COUNT += ncases

    print('ncases', ncases)
    loss = 0
    diff_w = np.zeros_like(w)
    diff_b = np.zeros_like(b)
    nminibs = int(np.ceil(ncases / minibsize))
    nc = 0.
    for _i in range(nminibs):
        _inds = inds[minibsize * _i: minibsize * (_i+1)]
        _loss, _diff_w, _diff_b = stuff(_inds, trans_vals = (w, b))
        f0 = (nc / (nc + len(_inds)))
        loss = f0 *  loss + (1 - f0) * _loss / len(_inds)
        diff_w = f0 * diff_w + (1 - f0) * _diff_w / len(_inds)
        diff_b = f0 * diff_b + (1 - f0) * _diff_b / len(_inds)
        nc += len(_inds)
        print('...mini', _loss, np.abs(_diff_w).max(), np.abs(_diff_b).max(), len(_inds))

    diff = np.concatenate([diff_w.ravel(), diff_b.ravel()])

    print('loss', loss, np.abs(diff).max())
    return loss, diff
    
    
def main(seed=0, opt_kwargs=None):
    if opt_kwargs is None:
        opt_kwargs = {}
    
    wshp = N.layers[-3].blobs[0].data.shape
    bshp = N.layers[-3].blobs[1].data.shape
    L = np.prod(wshp) + np.prod(bshp)
    x0 = np.random.RandomState(seed=seed).normal(size=L)
    print(x0.shape)

    r = opt.fmin_l_bfgs_b(get_val_and_diff, 
                          x0,
                          args = (wshp, bshp),
                          **opt_kwargs)

    return r
