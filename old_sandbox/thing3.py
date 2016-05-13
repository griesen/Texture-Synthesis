import functools
import caffe
import numpy as np
import scipy.optimize as opt

N = caffe.Net('/home/ubuntu/new/caffe/examples/cifar10/cifar10_quick_train_test_rand.prototxt', 
              '/home/ubuntu/new/caffe/examples/cifar10/cifar10_quick_iter_5000.caffemodel.h5', 
              caffe.TRAIN)

def get_corr(v):
    vshp = v.shape
    vm = v[0].reshape((vshp[1], vshp[2] * vshp[3]))
    return np.dot(vm, vm.T)


def get_val_and_diff(x, tlayer, target):
    #x = x.reshape((1, 3, 32, 32))
    N.blobs['data'].data[:] = x
    N.forward(start='conv1')
    v = N.blobs[tlayer].data[0].copy()

    nv = np.prod(v.shape)
    val = ((v - target)**2).sum() / nv

    diff_top = 2 * (v - target) / nv

    kwargs = {tlayer: diff_top}
    N.backward(start=tlayer,
               **kwargs)

    diff = N.blobs['data'].diff

    return val, diff


def get_val_and_diff_corr(x, tlayer, target):
    #x = x.reshape((1, 3, 32, 32))
    N.blobs['data'].data[:] = x
    N.forward(start='conv1')

    v = N.blobs[tlayer].data.copy()
    vmC = get_corr(v)

    n = v.shape[1]
    m = v.shape[2] * v.shape[3]

    val = ((vmC - target)**2).sum() / 4. * (1./(n**2 * m**2))

    diff_top = (1./(n**2 * m**2)) *  np.dot(v[0].reshape(n, m).T, vmC - target)
    diff_top = diff_top.T.reshape((1, n, v.shape[2], v.shape[3]))

    kwargs = {tlayer: diff_top}
    N.backward(start=tlayer,
               **kwargs)
    
    diff = N.blobs['data'].diff

    return val, diff

    
def main(array_in, save_file, ss_weights=None, corr_weights=None):
    if ss_weights is None:
        ss_weights = {}
    if corr_weights is None:
        corr_weights = {}

    N.blobs['data'].data[:] = array_in
    N.forward(start='conv1')

    target_ss = {}
    for ss_w in ss_weights:
        target_ss[ss_w] = N.blobs[ss_w].data.copy()
    target_corr = {}
    for corr_w in corr_weights:
        target_corr[corr_w] = get_corr(N.blobs[corr_w].data.copy())

    def func(x):
        x = x.astype(np.float32)
        x = x.reshape((1, 3, 32, 32))
        loss = 0
        grad = np.zeros_like(x).astype(np.float32)
        #grad = np.zeros_like(x)
        for k, v in ss_weights.items():
            dloss, dgrad = get_val_and_diff(x, k, target_ss[k])
            loss += dloss * v
            grad += dgrad * v
        for k, v in corr_weights.items():
            dloss, dgrad = get_val_and_diff_corr(x, k, target_corr[k])
            loss += dloss * v
            grad += dgrad * v
    
        return loss, grad.reshape((3 * 32 * 32, )).astype(np.float64)

    x0 = np.random.RandomState(0).uniform(size=(3 * 32 *32 , ))
    r = opt.fmin_l_bfgs_b(func, x0)
    v1 = r[0].reshape((3, 32, 32))
    np.save(save_file, v1)
    return r, func, target_ss, target_corr
