import caffe
import numpy as np
import scipy.optimize as opt

N = caffe.Net('/home/ubuntu/new/caffe/examples/cifar10/cifar10_quick_train_test_rand.prototxt', 
              '/home/ubuntu/new/caffe/examples/cifar10/cifar10_quick_iter_5000.caffemodel.h5', 
              caffe.TRAIN)


#target = np.load('ip1_target.npy')[1: 2]
#target = np.load('pool3_target.npy')[1: 2]
target = np.load('conv1_target.npy')[4:5]

def get_corr(v):
    vshp = v.shape
    vm = v[0].reshape((vshp[1], vshp[2] * vshp[3]))
    return np.dot(vm, vm.T)

print(target.shape)
target = get_corr(target)
print(target.shape)

def get_val_and_diff(x):
    x = x.reshape((1, 3, 32, 32)).astype(np.float32)
    N.blobs['data'].data[:] = x
    N.forward(start='conv1')

    tlayer = 'conv1'
    v = N.blobs[tlayer].data.copy()
    vmC = get_corr(v)

    n = v.shape[1]
    m = v.shape[2] * v.shape[3]

    val = ((vmC - target)**2).sum() / 4. * (1./n**2)

    diff_top = (1./n**2) *  np.dot(v[0].reshape(n, m).T, vmC - target)
    diff_top = diff_top.T.reshape((1, n, v.shape[2], v.shape[3]))

    kwargs = {tlayer: diff_top}
    N.backward(start=tlayer,
               **kwargs)
    
    diff = N.blobs['data'].diff

    diff = diff.reshape((3 * 32 * 32, )).astype(np.float64)

    return val, diff

    
def main(save_file):
    x0 = np.random.RandomState(0).uniform(size=(3 * 32 *32 , ))
    r = opt.fmin_l_bfgs_b(get_val_and_diff, x0)
    v1 = r[0].reshape((3, 32, 32))
    np.save(save_file, v1)
    return r
