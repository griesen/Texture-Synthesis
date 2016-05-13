import caffe
import numpy as np
import scipy.optimize as opt

N = caffe.Net('/home/ubuntu/new/caffe/examples/cifar10/cifar10_quick_train_test_rand.prototxt', 
              '/home/ubuntu/new/caffe/examples/cifar10/cifar10_quick_iter_5000.caffemodel.h5', 
              caffe.TRAIN)


#target = np.load('ip1_target.npy')[1: 2]
target = np.load('pool3_target.npy')[1: 2]
#target = np.load('conv1_target.npy')[1:2]

def get_val_and_diff(x):
    x = x.reshape((1, 3, 32, 32)).astype(np.float32)
    N.blobs['data'].data[:] = x
    N.forward(start='conv1')
    tlayer = 'pool3'
    v = N.blobs[tlayer].data[0].copy()

    val = ((v - target)**2).sum()

    diff_top = 2 * (v - target)

    kwargs = {tlayer: diff_top}
    N.backward(start=tlayer,
               **kwargs)
    
    diff = N.blobs['data'].diff

    diff = diff.reshape((3 * 32 * 32, )).astype(np.float64)

    return val, diff


#def main():
#    x0 = np.random.RandomState(0).uniform(size=(3 * 32 * 32, ))
    
def main(save_file):
    x0 = np.random.RandomState(0).uniform(size=(3 * 32 *32 , ))
    r = opt.fmin_l_bfgs_b(get_val_and_diff, x0)
    v1 = r[0].reshape((3, 32, 32))
    np.save(save_file, v1)
