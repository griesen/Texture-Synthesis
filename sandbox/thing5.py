import caffe
import numpy as np
import scipy.optimize as opt
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import networkx

N_SIZE = None
N = None
COUNT = 0

class NewNet(caffe.Net):
    """
    Classifier extends Net for image class prediction
    by scaling, center cropping, or oversampling.
    Parameters
    ----------
    image_dims : dimensions to scale input for cropping/sampling.
        Default is to scale to net input size for whole-image crop.
    mean, input_scale, raw_scale, channel_swap: params for
        preprocessing options.
    """
    def __init__(self, model_file, pretrained_file, image_dims=None,
                 mean=None, input_scale=None, raw_scale=None,
                 channel_swap=None):
        caffe.Net.__init__(self, model_file, pretrained_file, caffe.TEST)

        in_ = "data"
        self.transformer = caffe.io.Transformer(
            {in_: self.blobs[in_].data.shape})
        self.transformer.set_transpose(in_, (2, 0, 1))
        if mean is not None:
            self.transformer.set_mean(in_, mean)
        if input_scale is not None:
            self.transformer.set_input_scale(in_, input_scale)
        if raw_scale is not None:
            self.transformer.set_raw_scale(in_, raw_scale)
        if channel_swap is not None:
            self.transformer.set_channel_swap(in_, channel_swap)

        self.crop_dims = np.array(self.blobs[in_].data.shape[2:])
        if not image_dims:
            image_dims = self.crop_dims
        self.image_dims = image_dims


def get_graph(net):
    G = networkx.DiGraph()
    for n in net.layer:
        for t in n.top:
            G.add_node(t)
        for b in n.bottom:
            for t in n.top:
                G.add_edge(b, t)
    return G

def get_corr(v):
    vshp = v.shape
    vm = v[0].reshape((vshp[1], vshp[2] * vshp[3]))
    return np.dot(vm, vm.T)


def get_val_and_diff(x, tlayer, target, start_layer, data_layer):
    #x = x.reshape((1, 3, 32, 32))
    N.blobs[data_layer].data[:] = x
    N.forward(start=start_layer)
    v = N.blobs[tlayer].data[0].copy()

    nv = np.prod(v.shape)
    val = ((v - target)**2).sum() / nv

    diff_top = 2 * (v - target) / nv

    kwargs = {tlayer: diff_top}
    N.backward(start=tlayer,
               **kwargs)

    diff = N.blobs[data_layer].diff

    return val, diff


def get_val_and_diff_corr(x, tlayer, target, start_layer, data_layer):
    #x = x.reshape((1, 3, 32, 32))
    N.blobs[data_layer].data[:] = x
    N.forward(start=start_layer)

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
    
    diff = N.blobs[data_layer].diff

    return val, diff



def main(param_file, state_file, save_file, array_in, ss_weights, corr_weights, data_layer, crop=None):

    global N
    global N_SIZE
    if N is None:
        imgn_mean = np.load('/home/ubuntu/new/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy')
        if crop is not None:
                imgn_mean = imgn_mean[:, crop[0]: crop[1]][:, :, crop[2]: crop[3]]
        N = NewNet(param_file, state_file, caffe.TRAIN,
                   mean=imgn_mean)
        N_SIZE = imgn_mean.shape[1]

    net = caffe_pb2.NetParameter()
    text_format.Merge(open(param_file).read(), net)
    G = get_graph(net)
    succ_layers = G.successors(data_layer)
    start_layer = [ln for ln in N._layer_names if ln in succ_layers][0]

    targets = {}

    if ss_weights is None:
        ss_weights = {}
    if corr_weights is None:
        corr_weights = {}

    N.blobs[data_layer].data[:] = array_in
    N.forward(start=start_layer)

    target_ss = {}
    for ss_w in ss_weights:
        target_ss[ss_w] = N.blobs[ss_w].data.copy()
    target_corr = {}
    for corr_w in corr_weights:
        target_corr[corr_w] = get_corr(N.blobs[corr_w].data.copy())

    def func(x):
        x = x.astype(np.float32)
        x = x.reshape((1, 3, N_SIZE, N_SIZE))
        loss = 0
        grad = np.zeros_like(x).astype(np.float32)
        #grad = np.zeros_like(x)
        for k, v in ss_weights.items():
            dloss, dgrad = get_val_and_diff(x, k, target_ss[k], start_layer, data_layer)
            loss += dloss * v
            grad += dgrad * v
        for k, v in corr_weights.items():
            dloss, dgrad = get_val_and_diff_corr(x, k, target_corr[k], start_layer, data_layer)
            loss += dloss * v
            grad += dgrad * v
        global COUNT
        print(COUNT)
        COUNT += 1
        return loss, grad.reshape((3 * N_SIZE**2, )).astype(np.float64)

    global COUNT
    COUNT = 0

    x0 = np.random.RandomState(0).uniform(size=(3 * N_SIZE**2 , ))
    r = opt.fmin_l_bfgs_b(func, x0)
    v1 = r[0].reshape((3, N_SIZE, N_SIZE))
    np.save(save_file, v1)
    return r
