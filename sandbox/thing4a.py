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

        # configure pre-processing
        #in_ = self.inputs[0]
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
    m = vshp[2] * vshp[3]
    vm = v[0].reshape((vshp[1], vshp[2] * vshp[3]))
    return (1./ m) * np.dot(vm, vm.T)


def get_val(v, tp):
    if tp == 'ss':
        val = v
    elif tp == 'corr':
        val = get_corr(v)
    else:
        raise ValueError('unknown metric type: %s' % tp)
    return val


def get_ss_diff(v, t):
    nv = np.prod(v.shape)
    val = ((v - t)**2).mean()
    diff = np.where(v > 0, 2 * (v - t) / nv, 0)
    return val, diff


def get_max_diff(v, t):
    val = (v * t).sum()
    diff = t
    return val, diff


def get_corr_diff(v, t):
    vmC = get_corr(v)
    n = v.shape[1]
    m = v.shape[2] * v.shape[3]
    val = ((vmC - t)**2).mean() / 4 * n * m
    f =  v[0].reshape(n, m).T
    diff = np.where(f > 0, np.dot(f, vmC - t) / (n * m), 0)
    diff = diff.T.reshape((1, n, v.shape[2], v.shape[3]))
    return val, diff


def get_diff(v, t, tp):
    if tp == 'ss':
        val, diff = get_ss_diff(v, t)
    elif tp == 'corr':
        val, diff = get_corr_diff(v, t)
    elif tp == 'max':
        val, diff = get_max_diff(v, t)
    else:
        raise ValueError('unknown metric type: %s' % tp)
    return val, diff


def get_val_and_diff(x, targets, data_layer, start_layer):
    """
    targert = {tlayer1: {[(weight, target, type), .... ]}, 
               ... }
    """
    global COUNT
    print('count %d' % COUNT)
    COUNT += 1

    x = x.astype(np.float32)
    x = x.reshape((1, 3, N_SIZE, N_SIZE))
    N.blobs[data_layer].data[:] = x
    N.forward(start=start_layer)
    names = list(N._layer_names)
    target_order = [t for t in names[::-1] if t in targets] 
    target_order = target_order + [None]
    
    val = 0
    N.blobs[target_order[0]].diff[:] = 0
 
    for tind, targ in enumerate(target_order[:-1]):
        diff = N.blobs[targ].diff
        v = N.blobs[targ].data.copy()
        for w, t, tp in targets[targ]:
            val_d, diff_d = get_diff(v, t, tp)
            val += w * val_d
            diff += w * diff_d
            #print(diff_d, tp)

        #kwargs = {targ: diff}
        #N.backward(start=targ, end=target_order[tind+1], **kwargs)
        N.backward(start=targ, end=target_order[tind+1])

    diff = N.blobs[data_layer].diff
    diff = diff.reshape((3 * N_SIZE**2, )).astype(np.float64)

    return val, diff


def main(param_file, state_file, save_file, targets0, data_layer='data', crop=None, mean=None, seed=0, use_bounds=False, start_normal=True, start_layer=None):
 
    global N
    global N_SIZE
    if N is None:
        if mean is None:
            mean = np.load('/home/ubuntu/new/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy')
        if crop is not None:
            mean = mean[:, crop[0]: crop[1]][:, :, crop[2]: crop[3]]
        N = NewNet(param_file, state_file, caffe.TRAIN,
                   mean=mean)
        N_SIZE = mean.shape[1]

    if start_layer is None:
        net = caffe_pb2.NetParameter()
        text_format.Merge(open(param_file).read(), net)
        G = get_graph(net)
        succ_layers = G.successors(data_layer)
        start_layer = [ln for ln in N._layer_names if ln in succ_layers][0]

    targets = {}
    """
    for (img_arr, lay_wt_tp_list) in targets0:
        N.blobs[data_layer].data[:] = img_arr
        N.forward(start=start_layer)
        for lay, wt, tp in lay_wt_tp_list:
            if lay not in targets:
                targets[lay] = []
            t = get_val(N.blobs[lay].data.copy(), tp)
            targets[lay].append((wt, t, tp))
    """

    for (img_arr, lay_wt_tp_list) in targets0:
        if not isinstance(img_arr, list):
            img_arr = [img_arr]

        for lay, wt, tp in lay_wt_tp_list:
            if lay not in targets:
                targets[lay] = []
            if not isinstance(tp, tuple) or tp[0] != 'max':
                T = None
                for _ind, imga in enumerate(img_arr):
                    N.blobs[data_layer].data[:] = imga
                    N.forward(start=start_layer)
                    t = get_val(N.blobs[lay].data.copy(), tp)
                    if _ind == 0:
                        T = t
                    else:
                        T = T * _ind / (_ind + 1.) + t * (1. / (_ind + 1.))
                print('%d item(s)' % len(img_arr))
            else:
                T = tp[1]
                tp = tp[0]
            targets[lay].append((wt, T, tp))

    global COUNT 
    COUNT = 0

    if start_normal:
        x0 = np.random.RandomState(seed=seed).uniform(size=(3 * N_SIZE**2 , ))
    else:
        print('starting not normal')
        x0 = np.random.RandomState(seed=seed).randint(255, size=(3 * N_SIZE**2, )).astype(np.float32)
    print(mean.shape, x0.shape)
    if use_bounds:
        r = opt.fmin_l_bfgs_b(get_val_and_diff, x0, 
                  args=(targets, data_layer, start_layer), 
                  bounds=[(0, 255)] * len(x0))
    else:
        r = opt.fmin_l_bfgs_b(get_val_and_diff, x0, 
                args=(targets, data_layer, start_layer))
    v1 = r[0].reshape((3, N_SIZE, N_SIZE))
    np.save(save_file, v1)
    return r, targets
