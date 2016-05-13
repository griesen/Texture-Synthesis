import os
import cPickle
import caffe
# caffe.set_device(0)
# caffe.set_mode_gpu()
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


def zeropad(x, t):
    xs = x.shape
    xs1 = tuple(np.array(xs) + 2 * np.array(t))
    t0 = np.array(t)
    t1 = np.array(xs1) - t0
    z = np.zeros(xs1).astype(x.dtype)
    slices = tuple([slice(_t0, _t1, None) for _t0, _t1 in zip(t0, t1)])
    z[slices] = x
    return z 


def get_smooth_diff(v):
    z = zeropad(v, [0] * (v.ndim - 2) + [1, 1])
    slicet0 = [slice(None)] * (v.ndim - 2) + [slice(2, None), slice(1, -1)]
    slicet1 = [slice(None)] * (v.ndim - 2) + [slice(None, -2), slice(1, -1)]
    slicet2 = [slice(None)] * (v.ndim - 2) + [slice(1, -1), slice(2, None)]
    slicet3 = [slice(None)] * (v.ndim - 2) + [slice(1, -1), slice(None, -2)]
    v1 = v - (1./4) * (z[slicet0] + z[slicet1] + z[slicet2] + z[slicet3])
    v1[:, :, 0] = 0; v1[:, :, -1] = 0; v1[:, :, :, 0] = 0; v1[:, :, :, -1] = 0
    
    #v1 = v - (1./4) * (z[:, :, 2:, 1:-1] + z[:, :, :-2, 1:-1] + z[:, :, 1:-1, 2:] + z[:, :, 1:-1, :-2])
    val = ( v1**2).sum() 

    v2 = zeropad(v1, [0] * (v.ndim - 2) + [1, 1])
    #diff = 2 * v1 - 0.5 * (v2[:, :, 2:, 1:-1] + v2[:, :, :-2, 1:-1] + v2[:, :, 1:-1, 2:] + v2[:, :, 1:-1, :-2]
    diff = 2 * v1 - (1./2) * (v2[slicet0] + v2[slicet1] + v2[slicet2] + v2[slicet3])

    return val, diff


def get_smoothsep_diff(v):
    z = zeropad(v, [0] * (v.ndim - 2) + [1, 1])
    slicet0 = [slice(None)] * (v.ndim - 2) + [slice(2, None), slice(1, -1)]
    slicet1 = [slice(None)] * (v.ndim - 2) + [slice(None, -2), slice(1, -1)]
    slicet2 = [slice(None)] * (v.ndim - 2) + [slice(1, -1), slice(2, None)]
    slicet3 = [slice(None)] * (v.ndim - 2) + [slice(1, -1), slice(None, -2)]
    slicet4 = [slice(None)] * (v.ndim - 2) + [slice(1, -1), slice(1, -1)]
    v1 = (z[slicet0] - z[slicet4])**2 + (z[slicet2] - z[slicet4])**2
    slicet5 = [slice(None)] * (v.ndim - 2) + [slice(None, 1), slice(None)]
    slicet6 = [slice(None)] * (v.ndim - 2) + [slice(None), slice(None, 1)]
    slicet7 = [slice(None)] * (v.ndim - 2) + [slice(-1, None), slice(None)]
    slicet8 = [slice(None)] * (v.ndim - 2) + [slice(None), slice(-1, None)]
    diff = 8 * z[slicet4] - 2 * (z[slicet0] + z[slicet1] + z[slicet2] + z[slicet3])
    #v1[slicet5] = 0 ; v1[slicet6] = 0; v1[slicet7] = v1[slicet8] = 0
    diff[slicet5] = diff[slicet6] = diff[slicet7] = diff[slicet8] = 0
    val = v1.sum()
    return val, diff


def get_smoothsep2_diff(v):
    z = zeropad(v, [0] * (v.ndim - 2) + [1, 1])
    slicet0 = [slice(None)] * (v.ndim - 2) + [slice(2, None), slice(1, -1)]
    slicet1 = [slice(None)] * (v.ndim - 2) + [slice(None, -2), slice(1, -1)]
    slicet2 = [slice(None)] * (v.ndim - 2) + [slice(1, -1), slice(2, None)]
    slicet3 = [slice(None)] * (v.ndim - 2) + [slice(1, -1), slice(None, -2)]
    slicet4 = [slice(None)] * (v.ndim - 2) + [slice(1, -1), slice(1, -1)]
    slicet5 = [slice(None)] * (v.ndim - 2) + [slice(None, 1), slice(None)]
    slicet6 = [slice(None)] * (v.ndim - 2) + [slice(None), slice(None, 1)]
    slicet7 = [slice(None)] * (v.ndim - 2) + [slice(-1, None), slice(None)]
    slicet8 = [slice(None)] * (v.ndim - 2) + [slice(None), slice(-1, None)]

    v1 = (z[slicet0] - z[slicet4])**2 + (z[slicet2] - z[slicet4])**2
    v1[slicet5] = 0 ; v1[slicet6] = 0; v1[slicet7] = v1[slicet8] = 0
    v2 = np.sqrt(v1)
    val = v2.sum()

    v3 = zeropad(v2, [0] * (v.ndim - 2) + [1, 1])
    diff = (z[slicet4] - z[slicet1]) / v3[slicet1] + \
           (2 * z[slicet4] - z[slicet0] - z[slicet2]) / v3[slicet4] + \
           (z[slicet4] - z[slicet3]) / v3[slicet3]
    diff[slicet5] = 0; diff[slicet6] = diff[slicet7] = diff[slicet8] = 0
    diff[np.isinf(diff)] = 0
    diff[np.isnan(diff)] = 0
    
    return val, diff



def get_corr(v):
    vshp = v.shape
    n = vshp[1]
    m = vshp[2] * vshp[3]
    vm = v[0].reshape(n, m)
    return (1./ m) * np.dot(vm, vm.T)


def get_corr_diag(v):
    vshp = v.shape
    n = vshp[1]
    m = vshp[2] * vshp[3]
    vm = v[0].reshape(n, m)
    return (1./ m) * np.dot(vm, vm.T).diagonal()


def get_corr_t(v):
    vshp = v.shape
    n = vshp[1] * vshp[2]
    m = vshp[3]
    vm = v[0].reshape(n, m)
    return (1./ m) * np.dot(vm, vm.T)


def get_corr_diag_t(v):
    vshp = v.shape
    n = vshp[1] * vshp[2]
    m = vshp[3]
    vm = v[0].reshape(n, m)
    return (1./ m) * np.dot(vm, vm.T).diagonal()


def get_val(v, tp):
    if tp in ['ss', 'ss_rs']:
        val = v
    elif tp in ['corr', 'corr_rs']:
        val = get_corr(v)
    elif tp in ['corr_diag']:
        val = get_corr_diag(v)
    elif tp in ['corr_t']:
        val = get_corr_t(v)
    elif tp in ['corr_diag_t']:
        val = get_corr_diag_t(v)
    else:
        raise ValueError('unknown metric type: %s' % tp)
    return val


def get_ss_diff(v, t):
    nv = np.prod(v.shape)
    val = ((v - t)**2).mean()
    diff = np.where(v > 0, 2 * (v - t) / nv, 0)
    return val, diff


def get_ss_diff_rescale(v, t):
    nv = np.prod(v.shape)
    val = ((v - t)**2).mean()
    diff = np.where(v > 0, 2 * (v - t), 0)
    return val, diff


def get_max_diff(v, t):
    val = (v * t).sum()
    diff = t.copy()
    return val, diff


def get_softmax_diff(v, t):
    v = v[0]
    vexp = np.exp(v)
    vexps = np.exp(misc.logsumexp(v))
    vq = vexp / vexps
    #print(vq.min(), vq.max())
    val =  -np.dot(vq , t)
    diff = vq * (np.dot(t, vq) - t)
    return val, diff


import scipy.misc as misc
def get_logsoftmax_diff(v, t):
    v = v[0]
    lse = misc.logsumexp(v)
    se = np.exp(lse)
    val = lse * t.sum() - np.dot(t, v) 
    diff = -t * (1 - np.exp(v) / se)
    return val, diff


def get_max2_diff(v, t):
    val = (v**2 * t).sum()
    diff = 2 * v * t
    return val, diff


def get_corr_diff(v, t):
    vmC = get_corr(v)
    n = v.shape[1]
    m = v.shape[2] * v.shape[3]
    val = ((vmC - t)**2).mean() / 4 
    f =  v[0].reshape(n, m).T
    diff = np.where(f > 0, np.dot(f, vmC - t) / (m**2 * n**2), 0)
    diff = diff.T.reshape((1, n, v.shape[2], v.shape[3]))
    return val, diff


def get_corr_diag_diff(v, t):
    vmC = get_corr_diag(v)
    n = v.shape[1]
    m = v.shape[2] * v.shape[3]
    val = ((vmC - t)**2).mean() / 4 
    f =  v[0].reshape(n, m).T
    #diff = np.where(f > 0, np.dot(f, vmC - t) / (m**2 * n), 0)
    #diff = diff.T.reshape((1, n, v.shape[2], v.shape[3]))
    diff = np.where(f > 0, (vmC - t) * f, 0)
    diff = diff.T.reshape((1, n, v.shape[2], v.shape[3]))  / (m * n)
    return val, diff


def get_corr_diff_rescale(v, t):
    vmC = get_corr(v)
    n = v.shape[1]
    m = v.shape[2] * v.shape[3]
    val = ((vmC - t)**2).mean() / 4 
    f =  v[0].reshape(n, m).T
    diff = np.where(f > 0, np.dot(f, vmC - t) / (m * n), 0)
    diff = diff.T.reshape((1, n, v.shape[2], v.shape[3]))
    return val, diff


def get_corr_t_diff(v, t):
    vmC = get_corr_t(v)
    n = v.shape[1] * v.shape[2]
    m = v.shape[3]
    val = ((vmC - t)**2).mean() / 4 
    f =  v[0].reshape(n, m).T
    #diff = np.where(f > 0, np.dot(f, vmC - t) / (m**2 * n**2), 0)
    diff = np.where(f > 0, np.dot(f, vmC - t) / (m**2 * n**2), 0)
    #diff =  np.dot(f, vmC - t) / (m)
    diff = diff.T.reshape((1, v.shape[1], v.shape[2], v.shape[3])) 
    return val, diff


def get_corr_diag_t_diff(v, t):
    vmC = get_corr_diag_t(v)
    n = v.shape[1] * v.shape[2]
    m = v.shape[3]
    val = ((vmC - t)**2).mean() / 4 
    f =  v[0].reshape(n, m).T
    #diff = np.where(f > 0, np.dot(f, vmC - t) / (m**2 * n), 0)
    #diff = diff.T.reshape((1, n, v.shape[2], v.shape[3]))
    diff = np.where(f > 0, (vmC - t) * f, 0)
    diff = diff.T.reshape((1, v.shape[1], v.shape[2], v.shape[3]))  / (n)
    return val, diff


def get_diff(v, t, tp):
    if tp == 'ss':
        val, diff = get_ss_diff(v, t)
    elif tp == 'ss_rs':
        val, diff = get_ss_diff_rescale(v, t)
    elif tp == 'corr':
        val, diff = get_corr_diff(v, t)
    elif tp == 'corr_diag':
        val, diff = get_corr_diag_diff(v, t)
    elif tp == 'corr_rs':
        val, diff = get_corr_diff_rescale(v, t)
    elif tp == 'corr_t':
        val, diff = get_corr_t_diff(v, t)
    elif tp == 'corr_diag_t':
        val, diff = get_corr_diag_t_diff(v, t)
    elif tp == 'max':
        val, diff = get_max_diff(v, t)
    elif tp == 'max2':
        val, diff = get_max2_diff(v, t)
    elif tp == 'softmax':
        val, diff = get_softmax_diff(v, t)
    elif tp == 'logsoftmax':
        val, diff = get_logsoftmax_diff(v, t)
    elif tp == 'smooth':
        val, diff = get_smooth_diff(v)
    elif tp == 'smoothsep':
        val, diff = get_smoothsep_diff(v)
    elif tp == 'smoothsep2':
        val, diff = get_smoothsep2_diff(v)
    else:
        raise ValueError('unknown metric type: %s' % tp)
    return val, diff


def get_val_and_diff(x, targets, data_layer, start_layer, save_dir, save_freq):
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
    if data_layer not in names:
        names = [data_layer] + names
    target_order = [t for t in names[::-1] if t in targets] 
    target_order = target_order + [None]
    
    val = 0
    N.blobs[target_order[0]].diff[:] = 0
 
    stats = {}
    for tind, targ in enumerate(target_order[:-1]):
        diff = N.blobs[targ].diff
        v = N.blobs[targ].data.copy()
        for w, t, tp in targets[targ]:
            val_d, diff_d = get_diff(v, t, tp)
            val += w * val_d
            diff += w * diff_d
            stats[(targ, tp, 'val')] = w * val_d
            stats[(targ, tp, 'diff')] = w * np.abs(diff_d).mean()
            print(targ, tp, stats[(targ, tp, 'val')], stats[(targ, tp, 'diff')])
     
        #kwargs = {targ: diff}
        #N.backward(start=targ, end=target_order[tind+1], **kwargs)
        end = target_order[tind+1]
        if end not in N._layer_names:
            end = None
        if targ != data_layer:
            N.backward(start=targ, end=end)

    diff = N.blobs[data_layer].diff
    diff = diff.reshape((3 * N_SIZE**2, )).astype(np.float64)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if COUNT % save_freq == 0:
            im_file = os.path.join(save_dir, 'im_%d.npy' % COUNT)
            np.save(im_file, N.blobs[data_layer].data)
            diff_file = os.path.join(save_dir, 'diff_%d.npy' % COUNT)
            np.save(diff_file, N.blobs[data_layer].diff)
            stats_file = os.path.join(save_dir, 'stats_%d.pkl' % COUNT)
            with open(stats_file, 'w') as _f:
                cPickle.dump(stats, _f)
    
    return val, diff


def main(param_file, state_file, save_file, targets0, data_layer='data', crop=None, mean=None, seed=0, use_bounds=False, start_normal=True, start_layer=None, start_im=None, save_dir=None, save_freq=None, opt_kwargs=None, count_start = 0):

    if opt_kwargs is None:
        opt_kwargs = {}

    if mean is None:
        #mean = np.load('/home/ubuntu/new/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy')
        mean = np.load('/Users/babylab/caffe/python/caffe/imagenet/ilsvrc_2012_mean.npy')
    if crop is not None:
        mean = mean[:, crop[0]: crop[1]][:, :, crop[2]: crop[3]]

    global N
    global N_SIZE
    if N is None:
        N = NewNet(param_file, state_file, caffe.TRAIN,
                   mean=mean)
        N_SIZE = mean.shape[1]

    if start_layer is None:
        net = caffe_pb2.NetParameter()
        text_format.Merge(open(param_file).read(), net)
        G = get_graph(net)
        succ_layers = G.successors(data_layer)
        start_layer = [ln for ln in N._layer_names if ln in succ_layers][0]
    # caffe.set_mode_gpu()
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
            if tp in ['corr', 'ss', 'corr_rs', 'ss_rs', 'corr_t', 'corr_diag', 'corr_diag_t']:
                T = None
                for _ind, imga in enumerate(img_arr):
                    N.blobs[data_layer].data[:] = imga
                    N.forward(start=start_layer)
                    print 'Yo'
                    t = get_val(N.blobs[lay].data.copy(), tp)
                    if _ind == 0:
                        T = t
                    else:
                        T = T * _ind / (_ind + 1.) + t * (1. / (_ind + 1.))
                print('%d item(s)' % len(img_arr))
            elif tp in ['smooth', 'smoothsep', 'smoothsep2']:
                T = None
            elif tp in ['max', 'max2', 'softmax', 'logsoftmax']:
                wt, T = wt
            else:
                raise ValueError("diff type %s not recognized" % tp)
            targets[lay].append((wt, T, tp))

    global COUNT 
    COUNT = count_start

    if start_normal == 'zeros':
        x0 = np.zeros((3 * N_SIZE**2, )).astype(np.float32)
    elif start_normal == 'ones':
        x0 = np.ones((3 * N_SIZE**2, )).astype(np.float32)
    elif start_normal:
        x0 = np.random.RandomState(seed=seed).uniform(size=(3 * N_SIZE**2 , ))
    elif start_im is not None:
        x0 = start_im.reshape((3 * N_SIZE**2, ))
    else:
        print('starting not normal')
        x0 = np.random.RandomState(seed=seed).randint(255, size=(3 * N_SIZE**2, )).astype(np.float32)
    print(mean.shape)
    print(x0.shape)
    if use_bounds:
        r = opt.fmin_l_bfgs_b(get_val_and_diff, x0, 
                              args=(targets, data_layer, start_layer, save_dir, save_freq), 
                              bounds=[(0, 255)] * len(x0), **opt_kwargs)
    else:
        r = opt.fmin_l_bfgs_b(get_val_and_diff, x0, 
                              args=(targets, data_layer, start_layer, save_dir, save_freq), **opt_kwargs)
    v1 = r[0].reshape((3, N_SIZE, N_SIZE))
    np.save(save_file, v1)
    return r, targets
