import caffe
import numpy as np
import scipy.optimize as opt
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import networkx

N = None

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
    val = ((v - t)**2).sum() / nv
    diff = 2 * (v - t) / nv
    return val, diff


def get_corr_diff(v, t):
    vmC = get_corr(v)
    n = v.shape[1]
    m = v.shape[2] * v.shape[3]
    val = ((vmC - t)**2).sum() / 4. * (1./(n**2 * m**2))
    diff = (1./(n**2 * m**2)) *  np.dot(v[0].reshape(n, m).T, vmC - t)
    diff = diff.T.reshape((1, n, v.shape[2], v.shape[3]))
    return val, diff


def get_diff(v, t, tp):
    if tp == 'ss':
        val, diff = get_ss_diff(v, t)
    elif tp == 'corr':
        val, diff = get_corr_diff(v, t)
    else:
        raise ValueError('unknown metric type: %s' % tp)
    return val, diff


def get_val_and_diff(x, targets, data_layer, start_layer):
    """
    targert = {tlayer1: {[(weight, target, type), .... ]}, 
               ... }
    """
    x = x.astype(np.float32)
    x = x.reshape((1, 3, 32, 32))
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
    diff = diff.reshape((3 * 32 * 32, )).astype(np.float64)

    return val, diff


def main(param_file, state_file, save_file, targets0, data_layer):
 
    global N
    if N is None:
        N = caffe.Net(param_file, state_file, caffe.TRAIN)

    net = caffe_pb2.NetParameter()
    text_format.Merge(open(param_file).read(), net)
    G = get_graph(net)
    succ_layers = G.successors(data_layer)
    start_layer = [ln for ln in N._layer_names if ln in succ_layers][0]

    targets = {}
    for (img_arr, lay_wt_tp_list) in targets0:
        N.blobs[data_layer].data[:] = img_arr
        N.forward(start=start_layer)
        for lay, wt, tp in lay_wt_tp_list:
            if lay not in targets:
                targets[lay] = []
            t = get_val(N.blobs[lay].data.copy(), tp)
            targets[lay].append((wt, t, tp))

    x0 = np.random.RandomState(0).uniform(size=(3 * 32 *32 , ))
    r = opt.fmin_l_bfgs_b(get_val_and_diff, x0, args=(targets, data_layer, start_layer))
    v1 = r[0].reshape((3, 32, 32))
    np.save(save_file, v1)
    return r, targets
