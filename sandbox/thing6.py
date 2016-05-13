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


def get_diff(v, w):
    val = (v * w).sum()
    diff = w
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
        for w in targets[targ]:
            val_d, diff_d = get_diff(v, w)
            val += val_d
            diff += diff_d
            #print(diff_d, tp)

        #kwargs = {targ: diff}
        #N.backward(start=targ, end=target_order[tind+1], **kwargs)
        N.backward(start=targ, end=target_order[tind+1])

    diff = N.blobs[data_layer].diff
    diff = diff.reshape((3 * N_SIZE**2, )).astype(np.float64)

    return val, diff


def main(param_file, state_file, save_file, targets,
         data_layer='data', crop=None, mean=None, seed=0, use_bounds=False):

 
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

    net = caffe_pb2.NetParameter()
    text_format.Merge(open(param_file).read(), net)
    G = get_graph(net)
    succ_layers = G.successors(data_layer)
    start_layer = [ln for ln in N._layer_names if ln in succ_layers][0]


    global COUNT 
    COUNT = 0

    x0 = np.random.RandomState(seed=seed).uniform(size=(3 * N_SIZE**2 , ))
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
    return r
