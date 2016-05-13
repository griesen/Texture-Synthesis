import math
import h5py
from caffe import layers as L
from caffe import params as P

def prototxt_rep(r):
    n = caffe.NetSpec()
    layers = r['model_state']['layers']

    size = math.sqrt(layers['data']['outputs']/3.)
    n.data, n.label = L.DummyData(shape=[dict(dim=[1, 3, size, size]),
                                         dict(dim=[1, 1, 1, 1])])
    lays['data'] = n.data
    lays['label'] = n.label
    for lname in layer_names:
        layer0 = layers[lname]
        input_name = layer0['inputs'][0]
        if input_name + '_neuron' in lays:
            input_name = input_name +'_neuron'
        intput = lays[input_name]
        ltype == layer0['type']
        if ltype == 'conv':
            lay = L.Convolution(input, 
                               kernel_size = layer0['filterSize'][0],
                               pad = -layer0['padding'][0],
                               stride = layer0['stride'][0],
                               num_output = layer0['filters'])
        elif ltype == 'pool':
            ptype = P.Pooling.MAX if layer0['pool'] == 'max' else P.Pooling.AVE
            lay = L.Pooling(input,
                            kernel_size = layer0['sizeX'],
                            stride = layer0['stride'],
                            pool = ptype)
        elif ltype == 'fc':
            lay = L.InnerProduct(input, 
                                 num_output = layer0['outputs'])
        elif ltype == 'neuron':
            ntype = layer0['neuron']
            if ntype == 'relu':
                cname = 'ReLU'
            lay = getattr(L, cname)(input, in_place=True)
        elif ltype == 'cmrnorm':
            lay = L.LRN(input,
                        local_size = layer0['size'],
                        alpha = layer0['scale'],
                        beta = layer0['pow'])
        elif ltype == 'dropout2':
            lay = L.Dropout(input,
                            dropout_ratio = 0.5)
        else:
            raise ValueError("Layer type %s not recognized" % ltype)

        setattr(n, lname, lay)
        lays[lname] = lay
        if layer0.get('neuron'):
            ntype = layer0['neuron']
            nname = lname + '_neuron' 
            if ntype == 'relu':
                cname = 'ReLU'
            nlay = getattr(L, cname)(lay, in_place=True)
            setattr(n, nname, nlay)
            lays[nname] = nlay
    
    return n


def create_prototxt(r, out):
    layers = r['model_state']['layers']
    layers1 = prototxt_rep(r)
    write_protoxt_rep(r, out)


def create_caffemodel(r, out, layer_names):
    f = h5py.File(out, mode='a')
    grp = f.create_group('data')
    
    layers = r['model_state']['layers']
    for lname in layer_names:
        layer1 = grp.create_group(lname)
        print('Created group %s' % lname)
        if lname in layers:
            layer0 = layers[lname]
        else:
            print("Didn't find layer %s in original" % lname)
            continue
        if layer0['type'] == 'conv':
            weights = layer0['weights'][0]
            nc = layer0['channels'][0]
            nf = layer0['filters']
            assert nf == weights.shape[1], (nf, weights.shape)
            s = int(math.sqrt(weights.shape[0] / float(nc)))
            newshp = (nc, s, s, nf)
            weights = weights.reshape(newshp).transpose((3, 0, 1, 2))
            filt = layer1.create_dataset(name="0", 
                                     shape=weights.shape,
                                     dtype=weights.dtype)
            print('Filters of shape %s' % str(weights.shape))
            filt[:] = weights

            b = layer0['biases'][:, 0]
            bias = layer1.create_dataset(name="1",
                                         shape=b.shape,
                                         dtype=b.dtype)
            print('Bias of shape %s' % str(b.shape))
            bias[:] = b
        elif layer0['type'] == 'fc':
            weights = layer0['weights'][0]
            weights = weights.transpose((1, 0))
            filt = layer1.create_dataset(name="0", 
                                     shape=weights.shape,
                                     dtype=weights.dtype)
            filt[:] = weights

            b = layer0['biases'][0]
            bias = layer1.create_dataset(name="1",
                                         shape=b.shape,
                                         dtype=b.dtype)
            bias[:] = b
            print('Weights of shape %s' % str(weights.shape))
            print('Bias of shape %s' % str(b.shape))
        if layer0.get('neuron'):
            nname = lname + '_neuron'
            grp.create_group(nname)
            
    f.close()
