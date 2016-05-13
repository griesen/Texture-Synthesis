import os
import multiprocessing
import gen_tony_stim_vgg_func as F

def get_ofname(t, j, j1, s):
    l = F.STAT_LIST[:j][-1][0]
    l1 = F.STAT_LIST_2[:j1][-1][0]
    return t + '.%s_%s.%d'  % (l, l1, s)


def get_args():
    textures = filter(lambda x: x.endswith('.o.jpg'), 
                      os.listdir('/home/ubuntu/textures/'))
    stat_inds = [1, 3, 6, 11, None]
    seeds = [0]
    tjss = [('opie-4.19-256.o.jpg', None, 2, 1)]
    args = [dict(fname=t, stat_n=j, stat_n2=j1, seed=s, out_fname=get_ofname(t, j, j1, s)) for t, j, j1, s in tjss]
    print(args)
    return args


if __name__ == '__main__':
    args = get_args()
    pool = multiprocessing.Pool(processes=8)
    r = pool.map_async(F.get_stim_helper, args)
    h = r.get()
