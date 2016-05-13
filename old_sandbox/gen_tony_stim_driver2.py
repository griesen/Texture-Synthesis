import os
import multiprocessing
import gen_tony_stim_vgg_func as F

def get_ofname(t, j, s):
    l = F.STAT_LIST[:j][-1][0]
    return t + '.%s.%d'  % (l, s)


def get_args():
    textures = filter(lambda x: x.endswith('.o.jpg'), 
                      os.listdir('/home/ubuntu/textures/'))
    stat_inds = [1, 3, 6, 11, None]
    seeds = [0]
    tjss = [(t, j, 0) for t in ['opie-4.19-256.o.jpg', 'campbell256.o.jpg'] for j in stat_inds]
    args = [dict(fname=t, stat_n=j, seed=s, out_fname=get_ofname(t, j, s)) for t, j, s in tjss]
    print(args)
    return args


if __name__ == '__main__':
    args = get_args()
    pool = multiprocessing.Pool(processes=8)
    r = pool.map_async(F.get_stim_helper, args)
    h = r.get()
