import os
import multiprocessing
import gen_tony_stim_vgg_func as F

def get_ofname(t, j, s):
    l = F.STAT_LIST[:j][-1][0]
    return t + '.%s.%d'  % (l, s)


def get_args():
    textures = filter(lambda x: x.endswith('.o.jpg'), 
                      os.listdir('/home/render/synthesis/sandbox/textures/'))[:9]
    textures = textures + ['campbell256.o.jpg', 'opie-4.19-256.o.jpg']
    textures = list(set(textures))
    stat_inds = [1, 3, 6, 11, None]
    seeds = [0, 1]
    args = [dict(fname=t, stat_n=j, seed=s, out_fname=get_ofname(t, j, s)) for t in textures for j in stat_inds for s in seeds]
    args = [a for a in args if a['seed'] == 0 or a['stat_n'] in [3, None]]
    return args


if __name__ == '__main__':
    args = get_args()
    pool = multiprocessing.Pool(processes=8)
    r = pool.map_async(F.get_stim_helper, args)
    h = r.get()
