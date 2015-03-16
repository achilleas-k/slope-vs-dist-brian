import numpy as np
import spikerlib as sl
from brian import second, ms, Hz
import multiprocessing as mp

def calc_dist(args):
    n, r, dura = args
    r = r*Hz
    #for i in range(n):
    #    ptrains.append(sl.tools.poisson_spikes(dura, r))
    traingen = sl.tools.fast_synchronous_input_gen(n, r, 0.9, 0*ms, dura)
    ptrains = []
    for i in range(n):
        ptrains.append([])
    for spiketup in traingen.get_spiketimes():
        idx, time = spiketup
        ptrains[idx].append(time)
    t, k = sl.metrics.kreuz.multivariate(ptrains, 0*second, dura,
                                         int(dura/ms))
    d = np.trapz(k, t)/float(dura)
    print("{:.3f}".format(d))
    return d

N = range(50, 201, 50)
dura = 3*second
rate = range(50, 101, 50)

pool = mp.Pool()
args = []
for n in N:
    for r in rate:
        args.append((n, r, dura))
        #print("{}, {}".format(n, r))
result_iter = pool.imap(calc_dist, args)
pool.close()
pool.join()
dists = []
for res in result_iter:
    dists.append(res)
    #print(res)
print("Highest distance measured: {}".format(max(dists)))
