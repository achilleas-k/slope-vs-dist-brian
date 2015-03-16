import numpy as np
import spikerlib as sl
from brian import second, ms, Hz

N = range(10, 101, 10)
dura = 2*second
rate = range(10, 101, 10)

dists = []
for n in N:
    for r in rate:
        ptrains = []
        r = r*Hz
        for i in range(n):
            ptrains.append(sl.tools.poisson_spikes(dura, r))
        t, k = sl.metrics.kreuz.multivariate(ptrains, 0*second, dura,
                                             int(dura/ms))
        d = np.trapz(k, t)
        dists.append(d)
        print("Distance for {} poisson trains at {} Hz: {}".format(
            n, r, d))
print("Highest distance measured: {}".max(dists))
