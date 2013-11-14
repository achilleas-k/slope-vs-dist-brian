"""
Compare
the distance between N spike trains at f Hz
with
the distance between 2N spike trains at f/2 Hz


Try both Victor-Purpura and Kreuz.
"""
import numpy as np
import matplotlib.pyplot as plt
import random
from spike_distance_mp import mean_pairwise_distance as victor
from spike_distance_kreuz import multivariate_spike_distance as kreuz


def altpairs(thelist):
    theiter = iter(thelist)
    while True:
        yield(theiter.next(), theiter.next())

Nlist = range(4, 31, 2)
duration = 5  # seconds per spike train
freq = 20  # Hz

victor_results = []
kreuz_results = []
for N in Nlist:
    # generate N poisson spike trains
    Ntrains = []
    for _ in range(N):
        newtrain = []
        newspike = random.expovariate(freq)
        while newspike <= duration:
            newtrain.append(newspike)
            newspike += random.expovariate(freq)
        Ntrains.append(newtrain)
    vdist_N = victor(Ntrains, 200)
    # kreuz spazzes out when start time is 0
    kdist_N = np.mean(kreuz(Ntrains, 0.1, duration, 100)[1])
    halftrains = []
    for even, odd in altpairs(Ntrains):
        newtrain = sorted(even+odd)
        halftrains.append(newtrain)
    vdist_half = victor(halftrains, 200)
    kdist_half = np.mean(kreuz(Ntrains, 0.1, duration, 100)[1])
    victor_results.append((vdist_N, vdist_half))
    kreuz_results.append((kdist_N, kdist_half))

np.savez("dist_N-half.npz",
        Ntrains=Nlist,
        victor_dist=victor_results,
        kreuz_dist=kreuz_results)

vN, vH = zip(*victor_results)
kN, kH = zip(*kreuz_results)

vCoeff = np.mean(np.divide(vH,vN))
kCoeff = np.mean(np.divide(kH,kN))

plt.figure(figsize=(16,12), dpi=100)
plt.subplot(1,2,1)
plt.set_cmap('gray_r')
plt.scatter(vN, vH, c=Nlist, label='V-P')
plt.plot((min(vN), max(vN)), (min(vN)*vCoeff, max(vN)*vCoeff),
        '--k', label='Y=%.3f*X' % (vCoeff))
plt.xlabel('N spike trains -- f spikes per second')
plt.ylabel('N/2 spike trians -- 2f spikes per second')
plt.legend()
plt.title('Victor-Purpura distance')
plt.subplot(1,2,2)
plt.scatter(kN, kH, c=Nlist, label='kreuz')
plt.plot((min(kN), max(kN)), (min(kN)*kCoeff, max(kN)*kCoeff),
        '--k', label='Y=%.3f*X' % (kCoeff))
plt.xlabel('N spike trains -- f spikes per second')
plt.title('Kreuz multivariate spike distance')
plt.legend()
cbar = plt.colorbar()
cbar.set_label('N')
plt.subplots_adjust(wspace=0.4)
plt.suptitle("Duration: %.1f sec -- f: %.1f Hz" % (duration, freq))
plt.savefig('dist_num_freq.png', )





