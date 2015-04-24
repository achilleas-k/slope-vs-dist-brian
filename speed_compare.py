import time
import matplotlib.pyplot as plt
import spikerlib as sl
import pyspike as spk


def plot_spiketrains(spiketrains):
    h = 0
    for st in spiketrains:
        plt.plot(st, [h]*len(st), "b.")
        h+=1
    plt.show()

Nlist = [2, 10, 50, 100, 200, 300, 400, 500, 700, 1000]
sla = []
slm = []
spa = []
start = 0
end = 2
for N in Nlist:
    # contruct N Poisson spike trains
    spiketrains = []
    print("")
    print("Constructing {} spike trains ...".format(N))
    for n in range(N):
        spiketrains.append(sl.tools.poisson_spikes(end, 10))
    print("Running pairwise_mp ...")
    tstart = time.time()
    # _ = sl.metrics.kreuz.pairwise_mp(spiketrains, start, end, 2000)
    tend = time.time()
    sla.append(tend-tstart)
    print("Running multivariate ...")
    tstart = time.time()
    _ = sl.metrics.kreuz.multivariate(spiketrains, start, end, 2000)
    tend = time.time()
    slm.append(tend-tstart)
    for idx in range(len(spiketrains)):
        spiketrains[idx] = spk.add_auxiliary_spikes(spiketrains[idx], end)
    print("Running PySpike ...")
    tstart = time.time()
    _ = spk.spike_profile_multi(spiketrains)
    tend = time.time()
    spa.append(tend-tstart)
    print("Finished {}".format(N))

plt.figure()
plt.plot(Nlist, sla, label="PW-MP")
plt.plot(Nlist, slm, label="Multi")
plt.plot(Nlist, spa, label="PySpk")
plt.legend()
plt.show()
