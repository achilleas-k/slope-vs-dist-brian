"""
Attempt to reproduce the plots of Kreuz et al., 2012
"""

import numpy as np
import matplotlib.pyplot as plt
import spikerlib as sl
import random as rnd
import pyspike as spk


def draw_the_plots(N, spiketrains, avgbiv, multiv):
    plt.figure()
    plt.subplot(3,1,1)
    for h, st in enumerate(spiketrains):
        plt.plot(st, np.zeros_like(st)+N-h, 'b.')
    plt.subplot(3,1,2)
    plt.plot(avgbiv[0], avgbiv[1])
    plt.subplot(3,1,3)
    plt.plot(multiv[0], multiv[1])

def poisson_train(start, end, rate):
    newspike = start+rnd.expovariate(rate)
    strain = []
    while newspike < end:
        strain.append(newspike)
        newspike += rnd.expovariate(rate)
    return strain

N = 20
start = 0
end = 0.8
spiketrains = []


print("Creating figure 2A")
print("Constructing spike trains ...")
spikecentres = np.arange(start, end+0.1, 0.1)
for nst in range(N):
    st = [0]
    shift = 0.005
    halfpoint = len(spikecentres)//2
    for idx, sc in enumerate(spikecentres[:halfpoint]):
        st.append(sc+nst*shift)
    st.extend(spikecentres[halfpoint:])
    spiketrains.append(st)

print("Caclulating stuff ...")
avgbiv = sl.metrics.kreuz.pairwise_mp(spiketrains, start, end, 80)
multiv = sl.metrics.kreuz.multivariate(spiketrains, start, end, 80)

print("Drawing plots ...")
draw_the_plots(N, spiketrains, avgbiv, multiv)

print("\nCreating figure 2B")
print("Constrcuting spike trains ...")
N = 20
start = 0
end = 2.0
spiketrains = []
packet_times = [0.2, 0.5]
packet_times.extend(np.arange(1, 2, 0.1))
packet_jitter = [0, 0.05]
packet_jitter.extend(np.linspace(0.02, 0, len(packet_times)-len(packet_jitter)))

for _ in range(N):
    newst = poisson_train(start, end/2, 10)
    for pt, pj in zip(packet_times, packet_jitter):
        if pj:
            newst = np.append(newst, np.random.normal(pt, pj))
        else:
            newst = np.append(newst, pt)
    spiketrains.append(np.sort(newst))

print("Calculating stuff ...")
avgbiv = sl.metrics.kreuz.pairwise_mp(spiketrains, start, end, 2000)
multiv = sl.metrics.kreuz.multivariate(spiketrains, start, end, 2000)

print("Drawing plots ...")
draw_the_plots(N, spiketrains, avgbiv, multiv)

for idx in range(len(spiketrains)):
    spiketrains[idx] = spk.add_auxiliary_spikes(spiketrains[idx], end)

print("Running pyspike function ...")
f = spk.spike_profile_multi(spiketrains)
plt.figure()
plt.plot(*(f.get_plottable_data()))
# display all figures
plt.show()
