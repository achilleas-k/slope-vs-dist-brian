from __future__ import print_function
from brian import *
import numpy as np
import spikerlib as sl
import itertools as itt
import matplotlib.pyplot as plt


# unit aliases
ms = msecond
Hz = hertz
mV = mvolt

# measure aliases
victor_purpura = sl.metrics.victor_purpura
kreuz = sl.metrics.kreuz
modulus = sl.metrics.modulus_metric
npss = sl.tools.normalised_pre_spike_slopes

# measure aliases
victor_purpura = sl.metrics.victor_purpura
kreuz = sl.metrics.kreuz
modulus = sl.metrics.modulus_metric
npss = sl.tools.normalised_pre_spike_slopes

netw = Network()
Vth = 20*mV
neuron = NeuronGroup(2, "dV/dt = 0*volt/second : volt",
                     threshold="V>Vth", reset="V=0*mV",
                     refractory=5*ms)
neuron.V = 0*mV
netw.add(neuron)

Nin = 50
wide_packet = np.random.normal(0, 4.0*ms, 50)
narrow_packet = np.random.normal(0, 0.1*ms, 50)
wide_narrow = ([ni+15*ms for ni in wide_packet]+
               [ni+40*ms for ni in narrow_packet])
narrow_wide = ([ni+15*ms for ni in narrow_packet]+
               [ni+40*ms for ni in wide_packet])

wn_trains = [[] for i in range(Nin)]
for wn, t in zip(itt.cycle(wn_trains), wide_narrow):
    wn.append(t)
nw_trains = [[] for i in range(Nin)]
for nw, t in zip(itt.cycle(nw_trains), narrow_wide):
    nw.append(t)
#for _idx, wn in enumerate(wn_trains):
#    plt.scatter(wn, np.ones(len(wn))*_idx, c='b')
#for _idx, nw in enumerate(nw_trains):
#    plt.scatter(nw, np.ones(len(nw))*_idx, c='r')

dura = max(max(wide_narrow), max(narrow_wide), 50*ms)
wn_t, wn_kreuz = kreuz.pairwise_mp(wn_trains, 0, dura, 100)
nw_t, nw_kreuz = kreuz.pairwise_mp(nw_trains, 0, dura, 100)

# random spike trains for comparison
randspiketrains = [np.cumsum(np.random.random(10)) for i in range(50)]
randspiketrains = [st*dura/max(st) for st in randspiketrains]
rand_t, rand_kreuz = kreuz.pairwise_mp(randspiketrains, 0, dura, 100)

# plot
plt.plot(wn_t, wn_kreuz, 'b')
plt.plot(nw_t, nw_kreuz, 'r')
plt.plot(rand_t, rand_kreuz, 'k')
plt.figure()
norm_factor = max(rand_kreuz)
plt.plot(wn_t, wn_kreuz/norm_factor, 'b')
plt.plot(nw_t, nw_kreuz/norm_factor, 'r')
plt.plot(rand_t, rand_kreuz/norm_factor, 'k')
plt.show()

