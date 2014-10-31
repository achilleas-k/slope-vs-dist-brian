from __future__ import print_function, division
import brian
from brian import mV, ms, second
import numpy as np
import spikerlib as sl
import matplotlib.pyplot as plt

def seq_pairs(lst):
    i = iter(lst)
    first = prev = item = i.next()
    for item in i:
        yield prev, item
        prev = item
    yield item, first

def interval_integral(membrane, spiketrain, dt=0.0001):
    integrals = []
    for pre, cur in seq_pairs(spiketrain):
        pre_dt = int(pre/dt)
        cur_dt = int(cur/dt)
        integrals.append(np.sum(membrane[pre_dt:cur_dt]))
    return integrals

def scaled_integral(membrane, spiketrain, dt=0.0001):
    integrals = []
    for pre, cur in seq_pairs(spiketrain):
        pre_dt = int(pre/dt)
        cur_dt = int(cur/dt)
        scaling = np.linspace(membrane[pre_dt+1], membrane[cur_dt],
                              cur_dt-pre_dt)
        integrals.append(np.sum(membrane[pre_dt:cur_dt]*scaling))
    return integrals


duration = 5*second
print("Creating NeuronGroup ...")
netw = brian.Network()
Vth = 20*mV
tau = 10*ms
neurons = brian.NeuronGroup(1, "dV/dt = -V/tau : volt",
                            threshold="V>Vth", reset="V=0*mV",
                            refractory=5*ms)
neurons.V = 0*mV
netw.add(neurons)

# inputs
print("Creating inputs ...")
base_spiketrain = np.arange(0.1*second, duration, 0.1*second)
input_spiketimes = []
n_input_trains = 10
drift = 0*ms
for spiketime in base_spiketrain:
    drift -= 0.1*ms
    for tridx in range(n_input_trains):
        input_spiketimes.append((tridx, spiketime*second+tridx*drift))

ingroup = brian.SpikeGeneratorGroup(n_input_trains, input_spiketimes)
netw.add(ingroup)
print("Connecting inputs to neurons ...")
weight = Vth*1.5/n_input_trains
print("Synaptic weight: %s" % str(weight))
connection = brian.Connection(ingroup, neurons, state='V', weight=weight)
netw.add(connection)

# monitors
print("Setting up monitors ...")
tracemon = brian.StateMonitor(neurons, 'V', record=True)
spikemon = brian.SpikeMonitor(neurons)
netw.add(tracemon, spikemon)

print("Running simulation for %.2f seconds ..." % duration)
netw.run(duration, report='stderr')
tracemon.insert_spikes(spikemon, 40*mV)
print("Simulation done!")
input_spiketrains = []
for _ in range(n_input_trains):
    input_spiketrains.append([])
for spike in input_spiketimes:
    input_spiketrains[spike[0]].append(spike[1])
print("Calculating NPSS ...")
npss = sl.tools.npss(tracemon[0], spikemon[0], 0*mV, Vth, tau, 2*ms)
print("Calculating pairwise Kreuz distance (interval mode) ...")
krint = sl.metrics.kreuz.interval(input_spiketrains, spikemon[0], mp=True)
print("Calculating pairwise VP distance (interval mode) ...")
vpdist = sl.metrics.victor_purpura.interval(input_spiketrains, spikemon[0],
                                            float(1/(2*ms)))

print("Reticulating splines ...")
dt = brian.defaultclock.dt
winstarts = spikemon[0]-(2*ms)
winstarts_idx = (winstarts/dt).astype('int')
plt.figure()
plt.title("Membrane potential")
plt.plot(tracemon.times, tracemon[0])
plt.plot(tracemon.times[winstarts_idx], tracemon[0][winstarts_idx],
         'r.', markersize=10)

plt.figure()
plt.title("All measures")
plt.plot(spikemon[0], npss, label="NPSS")
plt.plot(krint[0], krint[1], label="Kreuz")
plt.plot(spikemon[0], vpdist, label="VP")
plt.legend()

plt.figure()
plt.title("Kreuz vs NPSS")
plt.scatter(krint[1], npss[1:])

plt.figure()
plt.title("VP vs NPSS")
plt.scatter(vpdist, npss[1:])

integrals = interval_integral(tracemon[0], spikemon[0])
scintegrals = scaled_integral(tracemon[0], spikemon[0])
plt.figure()
plt.title("Integrals")
plt.plot(spikemon[0], integrals, label="Integral")
plt.plot(spikemon[0], scintegrals, label="Scaled intgrl")
plt.legend()

plt.ion()
plt.show()

