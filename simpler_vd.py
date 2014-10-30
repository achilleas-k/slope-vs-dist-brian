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


duration = 30*second
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
drift = 0*ms
for spiketime in base_spiketrain:
    drift += 0.1*ms
    input_spiketimes.append((0, spiketime*second-drift))
    input_spiketimes.append((1, spiketime*second))

ingroup = brian.SpikeGeneratorGroup(2, input_spiketimes)
netw.add(ingroup)
print("Connecting inputs to neurons ...")
connection = brian.Connection(ingroup, neurons, state='V', weight=19*mV)
netw.add(connection)

# monitors
print("Setting up monitors ...")
tracemon = brian.StateMonitor(neurons, 'V', record=True)
spikemon = brian.SpikeMonitor(neurons)
netw.add(tracemon, spikemon)

print("Running simulation for %.2f seconds ..." % duration)
netw.run(duration, report='stderr')
tracemon.insert_spikes(spikemon, 40*mV)
npss = sl.tools.npss(tracemon[0], spikemon[0], 0*mV, Vth, tau, 2*ms)
print("Simulation done!")

print("Plots ...")
dt = brian.defaultclock.dt
winstarts = spikemon[0]-(2*ms)
winstarts_idx = (winstarts/dt).astype('int')
plt.figure()
plt.title("Membrane potential")
plt.plot(tracemon.times, tracemon[0])
plt.plot(tracemon.times[winstarts_idx], tracemon[0][winstarts_idx],
         'r.', markersize=10)

plt.figure()
plt.title("Kreuz and NPSS")
input_spiketrains = [[], []]
for spike in input_spiketimes:
    input_spiketrains[spike[0]].append(spike[1])
krint = sl.metrics.kreuz.interval(input_spiketrains,
                                  spikemon[0], mp=False)
plt.plot(krint[0], krint[1])
plt.plot(spikemon[0], npss)

plt.figure()
plt.title("Kreuz vs NPSS")
plt.scatter(krint[0], npss[1:])

integrals = interval_integral(tracemon[0], spikemon[0])
scintegrals = scaled_integral(tracemon[0], spikemon[0])
plt.figure()
plt.title("Integrals")
plt.plot(spikemon[0], integrals, label="Integral")
plt.plot(spikemon[0], scintegrals, label="Scaled intgrl")
plt.legend()

plt.ion()
plt.show()

