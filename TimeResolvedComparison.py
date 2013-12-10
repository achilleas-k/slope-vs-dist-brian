from brian import *
import warnings
from spike_distance_mp import mean_pairwise_distance
from neurotools import npss, genInputGroups


def tuples_to_spiketrains(tuples):
    spiketrains = {}
    for (i, t) in tuples:
        try:
            spiketrains[i].append(t)
        except KeyError:
            spiketrains[i] = [t]
    return spiketrains


defaultclock.dt = dt = 0.1*ms
duration = 1*second

warnings.simplefilter("always")
slope_w = 2*msecond
dcost = float(2/slope_w)

N_in = 100
f_in = 50*Hz

# neuron
neuron = NeuronGroup(1, "dV/dt = -V/(10*ms) : volt",
        threshold="V>15*mvolt", reset="V=0*mvolt")
neuron.V = 0*mvolt

# inputs
syncInp, randInp = genInputGroups(N_in, f_in, 0.5, 0*ms, duration, dt)
syncCon = Connection(syncInp, neuron, "V", weight=0.2*mV)
randCon = Connection(randInp, neuron, "V", weight=0.2*mV)

# monitors
syncmon = SpikeMonitor(syncInp)
randmon = SpikeMonitor(randInp)
vmon = StateMonitor(neuron, "V", record=True)
outmon = SpikeMonitor(neuron)
netw = Network(neuron, vmon, outmon, syncInp, randInp, syncCon, randCon,
        syncmon, randmon)
# run
netw.run(duration, report="stdout")

# TODO: Collect input spikes, calculate npss, calculate mean pairwise distance
# between input spikes for:
# (a) 10 ms (leak time constant) prior to firing
# (b) entire inter-spike interval
# Do for both V-P and Kreuz

