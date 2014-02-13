from __future__ import print_function
from brian import *
import numpy as np
import spikerlib as sl
import itertools as itt


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
neurons = NeuronGroup(3, "dV/dt = 0*volt/second : volt",
                     threshold="V>Vth", reset="V=0*mV",
                     refractory=5*ms)
neurons.V = 0*mV
netw.add(neurons)

Nin = 50
wide_packet = np.random.normal(0, 4.0, 50)
narrow_packet = np.random.normal(0, 0.1, 50)
wide_narrow = ([ni*ms+15*ms for ni in wide_packet]+
               [ni*ms+40*ms for ni in narrow_packet])
narrow_wide = ([ni*ms+15*ms for ni in narrow_packet]+
               [ni*ms+40*ms for ni in wide_packet])

wn_trains = [[] for i in range(Nin)]
for wn, t in zip(itt.cycle(wn_trains), wide_narrow):
    wn.append(t)
nw_trains = [[] for i in range(Nin)]
for nw, t in zip(itt.cycle(nw_trains), narrow_wide):
    nw.append(t)

dura = max(max(wide_narrow), max(narrow_wide), 50*ms)
wn_t, wn_kreuz = kreuz.pairwise_mp(wn_trains, 0*ms, dura, 100)
nw_t, nw_kreuz = kreuz.pairwise_mp(nw_trains, 0*ms, dura, 100)

# random spike trains for comparison
randspiketrains = [np.cumsum(np.random.random(10)) for i in range(50)]
randspiketrains = [st*dura/max(st) for st in randspiketrains]
rand_t, rand_kreuz = kreuz.pairwise_mp(randspiketrains, 0*ms, dura, 100)

# (input, time) pairs
wn_spikes = [(i, t) for i, t_train in enumerate(wn_trains) for t in t_train]
nw_spikes = [(i, t) for i, t_train in enumerate(nw_trains) for t in t_train]
rand_spikes = [(i, t) for i, t_train in enumerate(randspiketrains) for t in t_train]

# input groups
wn_ingroup = SpikeGeneratorGroup(Nin, wn_spikes)
nw_ingroup = SpikeGeneratorGroup(Nin, nw_spikes)
rand_ingroup = SpikeGeneratorGroup(Nin, rand_spikes)
netw.add(wn_ingroup, nw_ingroup, rand_ingroup)

# connections
wn_con = Connection(wn_ingroup, neurons[0], state='V', weight=0.5*mV)
nw_con = Connection(nw_ingroup, neurons[1], state='V', weight=0.5*mV)
rand_con = Connection(rand_ingroup, neurons[2], state='V', weight=0.5*mV)
netw.add(wn_con, nw_con, rand_con)

# input monitors
wn_monitor = SpikeMonitor(wn_ingroup)
nw_monitor = SpikeMonitor(nw_ingroup)
rand_monitor = SpikeMonitor(rand_ingroup)
netw.add(wn_monitor, nw_monitor, rand_monitor)

# other monitors
trace_mon = StateMonitor(neurons, 'V', record=True)
output_mon = SpikeMonitor(neurons)
netw.add(trace_mon, output_mon)

netw.run(dura)
trace_mon.insert_spikes(output_mon, 40*mV)

# plot
figure()
plot(wn_t, wn_kreuz, 'b')
plot(nw_t, nw_kreuz, 'r')
plot(rand_t, rand_kreuz, 'k')
figure()
norm_factor = max(rand_kreuz)
plot(wn_t, wn_kreuz/norm_factor, 'b')
plot(nw_t, nw_kreuz/norm_factor, 'r')
plot(rand_t, rand_kreuz/norm_factor, 'k')

# calculate interval kreuz from monitors
outspikes = output_mon.spiketimes.values()
wn_inputs = wn_monitor.spiketimes.values()
nw_inputs = nw_monitor.spiketimes.values()
rand_inputs = rand_monitor.spiketimes.values()

wn_int_t, wn_int_kreuz = kreuz.interval(wn_inputs,
                                        insert(outspikes[0], 0, 0), 10)
wn_int_t = [t for wit in wn_int_t for t in wit]
wn_int_kreuz = [k for wik in wn_int_kreuz for k in wik]

nw_int_t, nw_int_kreuz = kreuz.interval(nw_inputs,
                                        insert(outspikes[1], 0, 0), 10)
nw_int_t = [t for nit in nw_int_t for t in nit]
nw_int_kreuz = [k for nik in nw_int_kreuz for k in nik]

rand_int_t, rand_int_kreuz = kreuz.interval(rand_inputs,
                                            insert(outspikes[2], 0, 0), 10)
rand_int_t = [t for rit in rand_int_t for t in rit]
rand_int_kreuz = [k for rik in rand_int_kreuz for k in rik]

# plot them
figure()
plot(wn_int_t, wn_int_kreuz, 'b')
plot(nw_int_t, nw_int_kreuz, 'r')
plot(rand_int_t, rand_int_kreuz, 'k')

ion()  # for interactive
show()

