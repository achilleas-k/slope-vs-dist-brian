from __future__ import print_function
from brian import *
import spikerlib as sl
import itertools as itt


def gen_spikes(packet_seq):
    spikes = []
    for packet in packet_seq:
        pspread = packet["spread"]
        pt = packet["t"]
        pn = packet["n"]
        times = np.random.normal(pt, pspread, pn)
        for _idx, t in zip(range(pn), times):
            spikes.append((_idx, t))
    return spikes

# unit aliases
ms = msecond
Hz = hertz
mV = mvolt

# measure aliases
victor_purpura = sl.metrics.victor_purpura
kreuz = sl.metrics.kreuz
modulus = sl.metrics.modulus_metric
npss = sl.tools.normalised_pre_spike_slopes

print("Creating NeuronGroup ...")
netw = Network()
Vth = 20*mV
neurons = NeuronGroup(3, "dV/dt = 0*volt/second : volt",
                     threshold="V>Vth", reset="V=0*mV",
                     refractory=5*ms)
neurons.V = 0*mV
netw.add(neurons)

print("Generating inputs ... ")
Nin = 50
wide_packet = np.random.normal(0, 4.0, 50)
narrow_packet = np.random.normal(0, 0.1, 50)

#wide_narrow = ([ni*ms+15*ms for ni in wide_packet]+
#               [ni*ms+40*ms for ni in narrow_packet])
narrow_wide = ([ni*ms+15*ms for ni in narrow_packet]+
               [ni*ms+40*ms for ni in wide_packet])

#wn_trains = [[] for i in range(Nin)]
#for wn, t in zip(itt.cycle(wn_trains), wide_narrow):
#    wn.append(t)
nw_trains = [[] for i in range(Nin)]
for nw, t in zip(itt.cycle(nw_trains), narrow_wide):
    nw.append(t)

packet_sequence = [{"spread": 4.0*ms, "t": 50*ms, "n": 50},
                   {"spread": 0.1*ms, "t": 100*ms, "n": 50},
                   {"spread": 0.1*ms, "t": 150*ms, "n": 50},
                   {"spread": 6.0*ms, "t": 200*ms, "n": 50},
                   {"spread": 0.1*ms, "t": 250*ms, "n": 50},
                   {"spread": 0.1*ms, "t": 300*ms, "n": 50},
                   {"spread": 3.0*ms, "t": 350*ms, "n": 50},
                   {"spread": 3.0*ms, "t": 400*ms, "n": 50},
                  ]
dura = 500*ms
#wn_t, wn_kreuz = kreuz.pairwise_mp(wn_trains, 0*ms, dura, 100)
nw_t, nw_kreuz = kreuz.pairwise_mp(nw_trains, 0*ms, dura, 100)

# random spike trains for comparison
randspiketrains = [np.cumsum(np.random.random(10)) for i in range(50)]
randspiketrains = [st*dura/max(st) for st in randspiketrains]
rand_t, rand_kreuz = kreuz.pairwise_mp(randspiketrains, 0*ms, dura, 100)

# (input, time) pairs
wn_spikes = gen_spikes(packet_sequence)
#wn_spikes = [(i, t) for i, t_train in enumerate(wn_trains) for t in t_train]
nw_spikes = [(i, t) for i, t_train in enumerate(nw_trains) for t in t_train]
rand_spikes = [(i, t) for i, t_train in enumerate(randspiketrains) for t in t_train]

# input groups
wn_ingroup = SpikeGeneratorGroup(Nin, wn_spikes)
nw_ingroup = SpikeGeneratorGroup(Nin, nw_spikes)
rand_ingroup = SpikeGeneratorGroup(Nin, rand_spikes)
netw.add(wn_ingroup, nw_ingroup, rand_ingroup)

# connections
print("Setting up connections ...")
weight = 0.21*mV
wn_con = Connection(wn_ingroup, neurons[0], state='V', weight=weight)
nw_con = Connection(nw_ingroup, neurons[1], state='V', weight=weight)
rand_con = Connection(rand_ingroup, neurons[2], state='V', weight=weight)
netw.add(wn_con, nw_con, rand_con)

print("Setting up monitors ...")
# input monitors
wn_monitor = SpikeMonitor(wn_ingroup)
nw_monitor = SpikeMonitor(nw_ingroup)
rand_monitor = SpikeMonitor(rand_ingroup)
netw.add(wn_monitor, nw_monitor, rand_monitor)

# other monitors
trace_mon = StateMonitor(neurons, 'V', record=True)
output_mon = SpikeMonitor(neurons)
netw.add(trace_mon, output_mon)

print("Running ...")
netw.run(dura)
trace_mon.insert_spikes(output_mon, 40*mV)
print("Simulation run finished!")

print("Calculating measures ...")
# plot
#figure("measures")
#subplot(211)
#plot(wn_t, wn_kreuz, 'b')
#plot(nw_t, nw_kreuz, 'r')
#plot(rand_t, rand_kreuz, 'k')
#axis(xmin=0*second, xmax=dura)
#figure("normed")
#norm_factor = max(rand_kreuz)
#plot(wn_t, wn_kreuz/norm_factor, 'b')
#plot(nw_t, nw_kreuz/norm_factor, 'r')
#plot(rand_t, rand_kreuz/norm_factor, 'g')
#axis(xmin=0*second, xmax=dura)

outspikes = output_mon.spiketimes.values()
wn_inputs = wn_monitor.spiketimes.values()
nw_inputs = nw_monitor.spiketimes.values()
rand_inputs = rand_monitor.spiketimes.values()

# calculate interval kreuz from monitors
wn_int_t, wn_int_kreuz = kreuz.interval(wn_inputs,
                                        insert(outspikes[0], 0, 0), 100)
wn_int_t = [t for wit in wn_int_t for t in wit]
wn_int_kreuz = [k for wik in wn_int_kreuz for k in wik]

nw_int_t, nw_int_kreuz = kreuz.interval(nw_inputs,
                                        insert(outspikes[1], 0, 0), 100)
nw_int_t = [t for nit in nw_int_t for t in nit]
nw_int_kreuz = [k for nik in nw_int_kreuz for k in nik]

rand_int_t, rand_int_kreuz = kreuz.interval(rand_inputs,
                                            insert(outspikes[2], 0, 0), 100)
rand_int_t = [t for rit in rand_int_t for t in rit]
rand_int_kreuz = [k for rik in rand_int_kreuz for k in rik]

# calculate victor-purpura from monitors
wn_vp = victor_purpura.interval(wn_inputs,
                                insert(outspikes[0], 0, 0),
                                1/0.002)
nw_vp = victor_purpura.interval(nw_inputs,
                                insert(outspikes[1], 0, 0),
                                1/0.002)
rand_vp = victor_purpura.interval(rand_inputs,
                                  insert(outspikes[2], 0, 0),
                                  1/0.002)

# calculate NPSS
norm_slopes = []
for _idx in range(len(neurons)):
    norm_slopes.append(npss(trace_mon[_idx], insert(outspikes[_idx], 0, 0),
                            0*mV, Vth, 10*ms, 2*ms))

print("Plotting ...")
# plot them
figure("Measures")
subplot(311)
raster_plot(wn_monitor, c='b')
raster_plot(nw_monitor, c='r')
raster_plot(rand_monitor, c='g')
axis(xmin=0, xmax=dura/ms)

subplot(312)
plot(wn_int_t, wn_int_kreuz, 'b')
for out in outspikes[0]:
    plot([out, out], [0, 0.2], 'b--')
plot(outspikes[0], wn_vp, 'm')

plot(nw_int_t, nw_int_kreuz, 'r')
for out in outspikes[1]:
    plot([out, out], [0, 0.2], 'r--')

plot(rand_int_t, rand_int_kreuz, 'g')
for out in outspikes[2]:
    plot([out, out], [0, 0.2], 'g--')
axis(xmin=0*second, xmax=dura)

subplot(313)
plot(outspikes[0], norm_slopes[0], 'b')
for out in outspikes[0]:
    plot([out, out], [0, 0.2], 'b--')

plot(outspikes[1], norm_slopes[1], 'r')
for out in outspikes[1]:
    plot([out, out], [0, 0.2], 'r--')

plot(outspikes[2], norm_slopes[2], 'g')
for out in outspikes[2]:
    plot([out, out], [0, 0.2], 'g--')
axis(xmin=0*second, xmax=dura)

figure("Traces")
colours = ['b', 'r', 'g']
for _idx in range(len(neurons)):
    plot(trace_mon.times, trace_mon[_idx], colours[_idx])
plot([0*second, dura], [Vth, Vth], 'k--')

ion()  # for interactive
show()

print("DONE!")



