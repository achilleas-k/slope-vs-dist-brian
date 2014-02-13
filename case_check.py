from __future__ import print_function
from brian import *
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

netw = Network()
Vth = 20*mV
neuron = NeuronGroup(1, "dV/dt = 0*volt/second : volt",
                     threshold="V>Vth", reset="V=0*mV",
                     refractory=5*ms)
neuron.V = 0*mV
netw.add(neuron)

Nin = 20
input_idx = range(Nin)
input_times = []
# spikes caused by high early synchrony with low late synchrony
for strt in [50, 100, 150, 200]:
    new_its = (#[ni*ms for ni in np.random.normal(0, 0.01, 10)]+
               [0*ms]*10+
               [ni*ms for ni in np.random.normal(30, 3.0, 10)])
    input_times += [ni+strt*ms for ni in new_its]
# spikes caused by the opposite (high-early, low-late)
for strt in [250, 300, 350, 400, 450]:
    new_its = (#[ni*ms for ni in np.random.normal(30, 0.01, 10)]+
               [ni*ms for ni in np.random.normal(0, 3.0, 10)]+
               [30*ms]*10)
    input_times += [ni+strt*ms for ni in new_its]
# convert to (input, time) pairs
input_spikes = [(i, t) for i, t in zip(itt.cycle(input_idx), input_times)]
#print(input_spikes)
input_group = SpikeGeneratorGroup(Nin, input_spikes)
connection = Connection(input_group, neuron, "V", weight=1.0*mV)
netw.add(input_group, connection)

input_mon = SpikeMonitor(input_group)
output_mon = SpikeMonitor(neuron)
trace_mon = StateMonitor(neuron, "V", record=True)
netw.add(input_mon, output_mon, trace_mon)

netw.run(500*ms)

spiketimes = [0]+list(output_mon[0])
npss_slope = []
kreuz_dist = [[],[]]
kreuz_means = []
victor = []
for prev_st, spiketime in zip(spiketimes[:-1], spiketimes[1:]):
    print("%f --> %f" % (prev_st, spiketime))
    npss_slope.append(npss(trace_mon[0], array([prev_st, spiketime]),
                      0*mV, 15*mV, 10*ms, 2*ms)[0])

    input_spikes = input_mon.spiketimes.values()

    isi = spiketime-prev_st
    kreuz_bins = int(isi/(2*ms))
    t, krd = kreuz.interval(input_spikes, [prev_st, spiketime], kreuz_bins)
    kreuz_dist[0].extend(list(t[0]))
    kreuz_dist[1].extend(list(krd[0]))
    kreuz_means.append(mean(krd))

    victor.append(victor_purpura.interval(input_spikes,
                                          [prev_st, spiketime],
                                          float(1/(2*ms)))[0])


interval_midpoints = add(spiketimes[1:], spiketimes[:-1])/2
xmax=trace_mon.times[-1]
trace_mon.insert_spikes(output_mon, 40*mV)
subplot(3,1,1)
raster_plot(input_mon)
for st in spiketimes:
    plot([st/ms, st/ms], [-1, max(input_idx)+1], "k-")
axis(xmin=0, xmax=xmax/float(ms))
subplot(3,1,2)
plot(trace_mon.times, trace_mon[0], "k")
plot([0, xmax], [Vth, Vth], "k--")
subplot(3,1,3)
plot(interval_midpoints, npss_slope, "b-", label="NPSS")
plot(kreuz_dist[0], kreuz_dist[1], "r--", label="Kreuz")
plot(interval_midpoints, kreuz_means, "r-", label="Kreuz mean")
plot(interval_midpoints, victor, "g-", label="V-P")
metric_axes = axis()
for st in spiketimes:
    plot([st, st], [0, 10], "k-")
axis(metric_axes)
axis(xmin=0, xmax=xmax)
legend()
show()


