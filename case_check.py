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
lif_neuron = NeuronGroup(1, "dV/dt = -V/(10*ms) : volt",
                         threshold="V>15*mV", reset="V=0*mV")
lif_neuron.V = 0*mV
netw.add(lif_neuron)

Nin = 30
input_idx = range(10)
input_times = [10*ms]*5+list(arange(20, 35)*ms)
input_times.extend(array([150*ms]*10))
#input_times.extend(array([200*ms]*10)-rand(10)*10*ms)
#input_times.extend(array([300*ms]*20)-rand(20)*30*ms)
input_spikes = [(i, t) for i, t in zip(itt.cycle(input_idx), input_times)]
#print(input_spikes)
input_group = SpikeGeneratorGroup(Nin, input_spikes)
connection = Connection(input_group, lif_neuron, "V", weight=2.3*mV)
netw.add(input_group, connection)

input_mon = SpikeMonitor(input_group)
output_mon = SpikeMonitor(lif_neuron)
trace_mon = StateMonitor(lif_neuron, "V", record=True)
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
    kreuz_bins = int(isi/ms)
    t, krd = kreuz.interval(input_spikes, [prev_st, spiketime], kreuz_bins)
    kreuz_dist[0].extend(list(t[0]))
    kreuz_dist[1].extend(list(krd[0]))
    kreuz_means.append(mean(krd))

    victor.append(victor_purpura.interval(input_spikes,
                                          [prev_st, spiketime],
                                          float(1/(2*ms)))[0])


interval_midpoints = add(spiketimes[1:], spiketimes[:-1])/2
xmax=trace_mon.times[-1]
trace_mon.insert_spikes(output_mon, 20*mV)
subplot(3,1,1)
raster_plot(input_mon)
for st in spiketimes:
    plot([st/ms, st/ms], [-1, max(input_idx)+1], "k-")
axis(xmin=0, xmax=xmax/float(ms))
subplot(3,1,2)
plot(trace_mon.times, trace_mon[0], "k")
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

