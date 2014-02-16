from __future__ import print_function
from brian import *
import spikerlib as sl


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
neurons = NeuronGroup(1, "dV/dt = 0*volt/second : volt",
                     threshold="V>Vth", reset="V=0*mV",
                     refractory=5*ms)
neurons.V = 0*mV
netw.add(neurons)

print("Generating inputs ... ")
Nin = 50
packet_sequence = [{"spread": 4.0*ms, "t":  50*ms, "n": Nin},
                   {"spread": 0.1*ms, "t": 100*ms, "n": Nin},
                   {"spread": 0.1*ms, "t": 150*ms, "n": Nin},
                   {"spread": 4.0*ms, "t": 200*ms, "n": Nin},
                   {"spread": 0.1*ms, "t": 250*ms, "n": Nin},
                   {"spread": 0.1*ms, "t": 300*ms, "n": Nin},
                   {"spread": 4.0*ms, "t": 350*ms, "n": Nin},
                   {"spread": 4.0*ms, "t": 400*ms, "n": Nin},
                   {"spread": 6.0*ms, "t": 450*ms, "n": Nin},
                   {"spread": 6.0*ms, "t": 500*ms, "n": Nin},
                   {"spread": 9.0*ms, "t": 550*ms, "n": Nin},
                   {"spread": 9.0*ms, "t": 600*ms, "n": Nin},
                  ]
ptimes = [ps["t"] for ps in packet_sequence]
dura = max(ptimes)+100*ms
sigmas = [ps["spread"] for ps in packet_sequence]

# (input, time) pairs
input_it_pairs = gen_spikes(packet_sequence)

# input groups
input_group = SpikeGeneratorGroup(Nin, input_it_pairs)
netw.add(input_group)

# connections
print("Setting up connections ...")
weight = 0.21*mV
in_con = Connection(input_group, neurons, state='V', weight=weight)
netw.add(in_con)

print("Setting up monitors ...")
# input monitors
input_mon = SpikeMonitor(input_group)
netw.add(input_mon)

# other monitors
trace_mon = StateMonitor(neurons, 'V', record=True)
output_mon = SpikeMonitor(neurons)
netw.add(trace_mon, output_mon)

print("Running ...")
netw.run(dura)
trace_mon.insert_spikes(output_mon, 40*mV)
print("Simulation run finished!")

print("Calculating measures ...")
outspikes = output_mon.spiketimes.values()
inputspikes = input_mon.spiketimes.values()

print("\tKreuz ...")
# calculate interval kreuz from monitors
kreuz_t, kreuz_d = kreuz.interval(inputspikes,
                                  insert(outspikes[0], 0, 0), 10)
kreuz_t = [t for kt in kreuz_t for t in kt]
kreuz_d = [k for kd in kreuz_d for k in kd]

print("\tVictor-Purpura ...")
# calculate victor-purpura from monitors
vp_d = victor_purpura.interval(inputspikes,
                               insert(outspikes[0], 0, 0),
                               1/0.002)

print("\tNPSS ...")
# calculate NPSS
norm_slopes = []
for _idx in range(len(neurons)):
    norm_slopes.append(npss(trace_mon[_idx], insert(outspikes[_idx], 0, 0),
                            0*mV, Vth, 10*ms, 2*ms))

print("Plotting ...")
# plot them
figure("Measures")
subplot(211)
raster_plot(input_mon, c='b')
for out in outspikes[0]:
    plot([out/ms, out/ms], [0, Nin], 'k--', linewidth=3)
axis(xmin=0, xmax=dura/ms)
ax2 = twinx()
ax2.plot(array(ptimes)/ms, sigmas, 'r--')

subplot(212)
plot(outspikes[0], norm_slopes[0], 'b')
plot(kreuz_t, kreuz_d, 'r')
spike_height = max(max(kreuz_d), max(norm_slopes[0]))
for out in outspikes[0]:
    plot([out, out], [0, spike_height], 'k--', linewidth=3)
ax2 = twinx()
ax2.plot(outspikes[0], vp_d, 'g')
axis(xmin=0*second, xmax=dura)

figure("Traces")
colours = ['b', 'r', 'g']
plot(trace_mon.times, trace_mon[0], colours[0])
plot([0*second, dura], [Vth, Vth], 'k--')

ion()  # for interactive
show()

print("DONE!")



