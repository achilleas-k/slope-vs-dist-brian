from __future__ import print_function
from brian import *
import spikerlib as sl


def gen_spikes(packet_seq):
    spikes = []
    for packet in packet_seq:
        pspread = packet["spread"]
        pt = packet["t"]
        pn = packet["n"]
        if pspread:
            times = np.random.normal(pt, pspread, pn)
        else:
            times = np.array([pt]*pn)
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
tau = 72*ms
neurons = NeuronGroup(1, "dV/dt = -V/tau : volt",
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
packet_sequence = [{"spread": 4.0*ms, "t": 0*ms, "n": Nin},
                   {"spread": 0.1*ms, "t": 0*ms, "n": Nin},
                   {"spread": 0.1*ms, "t": 0*ms, "n": Nin},
                   {"spread": 4.0*ms, "t": 0*ms, "n": Nin},
                   {"spread": 4.0*ms, "t": 0*ms, "n": Nin},
                   {"spread": 0.1*ms, "t": 0*ms, "n": Nin},
                   {"spread": 0.1*ms, "t": 0*ms, "n": Nin},
                   {"spread": 4.0*ms, "t": 0*ms, "n": Nin},
                   {"spread": 4.0*ms, "t": 0*ms, "n": Nin},
                   {"spread": 0.1*ms, "t": 0*ms, "n": Nin},
                   {"spread": 0.1*ms, "t": 0*ms, "n": Nin},
                   {"spread": 4.0*ms, "t": 0*ms, "n": Nin},
                   {"spread": 4.0*ms, "t": 0*ms, "n": Nin},
                   {"spread": 0.1*ms, "t": 0*ms, "n": Nin},
                   {"spread": 0.1*ms, "t": 0*ms, "n": Nin},
                   {"spread": 4.0*ms, "t": 0*ms, "n": Nin},
                   {"spread": 0.1*ms, "t": 0*ms, "n": Nin},
                   {"spread": 0.1*ms, "t": 0*ms, "n": Nin},
                   {"spread": 4.0*ms, "t": 0*ms, "n": Nin},
                   {"spread": 4.0*ms, "t": 0*ms, "n": Nin},
                   {"spread": 6.0*ms, "t": 0*ms, "n": Nin},
                   {"spread": 6.0*ms, "t": 0*ms, "n": Nin},
                   {"spread": 9.0*ms, "t": 0*ms, "n": Nin},
                   {"spread": 9.0*ms, "t": 0*ms, "n": Nin},
                  ]

inter_packet_interval = 50*ms  # a little shorter than the half-life with tau=20*ms
packet_times = [idx*inter_packet_interval+50*ms
                for idx in range(0, len(packet_sequence), 1)]
for ps, pt in zip(packet_sequence, packet_times):
    ps["t"] = pt
for spr in random(10):
    new_t = packet_sequence[-1]["t"]+inter_packet_interval
    new_spr = spr*10*ms
    packet_sequence.append({"spread": new_spr, "t": new_t, "n": Nin})
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
weight = 0.35*mV
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
                                  insert(outspikes[0], 0, 0), 1)
kreuz_t = [t for kt in kreuz_t for t in kt]
kreuz_d = [k for kd in kreuz_d for k in kd]

print("\tVictor-Purpura ...")
# calculate victor-purpura from monitors
vp_d = victor_purpura.interval(inputspikes,
                               insert(outspikes[0], 0, 0),
                               1/0.002)

print("\tModulus ...")
# calculate modulus metric
modm = modulus.interval(inputspikes, insert(outspikes[0], 0, 0), 0, float(dura))

print("\tNPSS ...")
# calculate NPSS
norm_slopes = []
for _idx in range(len(neurons)):
    norm_slopes.append(npss(trace_mon[_idx], insert(outspikes[_idx], 0, 0),
                            0*mV, Vth, tau, 2*ms))


print("Plotting ...")
# plot them
figure("Input spikes")
subplot2grid((5,1), (0, 0), rowspan=4)
raster_plot(input_mon, c='gray')
axis(xmin=0, xmax=dura/ms)
ax2 = twinx()
ax2.plot(array(ptimes)/ms, sigmas, color='black', linestyle='--', linewidth=1.5)
subplot2grid((5,1), (4, 0))
for out in outspikes[0]:
    plot([out/ms, out/ms], [0, Nin], 'k-', linewidth=1.5)
axis(xmin=0, xmax=dura/ms)


figure("Traces")
colours = ['b', 'r', 'g']
plot(trace_mon.times, trace_mon[0], colours[0])
plot([0*second, dura], [Vth, Vth], 'k--')
xlabel("$t$ (sec)")
ylabel("$V$ (volt)")


figure("Measures")
interval_midpoints = (outspikes[0]-diff(insert(outspikes[0], 0, 0))/2)*1000
subplot(511)
#title("Packet widths")
plot(multiply(ptimes, 1000), multiply(sigmas, 1000), 'k')
ylabel("$\sigma$ (ms)")
spike_height = max(sigmas)*1000
for out in outspikes[0]:
    plot([out/ms, out/ms], [0, spike_height], color="gray", linestyle="--", linewidth=1.5)
axis(xmin=0*second, xmax=dura*1000)
subplot(512)
#title("NPSS")
plot(interval_midpoints, norm_slopes[0], 'b')
ylabel("$NPSS$")
spike_height = max(norm_slopes[0])
for out in outspikes[0]:
    plot([out/ms, out/ms], [0, spike_height], color="gray", linestyle="--", linewidth=1.5)
axis(xmin=0*second, xmax=dura*1000)
subplot(513)
#title("Kreuz")
plot(interval_midpoints, kreuz_d, 'r')
ylabel("$D_{V}$")
spike_height = max(kreuz_d)
for out in outspikes[0]:
    plot([out/ms, out/ms], [0, spike_height], color="gray", linestyle="--", linewidth=1.5)
axis(xmin=0*second, xmax=dura*1000)
subplot(514)
#title("Victor-Purpura")
plot(interval_midpoints, vp_d, 'g')
ylabel("$D_{S}$")
spike_height = max(vp_d)
for out in outspikes[0]:
    plot([out/ms, out/ms], [0, spike_height], color="gray", linestyle="--", linewidth=1.5)
axis(xmin=0*second, xmax=dura*1000)
subplot(515)
#title("Modulus")
plot(interval_midpoints, modm, 'm')
spike_height = max(modm)
ylabel("$D_{m}$")
xlabel("$t$ (ms)")
for out in outspikes[0]:
    plot([out/ms, out/ms], [0, spike_height], color="gray", linestyle="--", linewidth=1.5)
axis(xmin=0*second, xmax=dura*1000)

ion()  # for interactive
show()

print("DONE!")



