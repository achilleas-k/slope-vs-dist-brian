from __future__ import print_function
from brian import *
import spikerlib as sl


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
# input groups
Nin = 50
Sin = 0.7
dura = 1*second
inrate = 20*Hz
input_spike_times = []
for t in arange(0*ms, dura, 0.1*ms):
    if random() < float(1/inrate):
        # new event
        if random() < Sin:
            # event is synchronous
            input_spike_times.extend([t]*int(Nin*Sin))
        else:
            # event is single spike
            input_spike_times.append(t)
# add spike indices
input_it_pairs = [(idx % Nin, t) for idx, t in enumerate(input_spike_times)]

# input groups
input_group = SpikeGeneratorGroup(Nin, input_it_pairs)
netw.add(input_group)

# connections
print("Setting up connections ...")
weight = 0.5*mV
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
subplot2grid((5,1), (4, 0))
for out in outspikes[0]:
    plot([out/ms, out/ms], [0, Nin], 'k-', linewidth=1.5)
axis(xmin=0, xmax=dura/ms)


figure("Traces")
plot(trace_mon.times, trace_mon[0], 'k')
plot([0*second, dura], [Vth, Vth], 'k--')
xlabel("$t$ (sec)")
ylabel("$V$ (volt)")

figure("Measures")
interval_midpoints = (outspikes[0]-diff(insert(outspikes[0], 0, 0))/2)*1000
subplot(511)
#title("Packet widths")
ylabel("$\sigma$ (ms)")
spike_height = 1
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



