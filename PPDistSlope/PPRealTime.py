from brian import *
import random

def arrays_to_spiketimes(thearrays):
    spiketimes = []
    for i, spiketrain in enumerate(thearrays):
        for spike in spiketrain:
            spiketimes.append((i, spike))
    return spiketimes


defaultclock.dt = dt = 0.1*ms
duration = 1*second

nprng = np.random

packet_times = [0.2*second, 0.5*second, 0.8*second]
N = 40  # spikes per packet
packets = []
sigma = 1*ms
nrand = 5
spikerates = 10*Hz

print("Setting up inputs ...")
# generate pulse packets
packets = []
for ptime in packet_times:
    if sigma:
        newpacket = nprng.normal(loc=ptime, scale=sigma, size=N)
    else:
        newpacket = ones(N)*ptime
    if len(packets):
        packets = [p+[np] for p, np in zip(packets, newpacket)]
    else:
        packets = [[t] for t in newpacket]

# generate random spike trains
randtrains = []
for nr in range(nrand):
    poissontrain = []
    poissonspike = random.expovariate(spikerates)
    while poissonspike <= duration:
        poissontrain.append(poissonspike)
        poissonspike += random.expovariate(spikerates)
    randtrains.append(poissontrain)
allspikes = randtrains+packets
spiketimes = arrays_to_spiketimes(allspikes)
if not len(allspikes) == nrand+N:
    warnings.warn("Something went horribly wrong. Check spike train generation")

print("Setting up simulation ...")
weight = 1.5*15*mV/len(allspikes)
inputgroup = SpikeGeneratorGroup(len(allspikes), spiketimes)
neuron = NeuronGroup(1, 'dV/dt = -V/(10*ms) : volt',
        threshold='V>15*mV', reset='V=0*mV', refractory=2*ms)
conn = Connection(source=inputgroup, target=neuron,
        state='V', weight=weight)
neuron.V = 0*mV

inpmon = SpikeMonitor(inputgroup, record=True)
vmon = RecentStateMonitor(neuron, 'V', record=True, duration=200*ms)
outmon = SpikeMonitor(neuron, record=True)

ion()
subplot(211)
raster_plot(inpmon, refresh=2*ms, showlast=200*ms, redraw=False)
subplot(212)
vmon.plot(refresh=2*ms, showlast=200*ms)

print("Running ...")

run(duration)

ioff() # switch interactive mode off
show() # and wait for user to close the window before shutting down


