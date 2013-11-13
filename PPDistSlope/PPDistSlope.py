"""

Step by step investigation of correlation between slope and input spike
train distance.  Instead of going all out with long simulations and playing
around with lots of data, this script will start simple and increase the
complexity to see whether the correlation exists on each level.

"""
from brian import *
import random
import gc
from warnings import warn
from spike_distance_mp import mean_pairwise_distance

def arrays_to_spiketimes(thearrays):
        '''
        Converts a 2D array of spike times to a format that works with
        brian.SpikeGeneratorGroup
        '''
        spiketimes = []
        for i, spiketrain in enumerate(thearrays):
            for spike in spiketrain:
                spiketimes.append((i, spike))
        return spiketimes


nprng = np.random
dcost = 1000  # cost 1000/second

# First, we'll generate a few pulse packets and calculate the distance on each
# one individually

packet_time = 0.5*second  # reduce the chance of negative spike times
N = 20  # spikes per packet
packets = []
sigmas = frange(0*ms, 4*ms, 0.1*ms)
samples = 1
randtrains = [0, 5, 10, 15, 20, 25, 30]
spikerates = frange(10*Hz, 100*Hz, 10*Hz)

print("Calculating distance of single pulse packets with background noise ...")
results = []
slope_w = 2*ms
for bg_rate in spikerates:
    interm_results = []
    for nrand in randtrains:
        for sigma in sigmas:
            for smpl in range(samples):
                print("nrand: %i, bg rate: %f, sigma: %f, sample: %i" % (
                    nrand, bg_rate, sigma, smpl))
                if sigma:
                    newpacket = nprng.normal(loc=packet_time, scale=sigma, size=N)
                else:
                    newpacket = ones(N)*packet_time
                lastspike = max(newpacket)
                newpacket = [[t] for t in newpacket]
                randspiketrains = []
                for nr in range(nrand):
                    poissontrain = []
                    poissspike = random.expovariate(bg_rate)
                    while poissspike <= 1:  # 1 second duration
                        poissontrain.append(poissspike)
                        poissspike += random.expovariate(bg_rate)
                    randspiketrains.append(poissontrain)
                allspikes = newpacket+randspiketrains  # must always be sure they are lists
                idist = mean_pairwise_distance(allspikes, 1000)
                spiketimes = arrays_to_spiketimes(allspikes)
                if not len(allspikes) == nrand + N:
                    warn("Whaaaat?! Something fucked up. Check spiketrains.")
                weight = 1.5*15*mV/len(allspikes)
                inputgroup = SpikeGeneratorGroup(len(allspikes), spiketimes)
                neuron = NeuronGroup(1, 'dV/dt = -V/(10*ms) : volt',
                        threshold='V>15*mV', reset='V=0*mV')
                conn = Connection(source=inputgroup, target=neuron,
                        state='V', weight=weight)
                neuron.V = 0*mV
                statemon = StateMonitor(neuron, 'V', record=True)
                spikemon = SpikeMonitor(neuron, record=True)
                netw = Network(inputgroup, neuron, conn, statemon, spikemon)
                netw.run(1.5*second)
                statemon.insert_spikes(spikemon, 15*mV)
                if len(spikemon[0]):
                    for outspike in spikemon[0]:
                        outspike *= second
                        spike_dt = outspike/(0.1*ms)
                        w_start = outspike-slope_w
                        w_start_dt = w_start/(0.1*ms)
                        v_w_start = statemon[0][w_start_dt]
                        slope = (15*mV-v_w_start*volt)/slope_w
                        interm_results.append((nrand, bg_rate, sigma, idist, slope))
                        results.append((nrand, bg_rate, sigma, idist, slope))
                else:
                    interm_results.append((nrand, bg_rate, sigma, idist, 0*mV/ms))
                    results.append((nrand, bg_rate, sigma, idist, 0*mV/ms))
                # clear everything related to brian
                del(inputgroup, neuron, conn, statemon, spikemon, netw)
                defaultclock.reinit()
                clear(True, True)
                gc.collect()

    rands, rates, sigs, dists, slopes = zip(*interm_results)
    scatter(slopes, dists, c=sigs)
    cbar = colorbar()
    xlabel('slopes')
    ylabel('spike distance')
    cbar.set_label('sigma')
    savefig("slope_vs_dist_rate_%i.png" % (bg_rate))
    clf()
    print("Saved figure for %i ..." % (bg_rate))

rands, rates, sigs, dists, slopes = zip(*results)
scatter(slopes, dists)
xlabel('slope')
ylabel('spike distance')
savefig("all_slope_vs_dist.png")

