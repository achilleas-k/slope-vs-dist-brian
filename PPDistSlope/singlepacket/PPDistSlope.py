"""

Step by step investigation of correlation between slope and input spike
train distance.  Instead of going all out with long simulations and playing
around with lots of data, this script will start simple and increase the
complexity to see whether the correlation exists on each level.

"""
from brian import *
import random
import gc
import warnings
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

warnings.simplefilter('always')
nprng = np.random
slope_w = 2*ms
defaultclock.dt = dt = 0.1*ms
duration = 1*second
# cost of moving a spike should be < 2 for moving within w
# and equal 2 when moving across a time w
dcost = 2/slope_w

# First, we'll generate a few pulse packets and calculate the distance on each
# one individually

packet_time = 0.5*second  # reduce the chance of negative spike times
N = 20  # spikes per packet
packets = []
sigmas = [0*ms]
samples = 10
randtrains = [20]
spikerates = frange(10*Hz, 100*Hz, 20*Hz)

print("Calculating distance of single pulse packets with background noise ...")
results = []
for bg_rate in spikerates:
    bg_rate*=Hz
    interm_results = []
    for nrand in randtrains:
        for sigma in sigmas:
            for smpl in range(samples):
                defaultclock.reinit()
                clear(True, True)
                gc.collect()
                # need to re-define stop condition after deleting
                # everything
                @network_operation
                def stop_condition(clock):
                    if spikemon.nspikes:
                        stop()

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
                    while poissspike <= duration:
                        poissontrain.append(float(poissspike))
                        poissspike += random.expovariate(bg_rate)
                    randspiketrains.append(poissontrain)
                allspikes = randspiketrains+newpacket
                spiketimes = arrays_to_spiketimes(allspikes)
                if not len(allspikes) == nrand + N:
                    warningsd.warn("Check spiketrains.")
                weight = 1.5*15*mV/N
                inputgroup = SpikeGeneratorGroup(len(allspikes), spiketimes)
                # TODO: no reset - no clipping - Jacob's suggestion
                neuron = NeuronGroup(1, 'dV/dt = -V/(10*ms) : volt',
                        threshold='V>15*mV', reset='V=0*mV', refractory=2*ms)
                conn = Connection(source=inputgroup, target=neuron,
                        state='V', weight=weight)
                neuron.V = 0*mV
                statemon = StateMonitor(neuron, 'V', record=True)
                spikemon = SpikeMonitor(neuron, record=True)
                netw = Network(inputgroup, neuron, conn, statemon,
                        spikemon, stop_condition)
                netw.run(duration)
                statemon.insert_spikes(spikemon, 15*mV)
                if len(spikemon[0]) > 1:
                    warnings.warn("More than one spike fired.")
                print("nrand: %i, bg rate: %f, sigma: %f, sample: %i" % (
                    nrand, bg_rate, sigma, smpl))
                if len(spikemon[0]):
                    idist = mean_pairwise_distance(allspikes, 1000)

                    for outspike in spikemon[0]:
                        outspike *= second
                        spike_dt = outspike/(0.1*ms)
                        w_start = outspike-slope_w
                        w_start_dt = w_start/(0.1*ms)
                        v_w_start = statemon[0][w_start_dt]
                        slope = (15*mV-v_w_start*volt)/slope_w
                        interm_results.append((nrand, bg_rate, sigma,
                            idist, slope, weight))
                        results.append((nrand, bg_rate, sigma, idist, slope, weight))
                        print("\t\tdist: %f, slope: %f" % (idist, slope))
                        if slope < 0:
                            warnings.warn("Negative slope - this is a problem!")
                else:
                    print("\t\tNo spike fired.")
                # clear everything related to brian
                del(inputgroup, neuron, conn, statemon, spikemon, netw)

    if interm_results == 0:
        print("No results so far - skipping plot")
        continue
    rands, rates, sigs, dists, slopes, weight = zip(*interm_results)
    scatter(slopes, dists)#, c=sigs)
    #cbar = colorbar()
    xlabel('slopes')
    ylabel('spike distance')
    #cbar.set_label('sigma')
    savefig("slope_vs_dist_rate_%i.png" % (bg_rate))
    clf()
    print("Saved figure for %i ..." % (bg_rate))

rands, rates, sigs, dists, slopes, weights = zip(*results)
scatter(slopes, dists)
xlabel('slope')
ylabel('spike distance')
savefig("all_slope_vs_dist.png")
np.savez("singlepacket.npz",
        Nrand=rands,
        randrate=rates,
        sigma=sigs,
        dist=dists,
        slope=slopes,
        weight=weights)

