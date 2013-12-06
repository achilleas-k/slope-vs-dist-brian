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
duration = 0.2*second
# cost of moving a spike should be < 2 for moving within w
# and equal 2 when moving across a time w
dcost = 2/slope_w

# First, we'll generate a few pulse packets and calculate the distance on each
# one individually

packet_time = 0.1*second  # reduce the chance of negative spike times
N = 20  # spikes per packet
packets = []
sigmas = [0*ms, 1*ms, 2*ms, 3*ms, 4*ms]
samples = 30
randtrains = [20]
spikerates = frange(20*Hz, 100*Hz, 40*Hz)

print("Calculating distance of single pulse packets with background noise ...")
results = []
averages = []
for bg_rate in spikerates:
    bg_rate*=Hz
    interm_results = []
    for nrand in randtrains:
        for sigma in sigmas:
            for smpl in range(samples):
                print("nrand: %i, bg rate: %f, sigma: %f, sample: %i" % (
                    nrand, bg_rate, sigma, smpl))
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
                    # start random spike trains 10 ms before packet
                    poissspike = random.expovariate(bg_rate)+0.09*second
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
                        threshold='V>15*mV', refractory=2*ms,
                        #reset='V=0*mV',
                        )
                conn = Connection(source=inputgroup, target=neuron,
                        state='V', weight=weight)
                neuron.V = 0*mV
                statemon = StateMonitor(neuron, 'V', record=True)
                spikemon = SpikeMonitor(neuron, record=True)
                netw = Network(inputgroup, neuron, conn, statemon,
                        spikemon, stop_condition)
                netw.run(duration)
                #statemon.insert_spikes(spikemon, 15*mV)
                act_dura = defaultclock.t
                # use only input spikes that arrived before the output
                # spike
                inputspikes = [[aspike for aspike in atrain if aspike*second <= act_dura+dt*0.5] for atrain in allspikes]
                # remove empty spike trains
                inputspikes = [atrain for atrain in inputspikes if atrain]
                if len(spikemon[0]) > 1:
                    warnings.warn("More than one spike fired.")
                if len(spikemon[0]):
                    idist = mean_pairwise_distance(inputspikes, 1000)
                    for outspike in spikemon[0]:
                        outspike *= second
                        spike_dt = outspike/dt
                        w_start = outspike-slope_w
                        w_start_dt = w_start/dt
                        v_w_start = statemon[0][w_start_dt]
                        slope = (15*mV-v_w_start*volt)/slope_w
                        # new stuff here
                        spikepeak = statemon[0][spike_dt]
                        results.append((nrand, bg_rate, sigma, idist, slope, weight, len(inputspikes), spikepeak))
                        print("\t\tdist: %f, slope: %f" % (idist, slope))
                        if slope < 0:
                            warnings.warn("Negative slope - this is a problem!")
                        cur_slope.append(slope)
                        cur_dist.append(idist)
                else:
                    print("\t\tNo spike fired.")

                # clear everything related to brian
                del(inputgroup, neuron, conn, statemon, spikemon, netw)

rands, rates, sigs, dists, slopes, weights, ninputs, spikepeak = zip(*results)
scatter(slopes, dists)
xlabel('slope')
ylabel('spike distance')
savefig("all_slope_vs_dist.png")
np.savez("singlepacket_noclip.npz",
        Nrand=rands,
        randrate=rates,
        sigma=sigs,
        dist=dists,
        slope=slopes,
        weight=weights,
        ninputs=ninputs,
        spikepeak=spikepeak)

