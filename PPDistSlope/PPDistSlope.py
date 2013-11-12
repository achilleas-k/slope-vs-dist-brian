"""

Step by step investigation of correlation between slope and input spike
train distance.  Instead of going all out with long simulations and playing
around with lots of data, this script will start simple and increase the
complexity to see whether the correlation exists on each level.

"""
from brian import *
import random
from spike_distance_mp import mean_pairwise_distance

nprng = np.random
dcost = 1000  # cost 1000/second

# First, we'll generate a few pulse packets and calculate the distance on each
# one individually

packet_time = 0.5*second  # reduce the chance of negative spike times
sigmas = frange(0*ms, 4*ms, 0.1*ms)
N = 20  # spikes per packet
packets = []
intradists = []

print("Calculating distance of single pulse packets without background noise ...")
#samples = 20
#for sigma in sigmas:
#    for smpl in range(samples):
#        print("sigma: %f, sample: %i" % (sigma, smpl))
#        if sigma:
#            newpacket = nprng.normal(loc=packet_time, scale=sigma, size=N)
#        else:
#            newpacket = ones(N)*packet_time
#        newpacket = [[t] for t in newpacket]
#        packets.append(newpacket)
#        idist = mean_pairwise_distance(newpacket, 1000)
#        intradists.append((sigma, idist))
#sigmas, dists = zip(*intradists)
#scatter(sigmas, dists)
#savefig("singlepacket.png")
print("Adding noise ...")
intradists = []
samples = 1
randtrains = [10]
spikerates = frange(10*Hz, 100*Hz, 10*Hz)
for nrand in randtrains:
    for bg_rate in spikerates:
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
                    while poissspike <= lastspike:
                        poissontrain.append(poissspike)
                        poissspike += random.expovariate(bg_rate)
                    randspiketrains.append(poissontrain)
                allspikes = newpacket+randspiketrains  # must always be sure they are lists
                idist = mean_pairwise_distance(allspikes, 1000)
                intradists.append((bg_rate, sigma, idist))
    rates, sigmas, dists = zip(*intradists)
    scatter(sigmas, dists, c=rates)
    colorbar()
    savefig("singlepacket_noise_nrand_%i.png" % (nrand))
    intradists = []
    clf()
    print("Saved figure for %i ..." % (nrand))

"""
plot and save

add noise

plot and save

multiple packets

plot and save

add noise

plot and save

go back and include LIF neuron and calc slope
"""
