"""

Step by step investigation of correlation between slope and input spike
train distance.  Instead of going all out with long simulations and playing
around with lots of data, this script will start simple and increase the
complexity to see whether the correlation exists on each level.

"""
from brian import *
import spike_distance_mp as spikedist

rng = np.random
dcost = 1000  # cost 1000/second

# First, we'll generate a few pulse packets and calculate the distance on each
# one individually

packet_time = 0.5*second  # reduce the chance of negative spike times
sigmas = arange(0*ms, 10*ms, 0.1*ms)
N = 20  # spikes per packet
packets = []
intradists = []

samples = 20
for sigma in sigmas:
    for smpl in range(samples):
        print("sigma: %f, sample: %i" % (sigma, smpl))
        if sigma:
            newpacket = rng.normal(loc=packet_time, scale=sigma, size=N)
        else:
            newpacket = ones(N)*packet_time
        newpacket = [[t] for t in newpacket]
        packets.append(newpacket)
        idist = spikedist.mean_pairwise_distance(newpacket, 1000)
        intradists.append((sigma, idist))

print("sigma\tdist")
for (s, d) in intradists:
    print("%f\t%f" % (s, d))

sigmas, dists = zip(*intradists)
scatter(sigmas, dists)
savefig("singlepacket.png")

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
