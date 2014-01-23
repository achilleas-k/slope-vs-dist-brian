from brian import *
import numpy as np
import sys
import spike_distance_mp as sdist
import neurotools as nt

if __name__=='__main__':
    data_file = sys.argv[1]
    mem_all = []
    spikes_all = []
    inputs_all = []
    print("Loading data from %s ..." % (data_file))
    data = np.load(data_file)
    mem = data['mem']
    spikes = data['spikes']
    inputs = data['input_spikes']
    print("%s loaded ..." % (data_file))
    w=2*ms
    slopes = []
    distances = []
    num_items = len(mem)
    count = 0
    spikes = spikes.reshape(1,1)[0][0] # don't ask
    for mem_idx, spikes_idx in zip(mem, spikes.itervalues()):
        count += 1
        print("Processing slopes of item %i of %i ..." % (count, num_items))
        avg, _ign = nt.norm_firing_slope(mem_idx, spikes_idx,
                15*mV, 10*ms, w=w)
        slopes.append(avg)
    distances = []
    count = 0
    for spiketrains in inputs:
        count += 1
        print("Processing spiketrain dist of item %i of %i ..." % (count, num_items))
        d = sdist.mean_pairwise_distance(spiketrains, cost=1000)
        distances.append(d)
    print("All done. Saving data ...")
    np.savez('results.npz',
                distances=distances,
                slopes=slopes)
    print("DONE!")


