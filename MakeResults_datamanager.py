from brian import *
import sys
import neurotools as nt
from brian.tools.datamanager import *
import spike_distance_mp as sdist

if __name__=='__main__':
    data_directory = sys.argv[1]
    if data_directory.endswith('/'):
        data_directory = data_directory.replace('/', '')
    if data_directory.endswith('.data'):
        data = DataManager(data_directory.replace('.data', ''))
        mem_all = []
        spikes_all = []
        inputs_all = []
        for item in data.itervalues():
            mem_all.append(item['mem'])
            spikes_all.append(item['spikes'].values())
            inputs_all.append(item['input_spikes'])
    w=2*ms
    slopes = []
    distances = []
    num_items = len(mem_all)
    count = 0
    for memset, spikeset, inputset in zip(mem_all, spikes_all, inputs_all):
        count += 1
        print("Processing item %i of %i ..." % (count, num_items))
        slopes_set = []
        for mem, spikes in zip(memset, spikeset):
            avg, _ign = nt.norm_firing_slope(mem, spikes, 15*mV, 10*ms, w=w)
            slopes_set.append(avg)
        slopes.append(slopes_set)
        distances_set = []
        for spiketrains in inputset:
            d = sdist.mean_pairwise_distance(spiketrains, cost=1000)
            distances_set.append(d)
        distances.append(distances_set)
        print("Processing of %i complete!" % (count))
    print("All done. Saving data ...")
    np.savez('results.npz',
                distances=distances,
                slopes=slopes)
    print("DONE!")


