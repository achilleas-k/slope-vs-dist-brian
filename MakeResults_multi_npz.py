from brian import *
import spike_distance_mp as sdist
import neurotools as nt
import os
import zipfile

if __name__=='__main__':
    data_dest = 'unziped_data'
    data_zip_filename = sys.argv[1]
    zip_file = zipfile.ZipFile(data_zip_filename)
    zip_file.extractall(data_dest)
    data_dir = os.path.join(data_dest, 'data')
    data_files = os.listdir(data_dir)
    mem_all = []
    spikes_all = []
    inputs_all = []
    print("Loading data from %s ..." % (data_dir))
    for df in data_files:
        data = np.load(os.path.join(data_dir, df))
        mem_all.append(data['mem'])
        spikes_all.append(data['spikes'])
        inputs_all.append(data['input_spikes'])
        print("%s loaded ..." % (df))
    w=2*ms
    slopes = []
    distances = []
    num_items = len(mem_all)
    count = 0
    for memset, spikeset, inputset in zip(mem_all, spikes_all, inputs_all):
        count += 1
        print("Processing item %i of %i ..." % (count, num_items))
        slopes_set = []
        spikeset = spikeset.reshape(1,1)[0][0] # don't ask
        num_items_sub = len(memset)
        count_sub = 0
        for mem, spikes in zip(memset, spikeset.itervalues()):
            count_sub += 1
            print("Processing slopes of child %i of %i in (item %i / %i) ..." % (
                count_sub, num_items_sub, count, num_items))
            avg, _ign = nt.norm_firing_slope(mem, spikes, 15*mV, 10*ms, w=w)
            slopes_set.append(avg)
        slopes.append(slopes_set)
        distances_set = []
        count_sub = 0
        for spiketrains in inputset:
            count_sub += 1
            print("Processing dist of child %i of %i in (item %i / %i) ..." % (
                count_sub, num_items_sub, count, num_items))
            d = sdist.mean_pairwise_distance(spiketrains, cost=1000)
            distances_set.append(d)
        distances.append(distances_set)
        print("Processing of %i complete!" % (count))
    print("All done. Saving data ...")
    np.savez('results.npz',
                distances=distances,
                slopes=slopes)
    print("DONE!")


