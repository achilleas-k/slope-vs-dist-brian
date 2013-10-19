from brian import *
from brian.tools.datamanager import *
import sys
import spike_distance_kreuz as sd
import neurotools as nt
import pickle

_numitems = 0

def load_data(dirname):
    print("Loading data from %s" % (data.basepath))
    dataman = DataManager(dirname)
    _numitems = dataman.itemcount()
    voltage = []
    output_spikes = []
    input_rate = []
    input_spikes = []
    jitter = []
    sync = []
    num_inputs = []
    weight = []
    for idx, data in enumarete(dataman.itervalues()):
        if not len(data):
            continue
        voltage.append(data['mem'])
        output_spikes.append(data['output_spikes'])
        input_rate.append(data['f_in'])
        jitter.append(data['jitter'])
        sync.append(data['sync'])
        num_inputs.append(data['N_in'])
        weight.append(data['w_in'])
        input_spikes.append(data['input_spikes'])
        sys.stdout.write("%i/%i ...\r" % (idx, _numitems))
        sys.stdout.flush()
    print("\nDone!\n")
    flatdata = {
            'mem': voltage,
            'output_spikes': output_spikes,
            'input_rate': input_rate,
            'jitter': jitter,
            'sync': sync,
            'num_inputs': num_inputs,
            'input_weight': weight,
            'input_spikes': input_spikes,
            }
    return flatdata

def calculate_slopes(data):
    print("Calculating slopes ...")
    npss = []
    mems = data['mem']
    output_spikes = data['output_spikes']
    for idx, v, spikes in enumerate(zip(mems, output_spikes)):
        npss.append(nt.norm_firing_slope(v[0], spikes[0],
            th=15*mV, tau=10*ms, dt=0.1*ms, w=2*ms)[0])
        sys.stdout.write("%i/%i ...\r" % (idx, _numitems))
        sys.stdout.flush()
    print("\nDone!\n")
    return npss

def calculate_dist(data):
    '''
    Calculate spike train distance using Kreuz spike distance
    '''
    print("Calculating distances ...")
    input_spikes = data['input_spikes']
    dist = []
    for idx, inps in enumerate(input_spikes):
        t, sd_i = sd.multivariate_spike_distance(inputs, 0, 2.0, 5)
        dist.append(mean(sd_i))
        sys.stdout.write("%i/%i ...\r" % (idx, _numitems))
        sys.stdout.flush()
    print("\nDone!\n")
    return dist

def aggregate_slopes(data):
    npss = data['slopes']
    sync = data['sync']
    jitter = data['jitter']
    sync_range = sorted(unique(sync))
    jitter_range = sorted(unique(jitter))
    img = zeros((len(jitter_range), len(sync_range)))
    for si, ji, slp in zip(sync, jitter, npss):
        j = jitter_range.index(float(ji))
        s = sync_range.index(si)
        img[j,s] += slp
    return sync_range, jitter_range, img

if __name__=='__main__':
    data_dir = sys.argv[1]
    data = load_data(data_dir)
    npss = calculate_slopes(data)
    data['slopes'] = npss
    dists = calculate_dist(data)
    data['distances'] = dists
    save_filename = "simdata_slopes_dist.pkl"
    print("Saving all data to %s using pickle.dump ..." % save_filename)
    pickle.dump(data, open(save_filename, 'w'))
    print("All done!")


