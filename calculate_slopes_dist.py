from brian import *
from brian.tools.datamanager import *
import sys
#import spike_distance_kreuz as sd
import spike_distance_mp as sd
import multiprocessing as mp
import neurotools as nt
import pickle

_numitems = 0

def load_data(dirname):
    dataman = DataManager(dirname)
    print("Loading data from %s" % (dataman.basepath))
    global _numitems
    _numitems = dataman.itemcount()
    voltage = []
    output_spikes = []
    input_rate = []
    input_spikes = []
    jitter = []
    sync = []
    num_inputs = []
    weight = []
    for idx, data in enumerate(dataman.itervalues(), 1):
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
        sys.stdout.write("\r%i/%i ..." % (idx, _numitems))
        sys.stdout.flush()
    print("Done!\n")
    _numitems = len(voltage)
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
    for idx, (v, spikes) in enumerate(zip(mems, output_spikes), 1):
        npss.append(nt.norm_firing_slope(v[0], spikes[0],
            th=15*mV, tau=10*ms, dt=0.1*ms, w=2*ms)[0])
        sys.stdout.write("\r%i/%i ..." % (idx, _numitems))
        sys.stdout.flush()
    print("Done!\n")
    return npss

def _call_mv_sd(args):
    inps, start, end, samples = args
    t, dist = sd.multivariate_spike_distance(inps, start, end, samples)
    return mean(dist)

def _call_mean_victor(args):
    spikes, cost = args
    mdist = sd.mean_pairwise_distance(spikes, cost)
    return mdist

def calculate_dist_kreuz(data):
    '''
    Calculate spike train distance using Kreuz spike distance
    '''
    print("Calculating distances ...")
    input_spikes = data['input_spikes']
    args = []
    pool = mp.Pool()
    for inps in input_spikes:
        args.append((inps, 0, 2.0, 5))
    result_iter = pool.imap(_call_mv_sd, args)
    dist = []
    for idx, res in enumerate(result_iter, 1):
        dist.append(res)
        sys.stdout.write("\r%i/%i ..." % (idx, _numitems))
        sys.stdout.flush()
    print("Done!\n")
    return dist

def calculate_dist_victor(input_spikes):
    '''
    Calculate spike train distance using mean pairwise Victor distance
    '''
    print("Calculating distances ...")
    args = []
    pool = mp.Pool()
    for inps in input_spikes:
        args.append((inps, 1000))
    result_iter = pool.imap(_call_mean_victor, args)
    dist = []
    for idx, res in enumerate(result_iter, 1):
        dist.append(res)
        sys.stdout.write("\r%i/%i ..." % (idx, _numitems))
        sys.stdout.flush()
    print("Done!\n")
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

#if __name__=='__main__':
#    data_dir = sys.argv[1]
#    data = load_data(data_dir)
#    npss = calculate_slopes(data)
#    data['slopes'] = npss
#    dists = calculate_dist(data)
#    data['distances'] = dists
#    save_filename = "simdata_slopes_dist.pkl"
#    print("Saving all data to %s using pickle.dump ..." % save_filename)
#    pickle.dump(data, open(save_filename, 'w'))
#    print("All done!")

if __name__=='__main__':
    datafile = sys.argv[1]
    print("Loading data from %s ..." % (datafile))
    spikes = pickle.load(datafile)
    print("Data loaded!")
    dist = calculate_dist_victor(spikes)
    save_filename = "victor_dist.pkl"
    print("Saving calculated distances to %s ..." % (save_filename))
    pickle.dump(dist, open(save_filename, 'w'))
    print("All done!")

