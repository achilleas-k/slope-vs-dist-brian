import os
import sys
import numpy as np
import neurotools as nt
from brian import *

'''
Assumed data format:
    Directory containing .npz files or single .npz file.
    Each .npz contains the simulation results (input spikes, membrane potentials,
    output spikes) as well as parameter values for a set of simulations that
    share the same intrinsic neuron parameter values and input rates.
    What varies between simulations in a single .npz file are the input
    configurations, which determine the degree of synchrony and jitter of the
    simulation.

    Dictionary format of .npz file:
        duration:       simulation duration -- second (unique)
        input_configs: input configurations (S_in -- 1, sigma -- second)
                            - should be common across all .npz files in a directory
        f_in:           input frequency -- Hz (unique)
        N_in:           number of input spike trains -- 1 (unique)
        w_in:           input weight -- volt (unique)
        input_spikes:   input spike trains
        mem:            membrane potentials -- volt
        spikes:         output spike trains
        seed:           random seed for simulation
'''


def load_file(filename):
    if not filename.endswith('.npz'):
        print "%s doesn't have .npz extension. Is it a .npz file?"
        return None
    npz_data = np.load(filename)
    data = {}
    for k in npz_data.keys():
        data[k] = npz_data[k]
    return data

def load_directory(dirname):
    filenames = os.listdir(dirname)
    npzfiles = [fname for fname in filenames if fname.endswith('.npz')]
    dirdata = {}
    for fname in npzfiles:
        data = load_file(os.path.join(dirname,fname))
        dirdata[fname] = data
    return dirdata

def calc_slopes(data):
    """
    Calculates the mean slope of each simulation contained in the data and
    adds it to the dictionary.
    """
    spikes = data['spikes'].item()
    mem = data['mem']
    mslopes = []
    for m, s in zip(mem, spikes.itervalues()):
        _mslope, ign = nt.norm_firing_slope(m, s, 15*mV, 10*ms, w=2*ms)
        mslopes.append(_mslope)
    data['mslopes'] = mslopes
    return data

def plot_slopes(data):
    if not data.has_key('mslopes'):
        data = calc_slopes(data)
    n_in = data['N_in']; f_in = data['f_in']; w_in = data['w_in']
    input_configs = data['input_configs']
    mean_slopes = data['mslopes']
    unique_sync = unique(input_configs[:,0])
    unique_jitter = unique(input_configs[:,1])
    if len(input_configs) != len(unique_sync)*len(unique_jitter):
        warn('Not all sync-jitter pairs exist in input configurations.')
    figure_data = zeros((len(unique_jitter), len(unique_sync)))
    sync_idx = dict((v, idx) for idx, v in enumerate(sorted(unique_sync)))
    jitter_idx = dict((v, idx) for idx, v in enumerate(sorted(unique_jitter)))
    for ic, mslope in zip(input_configs, mean_slopes):
        sync = ic[0]
        jitter = ic[1]
        figure_data[jitter_idx[jitter], sync_idx[sync]] = mslope
    savename = "slopes_%i_%i_%f.png" % (n_in, f_in, w_in)
    extent = (min(sync_idx), max(sync_idx), min(jitter_idx), max(jitter_idx))
    imshow(figure_data, origin='lower', extent=extent, aspect='auto')
    colorbar()
    savefig(savename)
    clf()
    print("\tSaved figure %s" % (savename))

if __name__=='__main__':
    filename = sys.argv[1]
    if os.path.isdir(filename):
        print("Loading data from directory %s" % filename)
        dirdata = load_directory(filename)
    elif os.path.isfile(filename):
        print("Loading data from file %s" % filename)
        data = load_file(filename)
        dirdata = {filename: data}
    else:
        print("Error opening %s: argument not a file or directory" % (filename))
        sys.exit(3)
    for f, d in dirdata.iteritems():
        print("Plotting slopes for %s ..." % (f))
        plot_slopes(d)


