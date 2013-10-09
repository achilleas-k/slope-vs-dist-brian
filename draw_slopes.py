import os
import numpy as np
import neurotools as nt

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
        f_out:          output frequency -- Hz (varies)
                            - should be common across all .npz files in a directory
        seed:           random seed for simulation
'''


def load_file(filename):
    if not filename.endswith('.npz'):
        print "%s doesn't have .npz extension. Is it a .npz file?"
        return None
    npz_data = np.load(filename)
    data = {}
    for k in data.keys():
        data[k] = npz_data[k]
    return data

def load_directory(dirname):
    filenames = os.listdir(dirname)
    npzfiles = [fname for fname in filenames if not fname.endswith('.npz')]
    dirdata = {}
    for fname in npzfiles:
        data = load_file(fname)
        dirdata[fname] = data
    return dirdata



