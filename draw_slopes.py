__author__ = 'achilleas'

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
