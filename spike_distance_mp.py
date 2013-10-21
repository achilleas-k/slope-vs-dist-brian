import numpy as np
import multiprocessing
import itertools


def stdistance(tli, tlj, cost):
    '''
    Calculates the "spike time" distance (Victor & Purpura, 1996) for a single
    cost.

    tli: vector of spike times for first spike train
    tlj: vector of spike times for second spike train
    cost: cost per unit time to move a spike

    Translated to Python by Achilleas Koutsou from Matlab code by Daniel Reich.
    '''
    nspi = len(tli)
    nspj = len(tlj)
    if cost == 0:
        dist = abs(nspi-nspj)
        return dist
    elif cost == float('inf'):
        dist = nspi+nspj
        return dist
    scr = np.zeros((nspi+1, nspj+1))
    # Initialize margins with cost of adding a spike
    scr[:,0] = range(nspi+1)
    scr[0,:] = range(nspj+1)
    if nspi and nspj:
        for i in range(1, nspi+1):
            for j in range(1, nspj+1):
                scr[i,j] = min([scr[i-1,j]+1,
                                scr[i,j-1]+1,
                                scr[i-1,j-1]+cost*abs(tli[i-1]-tlj[j-1])])
    dist = scr[nspi, nspj]
    return dist

def all_dist_to_end(args):
    idx = args[0]
    spiketrains = args[1]
    cost = args[2]
    num_spiketrains = len(spiketrains)
    distances = []
    for jdx in range(idx + 1, num_spiketrains):
        dist = stdistance(spiketrains[idx], spiketrains[jdx], cost)
        distances.append(dist)
    return distances

def mean_pairwise_distance(spiketrains, cost):
    count = len(spiketrains)
    distances = []
    idx_all = range(count - 1)
    pool = multiprocessing.Pool()
    distances_nested = pool.map(all_dist_to_end,
                                zip(idx_all, itertools.repeat(spiketrains),
                                    itertools.repeat(cost)))
    distances = []
    for dn in distances_nested:
        distances.extend(dn)
    return np.mean(distances)

# test function
if __name__ == "__main__":
    spiketrains = [[0, 1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4.1]]
    print("Spiketrains are: ")
    print(spiketrains)
    result = mean_pairwise_distance(spiketrains, 1)
    print("The mean pairwise distance is: %f" % (result))



