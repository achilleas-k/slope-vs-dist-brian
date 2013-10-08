import numpy as np
import multiprocessing
import itertools

def stdistance(spiketrain_a, spiketrain_b, cost):
    num_spike_i=len(spiketrain_a)
    num_spike_j=len(spiketrain_b)
    if num_spike_i==0 or num_spike_j==0:
        return 0
    matrix=[[0 for col in range(num_spike_j)] for row in range(num_spike_i)]
    for i in range(num_spike_i):
        matrix[i][0] = i
    for i in range(num_spike_j):
        matrix[0][i] = i
    for m in range(1,num_spike_i):
        for l in range(1,num_spike_j):
            cost_a=matrix[m-1][l]+1
            cost_b=matrix[m][l-1]+1
            temp=abs((spiketrain_a[m])-(spiketrain_b[l]))
            cost_c=matrix[m-1][l-1]+(cost*temp)
            matrix[m][l]=min(cost_a,cost_b,cost_c)
    D_spike=matrix[num_spike_i-1][num_spike_j-1]
    return D_spike

def all_dist_to_end(args):
    idx = args[0]
    spiketrains = args[1]
    cost = args[2]
    num_spiketrains = len(spiketrains)
    distances = []
    for jdx in range(idx+1, num_spiketrains):
        dist = stdistance(spiketrains[idx], spiketrains[jdx], cost)
        distances.append(dist)
    return distances

def mean_pairwise_distance(spiketrains, cost):
    count=len(spiketrains)
    distances=[]
    idx_all = range(count-1)
    pool = multiprocessing.Pool()
    distances_nested = pool.map(all_dist_to_end,
            zip(idx_all, itertools.repeat(spiketrains), itertools.repeat(cost)))
    distances = []
    for dn in distances_nested:
        distances.extend(dn)
    return np.mean(distances)

# test function
if __name__=="__main__":
    spiketrains = [[0, 1, 2, 3, 4],[1,2,3,4],[1,2,3,4], [1,2,3,4.1]]
    print("Spiketrains are: ")
    print(spiketrains)
    result = mean_pairwise_distance(spiketrains, 1)
    print("The mean pairwise distance is: %f" % (result))



