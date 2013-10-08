from brian import *
import sys
import neurotools as nt
from brian import *
from brian.tools.datamanager import *
import os

def multidist(all_spikes, cost):
    count=len(all_spikes)
    distances=[]
    for i in range(count-1):
        for j in range(i+1, count):
            dist=stdistance(all_spikes[i], all_spikes[j], cost)
            distances.append(dist)
    return mean(distances)

def stdistance(spiketrain_a, spiketrain_b, cost):
    num_spike_i=len(spiketrain_a)      #num of spikes for the first spike train
    num_spike_j=len(spiketrain_b)      #num of spikes for the second spike train
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



if __name__=='__main__':
    data = np.load(sys.argv[1])
    mem_all = data['mem']
    spikes_all = data['spikes']
    inputs_all = data['inputs']
    w=2*ms
    slopes = []
    distances = []
    for memset, spikeset, inputset in zip(mem_all, spikes_all, inputs_all):
        slopes_set = []
        for mem, spikes in zip(memset, spikeset):
            avg, slopes = nt.norm_firing_slope(mem, spikes, 15*mV, 10*ms, w=w)
            slopes_set.append(avg)
        slopes.append(slopes_set)
        distances_set = []
        for spiketrains in inputset:
            d = multidist(spiketrains, cost=1000)
            distances_set.append(d)
        distances.append(distances_set)
    np.savez('results.npz',
                distances=distances,
                slopes=slopes)


