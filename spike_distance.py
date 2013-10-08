import numpy as np

def multidist(all_spikes, cost):
    count=len(all_spikes)
    distances=[]
    for i in range(count-1):
        for j in range(i+1, count):
            dist=stdistance(all_spikes[i], all_spikes[j], cost)
            distances.append(dist)
    return np.mean(distances)

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



