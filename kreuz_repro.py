"""
Attempt to reproduce the plots of Kreuz et al., 2012
"""

import numpy as np
import matplotlib.pyplot as plt
import spikerlib as sl

N = 20
start = 0
end = 0.8
spiketrains = []
spikecentres = np.arange(start, end+0.1, 0.1)

for nst in range(N):
    st = [0]
    shift = 0.005
    halfpoint = len(spikecentres)//2
    for idx, sc in enumerate(spikecentres[:halfpoint]):
        st.append(sc+nst*shift)
    st.extend(spikecentres[halfpoint:])
    spiketrains.append(st)

# calculate stuff
avgbiv = sl.metrics.kreuz.pairwise_mp(spiketrains, start, end, 800)
multiv = sl.metrics.kreuz.multivariate(spiketrains, start, end, 800)

# plots
plt.figure()
plt.subplot(3,1,1)
for h, st in enumerate(spiketrains):
    plt.plot(st, np.zeros_like(st)+h, 'b.')
plt.subplot(3,1,2)
plt.plot(avgbiv[0], avgbiv[1])
plt.subplot(3,1,3)
plt.plot(multiv[0], multiv[1])
plt.show()
