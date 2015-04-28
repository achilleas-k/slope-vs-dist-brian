import spikerlib as sl
import numpy as np
import matplotlib.pyplot as plt

rates = range(10, 101, 10)

krone = []
for rate in rates:
    poiss = sl.tools.poisson_spikes(5, rate)
    t, d = sl.metrics.kreuz.distance(poiss, [], 0, 5, 5000)
    krone.append(np.trapz(d, t))
    print(rate)

krtwo = []
for rate in rates:
    poiss1 = sl.tools.poisson_spikes(5, rate)
    poiss2 = sl.tools.poisson_spikes(5, rate)
    t, d = sl.metrics.kreuz.distance(poiss1, poiss2, 0, 5, 5000)
    krtwo.append(np.trapz(d, t))
    print(rate)

vpone = []
for rate in rates:
    poiss = sl.tools.poisson_spikes(5, rate)
    d = sl.metrics.victor_purpura.distance(poiss, [], 5)
    vpone.append(d)
    print(rate)

vptwo = []
for rate in rates:
    poiss1 = sl.tools.poisson_spikes(5, rate)
    poiss2 = sl.tools.poisson_spikes(5, rate)
    d = sl.metrics.victor_purpura.distance(poiss1, poiss2, 5)
    vptwo.append(d)
    print(rate)

plt.subplot(4, 1, 1)
plt.title("Sd 1")
plt.plot(rates, krone)
plt.subplot(4, 1, 2)
plt.title("Sd 2")
plt.plot(rates, krtwo)
plt.subplot(4, 1, 3)
plt.title("Vd 1")
plt.plot(rates, vpone)
plt.subplot(4, 1, 4)
plt.title("Vd 2")
plt.plot(rates, vptwo)
plt.show()
