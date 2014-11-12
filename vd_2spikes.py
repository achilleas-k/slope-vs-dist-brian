from __future__ import print_function, division
import brian
import numpy as np
import spikerlib as sl
import matplotlib.pyplot as plt

def seq_pairs(lst):
    i = iter(lst)
    first = prev = item = i.next()
    for item in i:
        yield prev, item
        prev = item
    yield item, first

def calc_Vws(input_spikes, output_spikes, w, DV, tau):
    t0 = 0
    Vws = []
    for outspk in output_spikes:
        interval_inputs = []
        for spiketrain in input_spikes:
            interval_inputs.extend([ti for ti in spiketrain
                                    if (t0 <= ti) and (ti <= outspk-w)])
        cur_vws = sum([DV*np.exp(-(outspk-w-ii)/tau) for ii in interval_inputs])
        Vws.append(cur_vws)
        t0 = outspk
    return Vws


tau = 0.01
w = 0.002
weight = 0.012

# inputs
print("Creating inputs ...")
n_input_trains = 2
spiketrain_one = np.arange(0.001, 2, 0.1)
spiketrain_two = spiketrain_one-np.linspace(0, 0.07, len(spiketrain_one))
input_spiketrains = [spiketrain_one, spiketrain_two]
output_spiketrain = spiketrain_one[:]
print("Calculating V(w_s) ...")
vws = calc_Vws(input_spiketrains, output_spiketrain, w, weight, tau)
npss = vws  # temporary
print("Calculating pairwise Kreuz distance (interval mode) ...")
krt, krdist = sl.metrics.kreuz.interval(input_spiketrains, output_spiketrain,
                                  samples=500, mp=True)
krdist = [np.trapz(kr, t) for t, kr in zip(krt, krdist)]
krt = [t[-1] for t in krt]

print("Reticulating splines ...")
plt.ion()
dt = brian.defaultclock.dt
winstarts = output_spiketrain-w
winstarts_idx = (winstarts/dt).astype('int')

plt.figure()
plt.title("All measures")
plt.plot(output_spiketrain, npss, label="NPSS")
plt.plot(krt, krdist, label="Kreuz")
plt.legend()

plt.figure()
plt.title("Kreuz vs NPSS")
plt.scatter(krdist, npss[1:])

