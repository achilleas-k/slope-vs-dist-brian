"""
Pick fixed values to demonstrate different cases and smooth lines.
"""
from __future__ import division, print_function
from brian import *
import numpy as np
import itertools as it
import sys


def lifsim(n_in, inrate, weight):
    sync = np.arange(0, 1.1, 0.1)
    sigma = np.arange(0, 4.1, 0.5)*ms
    inrate = inrate*Hz
    weight = weight*volt
    Vth = 15*mV
    tau = 10*ms
    nsims = len(sync)*len(sigma)

    # neuron
    neuron = NeuronGroup(nsims, "dV/dt = -V/tau : volt",
                         threshold="V>Vth", reset="V=0*mvolt", refractory=1*ms)
    neuron.V = 0*mvolt
    netw = Network(neuron)
    inputgroups = []
    inputconns = []
    inputmons = []
    sync_conf = []
    for s, j in it.product(sync, sigma):
        sync_conf.append((s, j))
        # inputs
        ing = sl.tools.fast_synchronous_input_gen(n_in, inrate,
                                                  s, j, duration)
        inputgroups.append(ing)
        # connection
        inc = Connection(ing, neuron, 'V', weight=weight)
        inputconns.append(inc)
        # monitor
        inm = SpikeMonitor(ing)
        inputmons.append(inp)
    netw.add(*inputgroups)
    netw.add(*inputconns)
    netw.add(*inputmons)
    # monitors
    vmon = StateMonitor(neuron, "V", record=True)
    outmon = SpikeMonitor(neuron)
    netw.add(vmon, outmon)
    # run
    netw.run(duration, report=None)
    if outmon.nspikes < 2:
        print("Warning: No spikes were fired from any of the simulations")
        return
    vmon.insert_spikes(outmon, Vth)
    krdists = []
    mnpss = []
    for idx in range(nsims):
        input_spiketrains = inmon[idx]
        t, d = sl.metrics.kreuz.multivariate(input_spiketrains,
                                             0*second, duration,
                                             int(duration/dt))
        krdists.append(np.trapz(d, t))
        npss = sl.tools.npss(vmon[idx], outmon[idx], 0*mV, Vth, tau, slope_w)
        mnpss.append(mean(npss))
    return sync_conf, krdists, mnpss




if __name__=='__main__':
    print("Setting up ...")
    Nin, fin, weight = sys.argv[1:4]
    Nin = int(Nin)
    fin = int(fin)
    weight = float(weight)
    print("Num inputs:   {}\n"
          "Input rate:   {} Hz\n"
          "Input weight: {} mV".format(Nin, fin, weight))
    drive = Nin*fin*weight*0.01
    peak = Nin*weight
    print("Asymptotic potential:  {} mV\n"
          "Volley peak potential: {} mV\n".format(drive, peak))
    sconf, krdists, mnpss = lifsim(Nin, fin, weight)
    print("Simulations done. Saving data...")
    filestring = "N{}_f{}_w{}.npz".format(Nin, fin, weight)
    np.savez(filestring,
             numin=Nin,
             inrate=fin,
             inweight=weight,
             syncconf=sconf,
             mnpss=mnpss,
             krdists=krdists)
    print("Results saved to {}".format(filestring))
