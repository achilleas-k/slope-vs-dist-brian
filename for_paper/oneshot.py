"""
Pick fixed values to demonstrate different cases and smooth lines.
"""
from __future__ import division, print_function
from brian import *
import spikerlib as sl
import numpy as np
import itertools as it
import sys
sys.path.append("spikerlib.egg")
import multiprocessing as mp
import os

Vth = 15*mV
tau = 10*ms
duration = 5*second
slope_w = 2*ms

def lifsim(n_in, inrate, weight):
    sync = np.arange(0, 1.1, 0.1)
    sigma = np.arange(0, 4.1, 1.0)*ms
    inrate = inrate*Hz
    weight = weight*mV
    nsims = len(sync)*len(sigma)

    print("Setting up simulation:")
    # neuron
    print("Configuring neuron...")
    neuron = NeuronGroup(nsims, "dV/dt = -V/tau : volt",
                         threshold="V>Vth", reset="V=0*mvolt", refractory=1*ms)
    neuron.V = 0*mvolt
    netw = Network(neuron)
    inputgroups = []
    inputconns = []
    inputmons = []
    sync_conf = []
    print("Creating inputs connections and monitors...")
    for idx, (s, j) in enumerate(it.product(sync, sigma)):
        sync_conf.append((s, j))
        # inputs
        ing = sl.tools.fast_synchronous_input_gen(n_in, inrate,
                                                  s, j, duration)
        inputgroups.append(ing)
        # connection
        inc = Connection(ing, neuron[idx], 'V', weight=weight)
        inputconns.append(inc)
        # monitor
        inm = SpikeMonitor(ing)
        inputmons.append(inm)
        print("{}/{}...".format(idx, nsims), end="\r")
        sys.stdout.flush()
    netw.add(*inputgroups)
    netw.add(*inputconns)
    netw.add(*inputmons)
    # monitors
    vmon = StateMonitor(neuron, "V", record=True)
    outmon = SpikeMonitor(neuron)
    netw.add(vmon, outmon)
    print("Running simulation for {} seconds...".format(duration))
    # run
    netw.run(duration, report="stdout")
    print("Simulation finished!")
    if outmon.nspikes < 2:
        print("Warning: No spikes were fired from any of the simulations")
        return
    vmon.insert_spikes(outmon, Vth*3)
    krdists = []
    mnpss = []
    print("Calculating spike train distance and NPSS...")
    pool = mp.Pool()
    args = []
    for idx in range(nsims):
        inspikes = inputmons[idx].spiketimes.values()
        voltage = vmon[idx]
        outspikes = outmon[idx]
        args.append((inspikes, voltage, outspikes, Vth, tau, slope_w, duration))
    result_iter = pool.imap(calculate_measures, args)
    krdists = []
    mnpss = []
    pool.close()
    pool.join()
    for idx, res in enumerate(result_iter, 1):
        mnpss.append(res[0])
        krdists.append(res[1])
        print("{}/{}...".format(idx, nsims), end="\r")
        sys.stdout.flush()
    print("Done!")
    return sync_conf, krdists, mnpss

def calculate_measures(args):
    inspikes, voltage, outspikes, vth, tau, w, duration = args
    if len(outspikes) < 2:
        return 0, 0
    t, d = sl.metrics.kreuz.multivariate(inspikes, 0*second, duration,
                                         int(duration/ms))
    krdist = np.trapz(d, t)
    npss = sl.tools.npss(voltage, outspikes, 0*mV, vth, tau, w)
    mnpss = mean(npss)
    return mnpss, krdist

def plot_results(figname, numin, inrate, inweight, syncconf, kreuz, mnpss):
    kreuz = np.array(kreuz)
    mnpss = np.array(mnpss)
    drive = numin*inrate*inweight*tau
    peaks = numin*inweight
    syncconf = np.array(syncconf)
    #s = syncconf[:,0]
    j = syncconf[:,1]
    for ji in np.unique(j):
        idx = (ji == j) & (mnpss > 0) & (kreuz > 0)
        lc = (ji-min(j))/(max(j)-min(j))
        linecolour = plt.get_cmap()(lc)
        plt.plot(mnpss[idx], kreuz[idx], linestyle="--", color=linecolour)
    plt.scatter(mnpss, kreuz, c=j)
    cbar = plt.colorbar()
    plt.axis(xmin=-0.1, xmax=1.1, ymin=-0.1, ymax=2.6)
    plt.xlabel(r"$\overline{M}$")
    plt.ylabel(r"$S_{m}$")
    cbar.set_label("$\sigma_{in}$")
    plt.title(r"$\langle V \rangle$ = {} mV, $\Delta_v$ = {}".format(
        drive, peaks))
    plt.savefig(figname)
    print("Saved figure {}".format(figname))

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
    filestring = "N{}_f{}_w{}.npz".format(Nin, fin, weight)
    ans = "_"
    if os.path.exists(filestring):
        while ans not in "LlOo":
            ans = raw_input("{} already exists. "
                            "(L)oad and plot, "
                            "or (O)verwrite? ".format(filestring))
    else:
        ans = "O"

    if ans == "":
        ans = "L"

    if ans in "Oo":
        sconf, krdists, mnpss = lifsim(Nin, fin, weight)
    elif ans in "Ll":
        data = np.load(filestring)
        sconf = data["syncconf"]
        krdists = data["krdists"]
        mnpss = data["mnpss"]
    else:
        raise Exception("WAT")
    figname = "N{}_f{}_w{}.png".format(Nin, fin, weight)
    plot_results(figname, Nin, fin, weight, sconf, krdists, mnpss)
    print("Simulations done. Saving data...")
    np.savez(filestring,
             numin=Nin,
             inrate=fin,
             inweight=weight,
             syncconf=sconf,
             mnpss=mnpss,
             krdists=krdists)
    print("Results saved to {}".format(filestring))
