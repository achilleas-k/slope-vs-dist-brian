"""
Pick fixed values to demonstrate different cases and smooth lines.
"""
from __future__ import division, print_function
from brian import *
import numpy as np
import sys
from glob import glob

Vth = 15*mV
tau = 10*ms

greytrunc_dict = {'red':   ((0.0, 0.0, 0.0), (1.0, 0.8, 0.8)),
                  'green': ((0.0, 0.0, 0.0), (1.0, 0.8, 0.8)),
                  'blue':  ((0.0, 0.0, 0.0), (1.0, 0.8, 0.8))}
plt.register_cmap(name="greytrunc", data=greytrunc_dict)
plt.set_cmap("greytrunc")


def plot_results(figname, numin, inrate, inweight, syncconf, kreuz, mnpss):
    kreuz = np.array(kreuz)
    mnpss = np.array(mnpss)
    drive = numin*inrate*inweight*tau
    peaks = numin*inweight
    syncconf = np.array(syncconf)
    #s = syncconf[:,0]
    j = syncconf[:,1]
    plt.clf()
    for ji in np.unique(j):
        idx = (ji == j) & (mnpss > 0) & (kreuz > 0)
        lc = (ji-min(j))/(max(j)-min(j))
        linecolour = plt.get_cmap()(lc)
        plt.plot(mnpss[idx], kreuz[idx], linestyle="--", color=linecolour)
    nonzero = (mnpss > 0) & (kreuz > 0)
    plt.scatter(mnpss[nonzero], kreuz[nonzero], c=j[nonzero])
    if (np.count_nonzero(nonzero)):
        cbar = plt.colorbar()
        cbar.set_label("$\sigma_{in}$")
    plt.axis(xmin=-0.1, xmax=1.1, ymin=-0.1, ymax=2.6)
    plt.xlabel(r"$\overline{M}$")
    plt.ylabel(r"$S_{m}$")
    plt.title(r"$\langle V \rangle$ = {} mV, $\Delta_v$ = {}".format(
        drive, peaks))
    plt.savefig(figname)
    print("Saved figure {}".format(figname))

if __name__=='__main__':
    print("Setting up ...")
    npzfiles = glob(sys.argv[1])
    print("{} files found".format(len(npzfiles)))
    count = 0
    for npz in npzfiles:
        data = np.load(npz)
        Nin = data["numin"].item()
        fin = data["inrate"].item()
        weight = data["inweight"].item()
        sconf = data["syncconf"]
        mnpss = data["mnpss"]
        krdists = data["krdists"]
        print("Num inputs:   {}\n"
              "Input rate:   {} Hz\n"
              "Input weight: {} mV".format(Nin, fin, weight))
        drive = Nin*fin*weight*0.01
        peak = Nin*weight
        print("Asymptotic potential:  {} mV\n"
              "Volley peak potential: {} mV\n".format(drive, peak))
        figname = "N{}_f{}_w{}.png".format(Nin, fin, weight)
        plot_results(figname, Nin, fin, weight, sconf, krdists, mnpss)
        print("Figure {} saved".format(figname))
        count += 1
        print("{}/{} done".format(count, len(npzfiles)))
