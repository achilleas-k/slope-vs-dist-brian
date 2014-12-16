"""
Pick fixed values to demonstrate different cases and smooth lines.
"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from brian import mV, ms
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
    allmnpss = []
    allkrdst = []
    alldrive = []
    allpeaks = []
    allsigma = []
    for npz in npzfiles:
        data = np.load(npz)
        Nin = data["numin"].item()
        fin = data["inrate"].item()
        weight = data["inweight"].item()
        sconf = data["syncconf"]
        mnpss = data["mnpss"]
        krdists = data["krdists"]
        allmnpss.extend(mnpss)
        allkrdst.extend(krdists)
        allsigma.extend([sc[1] for sc in sconf])
        print("Num inputs:   {}\n"
              "Input rate:   {} Hz\n"
              "Input weight: {} mV".format(Nin, fin, weight))
        drive = Nin*fin*weight*0.01
        peak = Nin*weight
        alldrive.extend([drive]*len(mnpss))
        allpeaks.extend([peak]*len(mnpss))
        print("Asymptotic potential:  {} mV\n"
              "Volley peak potential: {} mV\n".format(drive, peak))
        figname = "figures/N{}_f{}_w{}.pdf".format(Nin, fin, weight)
        plot_results(figname, Nin, fin, weight, sconf, krdists, mnpss)
        count += 1
        print("{}/{} done".format(count, len(npzfiles)))
    plt.clf()
    mnpss = np.array(allmnpss)
    kreuz = np.array(allkrdst)
    drive = np.array(alldrive)
    peaks = np.array(allpeaks)
    jitters = np.array(allsigma)
    njidx = jitters == 0
    pjidx = jitters > 0

    sanedrvidx =  (15 < drive) & (drive <= 30)
    sanepeakidx = (15 < peaks) & (peaks <= 30)
    saneidx = sanedrvidx & sanepeakidx

    fig = plt.figure("NPSS vs SPIKE-distance with jitter", dpi=100, figsize=(8,6))
    colour = jitters*1000
    vmax = max(colour)
    plt.subplot2grid((11,11), (0,0), rowspan=4, colspan=10)
    plt.title("(a)")
    idx = saneidx
    allpts = plt.scatter(mnpss[idx], kreuz[idx], vmin=0, vmax=vmax, c=colour[idx])
    plt.xlabel(r"$\overline{M}$")
    plt.ylabel(r"$D_S$")
    plt.axis(xmin=0, xmax=1, ymin=0)

    plt.subplot2grid((11,11), (6,0), rowspan=4, colspan=4)
    plt.title("(b)")
    idx = saneidx & njidx
    njpts = plt.scatter(mnpss[idx], kreuz[idx], vmin=0, vmax=vmax, c=colour[idx])
    plt.xlabel(r"$\overline{M}$")
    plt.ylabel(r"$D_S$")
    plt.axis(xmin=0, xmax=1, ymin=0)

    plt.subplot2grid((11,11), (6,6), rowspan=4, colspan=4)
    plt.title("(c)")
    idx = saneidx & pjidx
    plt.scatter(mnpss[idx], kreuz[idx], vmin=0, vmax=vmax, c=colour[idx])
    plt.xlabel(r"$\overline{M}$")
    plt.ylabel(r"$D_S$")
    plt.axis(xmin=0, xmax=1, ymin=0)

    cax = fig.add_axes([0.85, 0.15, 0.03, 0.75])
    cbar = plt.colorbar(allpts, cax=cax)
    cbar.set_label(r"$\sigma_{in}$ (ms)")

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.savefig("figures/npss_v_dist_jitter.pdf")
    plt.savefig("figures/npss_v_dist_jitter.png")
