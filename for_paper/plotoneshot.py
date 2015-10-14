"""
Pick fixed values to demonstrate different cases and smooth lines.
"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import sys
from glob import glob
import os

Vth = 15.0
tau = 0.01

#greytrunc_dict = {'red':   ((0.0, 0.0, 0.0), (1.0, 0.8, 0.8)),
#                  'green': ((0.0, 0.0, 0.0), (1.0, 0.8, 0.8)),
#                  'blue':  ((0.0, 0.0, 0.0), (1.0, 0.8, 0.8))}
#plt.register_cmap(name="greytrunc", data=greytrunc_dict)
#plt.set_cmap("greytrunc")

directories = {0: "figures/subsub",
               1: "figures/subsup",
               2: "figures/supsub",
               3: "figures/supsup"}

for dirname in directories.values():
    try:
        os.makedirs(dirname)
    except OSError:
        pass


def plot_results(figname, numin, inrate, inweight, syncconf, kreuz, mnpss):
    kreuz = np.array(kreuz)
    mnpss = np.array(mnpss)
    drive = numin*inrate*inweight*tau
    peaks = numin*inweight
    syncconf = np.array(syncconf)
    #s = syncconf[:,0]
    j = syncconf[:,1]*1000
    plt.figure(figsize=(4, 3))
    for ji in np.unique(j):
        idx = (ji == j) & (mnpss > 0) & (kreuz > 0)
        lc = (ji-min(j))/(max(j)-min(j))
        linecolour = plt.get_cmap()(lc)
        plt.plot(kreuz[idx], mnpss[idx], linestyle="--", color=linecolour)
    nonzero = (mnpss > 0) & (kreuz > 0)
    plt.scatter(kreuz[nonzero], mnpss[nonzero], c=j[nonzero])
    if (np.count_nonzero(nonzero)):
        cbar = plt.colorbar()
        cbar.set_label("$\sigma_{in}$ (ms)")
    plt.axis(ymin=0, ymax=1, xmin=-0.01, xmax=0.51)
    plt.xlabel(r"$D_S$")
    plt.ylabel(r"$\overline{M}$")
    #plt.title(r"$V_{\infty}$ = {} mV, $\Delta_v$ = {}".format(
    #    drive, peaks))
    case = int(drive > Vth)*2+int(peaks > Vth)
    dirname = directories[case]
    plt.savefig(os.path.join(dirname, figname)+".png", bbox_inches="tight")
    plt.savefig(os.path.join(dirname, figname)+".pdf", bbox_inches="tight")
    plt.close()
    print("Saved figure {} (png and pdf)".format(figname))

if __name__=='__main__':
    print("Setting up ...")
    npzfiles = glob(sys.argv[-1])
    plotind = True if "--full" in sys.argv else False
    print("{} files found".format(len(npzfiles)))
    count = 0
    allmnpss = []
    allkrdst = []
    alldrive = []
    allpeaks = []
    allsigma = []
    allsynch = []
    allinfrq = []
    allinwgt = []
    allnumin = []
    for npz in npzfiles:
        data = np.load(npz)
        Nin = data["numin"].item()
        fin = data["inrate"].item()
        weight = data["inweight"].item()
        sconf = data["syncconf"]
        mnpss = data["mnpss"]
        krdists = data["krdists"]/5.0
        allmnpss.extend(mnpss)
        allkrdst.extend(krdists)
        allsigma.extend([sc[1] for sc in sconf])
        allsynch.extend([sc[0] for sc in sconf])
        allnumin.extend([Nin]*len(mnpss))
        allinfrq.extend([fin]*len(mnpss))
        allinwgt.extend([weight]*len(mnpss))
        print("Num inputs:   {}\n"
              "Input rate:   {} Hz\n"
              "Input weight: {} mV".format(Nin, fin, weight))
        drive = Nin*fin*weight*0.01
        peak = Nin*weight
        alldrive.extend([drive]*len(mnpss))
        allpeaks.extend([peak]*len(mnpss))
        print("Asymptotic potential:  {} mV\n"
              "Volley peak potential: {} mV".format(drive, peak))
        figname = "N{}_f{}_w{}".format(Nin, fin, weight).replace(".", "_")
        if plotind:
            plot_results(figname, Nin, fin, weight, sconf, krdists, mnpss)
        count += 1
        print("{}/{} done".format(count, len(npzfiles)))
        print()
    plt.clf()
    mnpss = np.array(allmnpss)
    kreuz = np.array(allkrdst)
    drive = np.array(alldrive)
    peaks = np.array(allpeaks)
    jitters = np.array(allsigma)
    sync = np.array(allsynch)
    numin = np.array(allnumin)
    weight = np.array(allinwgt)
    fin = np.array(allinfrq)
    njidx = jitters == 0
    pjidx = jitters > 0

    sanedrvidx =  (10 < drive) & (drive <= 50)
    sanepeakidx = (10 < peaks) & (peaks <= 50)
    saneidx = sanedrvidx & sanepeakidx

    print("Plotting results with jitter...")
    ### PLOT NPSS V DISTANCE WITH JITTER AS COLOUR
    fig = plt.figure("NPSS vs SPIKE-distance with jitter", dpi=100, figsize=(8,6))
    lowseg, highseg = 0.0001, 0.0021
    ### All data points
    colour = jitters*1000
    vmin = min(colour)
    vmax = max(colour)
    plt.subplot2grid((11,30), (0,0), rowspan=4, colspan=26)
    plt.title("(a)")
    idx = saneidx
    allpts = plt.scatter(kreuz[idx], mnpss[idx], vmin=0, vmax=vmax, c=colour[idx])
    plt.xlabel(r"$D_S$")
    plt.ylabel(r"$\overline{M}$")
    plt.axis(ymin=0, ymax=1, xmin=-0.01, xmax=0.51)
    print("Number of simulations (total) {}".format(np.count_nonzero(idx)))

    ### Split three colour bands
    lowidx = jitters < lowseg
    mididx = (lowseg <= jitters) & (jitters < highseg)
    highidx = highseg <= jitters
    plt.subplot2grid((11,30), (6,0), rowspan=4, colspan=8)
    plt.title("(b)")
    #idx = saneidx & njidx
    idx = lowidx & saneidx
    njpts = plt.scatter(kreuz[idx], mnpss[idx], vmin=0, vmax=vmax, c=colour[idx])
    plt.xlabel(r"$D_S$")
    plt.xticks([0, 0.125, 0.25, 0.375, 0.5], ["", "", 0.25, "",  0.5])
    plt.ylabel(r"$\overline{M}$")
    plt.axis(ymin=0, ymax=1, xmin=-0.01, xmax=0.51)
    print("Number of simulations without jitter {}".format(np.count_nonzero(idx)))

    plt.subplot2grid((11,30), (6,9), rowspan=4, colspan=8)
    plt.title("(c)")
    #idx = saneidx & pjidx
    idx = mididx & saneidx
    plt.scatter(kreuz[idx], mnpss[idx], vmin=0, vmax=vmax, c=colour[idx])
    locs, labels = plt.yticks()
    plt.xlabel(r"$D_S$")
    plt.xticks([0, 0.125, 0.25, 0.375, 0.5], ["", "", 0.25, "",  0.5])
    #plt.ylabel(r"$\overline{M}$")
    plt.yticks(locs, [])
    plt.axis(ymin=0, ymax=1, xmin=-0.01, xmax=0.51)
    print("Number of simulations with low jitter {}".format(np.count_nonzero(idx)))

    plt.subplot2grid((11,30), (6,18), rowspan=4, colspan=8)
    plt.title("(d)")
    #idx = saneidx & pjidx
    idx = highidx & saneidx
    plt.scatter(kreuz[idx], mnpss[idx], vmin=0, vmax=vmax, c=colour[idx])
    print("min kreuz: {}".format(min(kreuz[idx])))
    print("max mnpss: {}".format(max(mnpss[idx])))
    plt.xlabel(r"$D_S$")
    plt.xticks([0, 0.125, 0.25, 0.375, 0.5], ["", "", 0.25, "",  0.5])
    #plt.ylabel(r"$\overline{M}$")
    locs, labels = plt.yticks()
    plt.yticks(locs, [])
    plt.axis(ymin=0, ymax=1, xmin=-0.01, xmax=0.51)
    print("Number of simulations with high jitter {}".format(np.count_nonzero(idx)))

    cax = fig.add_axes([0.85, 0.15, 0.03, 0.75])
    cbar = plt.colorbar(allpts, cax=cax)
    cbar.set_label(r"$\sigma_{in}$ (ms)")

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.savefig("figures/npss_v_dist_jitter.pdf")
    plt.savefig("figures/npss_v_dist_jitter.png")

    print("Plotting results with drive...")
    ### PLOT NPSS V DISTANCE WITH DRIVE AS COLOUR
    fig = plt.figure("NPSS vs SPIKE-distance with drive", dpi=100, figsize=(8,6))
    lowseg, highseg = Vth, 2*Vth
    ### All data points
    colour = drive
    plt.subplot2grid((11,30), (0,0), rowspan=4, colspan=26)
    plt.title("(a)")
    idx = saneidx
    vmax = max(colour[idx])
    vmin = min(colour[idx])
    allpts = plt.scatter(kreuz[idx], mnpss[idx], vmin=vmin, vmax=vmax, c=colour[idx])
    plt.xlabel(r"$D_S$")
    plt.ylabel(r"$\overline{M}$")
    plt.axis(ymin=0, ymax=1, xmin=-0.01, xmax=0.51)

    ### Split three colour bands
    lowidx = drive < lowseg
    mididx = (lowseg <= drive) & (drive < highseg)
    highidx = highseg <= drive
    plt.subplot2grid((11,30), (6,0), rowspan=4, colspan=8)
    plt.title("(b)")
    idx = lowidx & saneidx
    njpts = plt.scatter(kreuz[idx], mnpss[idx], vmin=vmin, vmax=vmax, c=colour[idx])
    plt.xlabel(r"$D_S$")
    plt.ylabel(r"$\overline{M}$")
    plt.xticks([0, 0.125, 0.25, 0.375, 0.5], ["", "", 0.25, "",  0.5])
    plt.axis(ymin=0, ymax=1, xmin=-0.01, xmax=0.51)
    print("Number of simulations with low drive {}".format(np.count_nonzero(idx)))

    plt.subplot2grid((11,30), (6,9), rowspan=4, colspan=8)
    plt.title("(c)")
    idx = mididx & saneidx
    plt.scatter(kreuz[idx], mnpss[idx], vmin=vmin, vmax=vmax, c=colour[idx])
    plt.xlabel(r"$D_S$")
    plt.xticks([0, 0.125, 0.25, 0.375, 0.5], ["", "", 0.25, "",  0.5])
    locs, labels = plt.yticks()
    plt.yticks(locs, [])
    plt.axis(ymin=0, ymax=1, xmin=-0.01, xmax=0.51)
    print("Number of simulations with mid drive {}".format(np.count_nonzero(idx)))

    plt.subplot2grid((11,30), (6,18), rowspan=4, colspan=8)
    plt.title("(d)")
    idx = highidx & saneidx
    plt.scatter(kreuz[idx], mnpss[idx], vmin=vmin, vmax=vmax, c=colour[idx])
    plt.xlabel(r"$D_S$")
    plt.xticks([0, 0.125, 0.25, 0.375, 0.5], ["", "", 0.25, "",  0.5])
    locs, labels = plt.yticks()
    plt.yticks(locs, [])
    plt.axis(ymin=0, ymax=1, xmin=-0.01, xmax=0.51)
    print("Number of simulations with high drive {}".format(np.count_nonzero(idx)))

    cax = fig.add_axes([0.85, 0.15, 0.03, 0.75])
    cbar = plt.colorbar(allpts, cax=cax)
    ticks = range(0, 101, 5)
    cbar.set_ticks(ticks)
    cbar.set_label(r"$V_{\infty}$ (mV)")

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.savefig("figures/npss_v_dist_drive.pdf")
    plt.savefig("figures/npss_v_dist_drive.png")

    print("Plotting results with peaks...")
    ### PLOT NPSS V DISTANCE WITH PEAKS AS COLOUR
    fig = plt.figure("NPSS vs SPIKE-distance with peaks", dpi=100, figsize=(8,6))
    lowseg, highseg = Vth, 2*Vth
    ### All data points
    colour = peaks
    plt.subplot2grid((11,30), (0,0), rowspan=4, colspan=26)
    plt.title("(a)")
    idx = saneidx
    vmax = max(colour[idx])
    vmin = min(colour[idx])
    allpts = plt.scatter(kreuz[idx], mnpss[idx], vmin=vmin, vmax=vmax, c=colour[idx])
    plt.xlabel(r"$D_S$")
    plt.ylabel(r"$\overline{M}$")
    plt.axis(ymin=0, ymax=1, xmin=-0.01, xmax=0.51)

    ### Split three colour bands
    lowidx = peaks < lowseg
    mididx = (lowseg <= peaks) & (peaks < highseg)
    highidx = highseg <= peaks
    plt.subplot2grid((11,30), (6,0), rowspan=4, colspan=8)
    plt.title("(b)")
    idx = lowidx & saneidx
    njpts = plt.scatter(kreuz[idx], mnpss[idx], vmin=vmin, vmax=vmax, c=colour[idx])
    plt.xlabel(r"$D_S$")
    plt.ylabel(r"$\overline{M}$")
    plt.xticks([0, 0.125, 0.25, 0.375, 0.5], ["", "", 0.25, "",  0.5])
    plt.axis(ymin=0, ymax=1, xmin=-0.01, xmax=0.51)
    print("Number of simulations with low peaks {}".format(np.count_nonzero(idx)))

    plt.subplot2grid((11,30), (6,9), rowspan=4, colspan=8)
    plt.title("(c)")
    idx = mididx & saneidx
    plt.scatter(kreuz[idx], mnpss[idx], vmin=vmin, vmax=vmax, c=colour[idx])
    plt.xlabel(r"$D_S$")
    plt.xticks([0, 0.125, 0.25, 0.375, 0.5], ["", "", 0.25, "",  0.5])
    locs, labels = plt.yticks()
    plt.yticks(locs, [])
    plt.axis(ymin=0, ymax=1, xmin=-0.01, xmax=0.51)
    print("Number of simulations with mid peaks {}".format(np.count_nonzero(idx)))

    plt.subplot2grid((11,30), (6,18), rowspan=4, colspan=8)
    plt.title("(d)")
    idx = highidx & saneidx
    plt.scatter(kreuz[idx], mnpss[idx], vmin=vmin, vmax=vmax, c=colour[idx])
    plt.xlabel(r"$D_S$")
    plt.xticks([0, 0.125, 0.25, 0.375, 0.5], ["", "", 0.25, "",  0.5])
    locs, labels = plt.yticks()
    plt.yticks(locs, [])
    plt.axis(ymin=0, ymax=1, xmin=-0.01, xmax=0.51)
    print("Number of simulations with high peaks {}".format(np.count_nonzero(idx)))

    cax = fig.add_axes([0.85, 0.15, 0.03, 0.75])
    cbar = plt.colorbar(allpts, cax=cax)
    ticks = range(0, 101, 5)
    cbar.set_ticks(ticks)
    cbar.set_label(r"$\Delta_v$ (mV)")

    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.savefig("figures/npss_v_dist_peaks.pdf")
    plt.savefig("figures/npss_v_dist_peaks.png")

    print("Plotting the three cases from our first paper...")
    colour = jitters*1000
    ### case 1: low peak, low drive
    case_one_idx = (peaks < Vth) & (drive < Vth)
    plt.subplot(4, 1, 1)
    idx = case_one_idx & saneidx
    allpts = plt.scatter(kreuz[idx], mnpss[idx], c=colour[idx])
    plt.axis(ymin=0, ymax=1, xmin=-0.01, xmax=0.51)
    plt.title("(a)")
    plt.ylabel("$\overline{M}$")
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], [])
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0], [0, "", 0.5, "", 1.0])
    print("Number of simulations in case 1 {}".format(np.count_nonzero(idx)))

    ### case 2: high peak, low drive
    case_two_idx = (peaks >= Vth) & (drive < Vth)
    idx = case_two_idx & saneidx
    plt.subplot(4, 1, 2)
    plt.scatter(kreuz[idx], mnpss[idx], c=colour[idx])
    plt.axis(ymin=0, ymax=1, xmin=-0.01, xmax=0.51)
    plt.title("(b)")
    plt.ylabel("$\overline{M}$")
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], [])
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0], [0, "", 0.5, "", 1.0])
    print("Number of simulations in case 2 {}".format(np.count_nonzero(idx)))

    ### case 3: low peak, high drive
    case_three_idx = (peaks < Vth) & (drive >= Vth)
    idx = case_three_idx & saneidx
    plt.subplot(4, 1, 3)
    plt.scatter(kreuz[idx], mnpss[idx], c=colour[idx])
    plt.axis(ymin=0, ymax=1, xmin=-0.01, xmax=0.51)
    plt.title("(c)")
    plt.ylabel(r"$\overline{M}$")
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], [])
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0], [0, "", 0.5, "", 1.0])
    print("Number of simulations in case 3 {}".format(np.count_nonzero(idx)))

    ### case 4: high peak, high drive
    case_four_idx = (peaks >= Vth) & (drive >= Vth)
    idx = case_four_idx & saneidx
    plt.subplot(4, 1, 4)
    plt.scatter(kreuz[idx], mnpss[idx], c=colour[idx])
    plt.axis(ymin=0, ymax=1, xmin=-0.01, xmax=0.51)
    plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
    plt.yticks([0, 0.25, 0.5, 0.75, 1.0], [0, "", 0.5, "", 1.0])
    plt.ylabel(r"$\overline{M}$")
    plt.title("(d)")
    plt.xlabel(r"$D_S$")
    print("Number of simulations in case 4 {}".format(np.count_nonzero(idx)))

    plt.subplots_adjust(wspace=0.2, hspace=0.5, right=0.8, bottom=0.15)
    ### l, b, w, h
    cax = fig.add_axes([0.85, 0.15, 0.03, 0.75])
    cbar = plt.colorbar(allpts, cax=cax)
    cbar.set_label(r"$\sigma_{in}$ (ms)")

    plt.savefig("figures/case_split.png")
    plt.savefig("figures/case_split.pdf")

    print("Parameter ranges for data points used")
    idx = saneidx
    print("S_in:            {:3.1f} --- {:3.1f}".format(
        min(sync[idx]), max(sync[idx])))
    print("sigma_in:        {:3.1f} --- {:3.1f} (ms)".format(
        min(jitters[idx])*1000, max(jitters[idx])*1000))
    print("Num inputs:      {:3d} --- {:3d}".format(
        min(numin[idx]), max(numin[idx])))
    print("Input weight:    {:3.1f} --- {:3.1f} (mV)".format(
        min(weight[idx]), max(weight[idx])))
    print("Input rate:      {:3d} --- {:3d} (Hz)".format(
        min(fin[idx]), max(fin[idx])))
    print("Done")

