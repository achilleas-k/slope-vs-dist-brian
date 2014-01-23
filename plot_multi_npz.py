from __future__ import print_function
from brian import *
import os
import sys
import pickle
from interval_modulus_metric import interval_modm

directories = sys.argv[1:]

results = []
configs = []
for datadir in directories:
    print("Loading files from %s ..." % datadir)
    filesindir = os.listdir(datadir)
    npzindir = [fid for fid in filesindir if fid.endswith("npz")]
    npzcount = len(npzindir)
    for idx, npzfile in enumerate(npzindir):
        npzdata = load(os.path.join(datadir, npzfile))
        res = npzdata["results"].item()
        conf = npzdata["config"].item()
        results.append(res)
        configs.append(conf)
        print("Finished reading %s (%i/%i)" % (npzfile, idx+1, npzcount),
              end="\r")
    print("")

# recalculate correlation coefficient with bin=w (takes ALL THE AGES)
#print("Recalculating correlation coefficients with bin=w ...")
#import SpiketrainCorrelation as stcorr
#corr = []
#duration = 5*second
#bin = 2*ms
#for idx, res in enumerate(results):
#    inputspikes = res["inspikes"]
#    outspikes = res["outspikes"]
#    int_corr = stcorr.interval_corr(inputspikes, outspikes, bin, duration)
#    corr.append(int_corr)
#    print("Finished %i/%i" % (idx+1, npzcount), end="\r")
#print("")


print("Reorganising data into arrays ...")
# organise data into arrays
# old implementations of npss can cause NaNs, so let's get rid of them
npss = [res["npss"] for res in results]
notnan = [flatnonzero(-isnan(n)) for n in npss]
npss = [n[i] for i, n in zip(notnan, npss)]
npss_mean = [mean(n) for n in npss]

corr = [array(res["correlations"]) for res in results]
corr = [c[i] for i, c in zip(notnan, corr)]
corr_mean = [mean(c) for c in corr]

vp_d = [array(res["vp_dists"]) for res in results]
vp_d = [v[i] for i, v in zip(notnan, vp_d)]
vp_d_mean = [mean(v) for v in vp_d]

kr_d = [array(res["kr_dists"]) for res in results]
kr_d = [k[i] for i, k in zip(notnan, kr_d)]
kr_d_mean = [mean(k) for k in kr_d]

print("Calculating mean interval modulus metric ...")
mm_d = []
mm_d_mean = []
for idx, (res, notnan_idx) in enumerate(zip(results, notnan)):
    inspikes = res["inspikes"]
    outspikes = res["outspikes"]
    mm_d_i = interval_modm(inspikes, outspikes, 0, 5)
    try:
        mm_d_i = [mm_d_i[i] for i in notnan_idx]  # remove the NaN indices
    except IndexError:
        import IPython; IPython.embed()
    mm_d.append(mean(mm_d_i))
    mm_d_mean.append(mean(mm_d))
    print("Finished calculating %i/%i" % (idx+1, len(results)), end="\r")
    sys.stdout.flush()

allresults = {"npss": npss, "npss_mean": npss_mean,
              "corr": corr, "corr_mean": corr_mean,
              "vp_d": vp_d, "vp_d_mean": vp_d_mean,
              "kr_d": kr_d, "kr_d_mean": kr_d_mean,
              "mm_d": mm_d, "mm_d_mean": mm_d_mean,
             }
pickle.dump(allresults, open("metric_comp_results.pkl", "w"))
print("Results and averages saved to metric_comp_results.pkl")

try:
    os.mkdir("plots")
except OSError:
    pass

# plot means
scatter(npss_mean, corr_mean)
xlabel("NPSS"); ylabel("Correlation coefficient")
savefig("plots/npss_corr.eps")
print("Plotted npss_corr.eps")
clf()
scatter(npss_mean, vp_d_mean)
xlabel("NPSS"); ylabel("V-P")
savefig("plots/npss_vp.eps")
print("Plotted npss_vp.eps")
clf()
scatter(npss_mean, kr_d_mean)
xlabel("NPSS"); ylabel("Kreuz")
savefig("plots/npss_kr.eps")
print("Plotted npss_kr.eps")
clf()
scatter(npss_mean, mm_d_mean)
savefig("plots/npps_modm.eps")
print("Plotted npss_modm.eps")
clf()

# plot individual points
#hold(True)
#for n, c, v, k in zip(npss, corr, vp_d, kr_d):
#    figure(1)
#    scatter(n, c, c='k')
#    figure(2)
#    scatter(n, v, c='k')
#    figure(3)
#    scatter(n, k, c='k')
#
#f = figure(1)
#xlabel("NPSS"); ylabel("Correlation coefficient")
#f.savefig("plots/npss_corr_ind.eps")
#print("Plotted npss_corr_ind.eps")
#f.savefig("plots/npss_corr_ind.eps")
#f = figure(2)
#xlabel("NPSS"); ylabel("V-P")
#f.savefig("plots/npss_vp_ind.eps")
#print("Plotted npss_vp_ind.eps")
#f = figure(3)
#xlabel("NPSS"); ylabel("Kreuz")
#f.savefig("plots/npss_kr_ind.eps")
#print("Plotted npss_kr_ind.eps")

print("Correlation between means:")
print("\tNPSS\tCorr\tV-P\tKreuz")
labels = ["NPSS", "Corr", "V-P", "Kreuz"]
for cc, l in zip(corrcoef((npss_mean, corr_mean,
                           vp_d_mean, kr_d_mean)),
                 labels):
    print(l, end="")
    for c in cc:
        print("\t%0.2f" % (c), end="")
    print("")

print("Done!")



