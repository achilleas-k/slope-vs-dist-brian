from __future__ import print_function
from brian import *
import os
import sys

datadir = sys.argv[1]

filesindir = os.listdir(datadir)
npzindir = [fid for fid in filesindir if fid.endswith("npz")]
npzcount = len(npzindir)
results = []
configs = []
for idx, npzfile in enumerate(npzindir):
    npzdata = load(os.path.join(datadir, npzfile))
    res = npzdata["results"].item()
    conf = npzdata["config"].item()
    results.append(res)
    configs.append(conf)
    print("Finished reading %s (%i/%i)" % (npzfile, idx+1, npzcount), end="\r")
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
_idx = [-isnan(n) for n in npss]
npss = [n[i] for i, n in zip(_idx, npss)]
npss_mean = [mean(n) for n in npss]

corr = [array(res["correlations"]) for res in results]
corr = [c[i] for i, c in zip(_idx, corr)]
corr_mean = [mean(c) for c in corr]

vp_d = [array(res["vp_dists"]) for res in results]
vp_d = [v[i] for i, v in zip(_idx, vp_d)]
vp_d_mean = [mean(v) for v in vp_d]

kr_d = [array(res["kr_dists"]) for res in results]
kr_d = [k[i] for i, k in zip(_idx, kr_d)]
kr_d_mean = [mean(k) for k in kr_d]

# plot means
scatter(npss_mean, corr_mean)
xlabel("NPSS"); ylabel("Correlation coefficient")
savefig("npss_corr.pdf")
print("Plotted npss_corr.pdf")
clf()
scatter(npss_mean, vp_d_mean)
xlabel("NPSS"); ylabel("V-P")
savefig("npss_vp.pdf")
print("Plotted npss_vp.pdf")
clf()
scatter(npss_mean, kr_d_mean)
xlabel("NPSS"); ylabel("Kreuz")
savefig("npss_kr.pdf")
print("Plotted npss_kr.pdf")
clf()

# plot individual points
hold(True)
for n, c, v, k in zip(npss, corr, vp_d, kr_d):
    figure(1)
    scatter(n, c, c='k')
    figure(2)
    scatter(n, v, c='k')
    figure(3)
    scatter(n, k, c='k')

f = figure(1)
xlabel("NPSS"); ylabel("Correlation coefficient")
f.savefig("npss_corr_ind.pdf")
print("Plotted npss_corr_ind.pdf")
f.savefig("npss_corr_ind.pdf")
f = figure(2)
xlabel("NPSS"); ylabel("V-P")
f.savefig("npss_vp_ind.pdf")
print("Plotted npss_vp_ind.pdf")
f = figure(3)
xlabel("NPSS"); ylabel("Kreuz")
f.savefig("npss_kr_ind.pdf")
print("Plotted npss_kr_ind.pdf")

print("Correlation between means:")
print("\tNPSS\tCorr\tV-P\tKreuz")
labels = ["NPSS", "Corr", "V-P", "Kreuz"]
for cc, l in zip(corrcoef((npss_mean, corr_mean, vp_d_mean, kr_d_mean)), labels):
    print(l, end="")
    for c in cc:
        print("\t%0.2f" % (c), end="")
    print("")

print("Done!")
