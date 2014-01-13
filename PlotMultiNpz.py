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
    print("Finished reading %s (%i/%i)" % (npzfile, idx+1, npzcount))

# organise data into arrays
npss = [res["npss"] for res in results]
npss_mean = [mean(n) for n in npss]
corr = [res["correlations"] for res in results]
corr_mean = [mean(c) for c in corr]
vp_d = [res["vp_dists"] for res in results]
vp_d_mean = [mean(v) for v in vp_d]
kr_d = [res["kr_dists"] for res in results]
kr_d_mean = [mean(k) for k in kr_d]

# plot means
scatter(npss_mean, corr_mean)
xlabel("NPSS"); ylabel("Correlation coefficient")
savefig("npss_corr.png")
print("Plotted npss_corr.png")
clf()
scatter(npss_mean, vp_d_mean)
xlabel("NPSS"); ylabel("V-P")
savefig("npss_vp.png")
print("Plotted npss_vp.png")
clf()
scatter(npss_mean, kr_d_mean)
xlabel("NPSS"); ylabel("Kreuz")
savefig("npss_kr.png")
print("Plotted npss_kr.png")
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
f.savefig("npss_corr_ind.png")
print("Plotted npss_corr_ind.png")
f.savefig("npss_corr_ind.png")
f = figure(2)
xlabel("NPSS"); ylabel("V-P")
f.savefig("npss_vp_ind.png")
print("Plotted npss_vp_ind.png")
f = figure(3)
xlabel("NPSS"); ylabel("Kreuz")
f.savefig("npss_kr_ind.png")
print("Plotted npss_kr_ind.png")


