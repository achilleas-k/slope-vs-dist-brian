import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import spikerlib as sl

def get_mnpss(results):
    return np.array([np.mean(r["npss"]) for r in results])

def get_kreuz(results):
    return np.array([np.trapz(r["kr_dists"], r["kr_times"]) for r in results])

def get_param(configs, paramkey):
    return np.array([c[paramkey] for c in configs])

plt.rcParams['image.cmap'] = 'gray'
data_nojitter = np.load("../2014-11-16-npz-nojitter/results.npz")
data_jitter = np.load("../2014-11-17-npz/results.npz")

configs_nj = data_nojitter["configs"]
results_nj = data_nojitter["results"]
configs_j = data_jitter["configs"]
results_j = data_jitter["results"]

fig = plt.figure("NPSS vs SPIKE-distance", dpi=100, figsize=(8,6))
mnpss = np.append(get_mnpss(results_nj), get_mnpss(results_j))
kreuz = np.append(get_kreuz(results_nj), get_kreuz(results_j))
jitters = np.append(get_param(configs_nj, "sigma"),
                    get_param(configs_j, "sigma"))
### All data points
plt.subplot2grid((11,11), (0,0), rowspan=4, colspan=10)
allpts = plt.scatter(mnpss, kreuz, c=jitters*1000)
plt.xlabel(r"$\bar{M}$")
plt.ylabel(r"$D_S$")
plt.axis(xmin=0, xmax=1, ymin=0)

### Split jitter from no-jitter
njidx = jitters == 0
plt.subplot2grid((11,11), (6,0), rowspan=4, colspan=4)
njpts = plt.scatter(mnpss[njidx], kreuz[njidx], c=jitters[njidx]*1000)
plt.xlabel(r"$\bar{M}$")
plt.ylabel(r"$D_S$")
plt.axis(xmin=0, xmax=1, ymin=0)

pjidx = jitters > 0
plt.subplot2grid((11,11), (6,6), rowspan=4, colspan=4)
plt.scatter(mnpss[pjidx], kreuz[pjidx], c=jitters[pjidx]*1000)
plt.xlabel(r"$\bar{M}$")
plt.ylabel(r"$D_S$")
plt.axis(xmin=0, xmax=1, ymin=0)

cax = fig.add_axes([0.9, 0.15, 0.03, 0.75])
cbar = plt.colorbar(cax=cax)
cbar.set_label(r"$\sigma_{in}$")

plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.savefig("npss_v_dist.eps")
