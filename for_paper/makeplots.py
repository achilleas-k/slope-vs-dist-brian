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

### All data points
mnpss = np.append(get_mnpss(results_nj), get_mnpss(results_j))
kreuz = np.append(get_kreuz(results_nj), get_kreuz(results_j))
jitters = np.append(get_param(configs_nj, "sigma"),
                    get_param(configs_j, "sigma"))
fig = plt.figure("All data points", dpi=100, figsize=(8,6))
plt.scatter(mnpss, kreuz, c=jitters)
cbar = plt.colorbar()
plt.xlabel("NPSS $\bar{M}$")
plt.ylabel("SPIKE-distance $D_S$")
plt.axis(xmin=0, xmax=1, ymin=0)
plt.show()

### Split jitter from no-jitter
njidx = jitters == 0
fig = plt.figure("No jitter", dpi=100, figsize=(8,3))
pjidx = jitters > 0


