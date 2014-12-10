"""
Attempt to find a functional between NPSS and S_m, determined by <V> and
Delta_v; NPSS = f(S_m, <V>, Delta_v) ???
"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
_ = Axes3D  # silence warning about unused import
from scipy.optimize import curve_fit
from glob import glob

def get_mnpss(results):
    return [np.mean(r["npss"]) for r in results]

def get_kreuz(results):
    return [np.trapz(r["kr_dists"], r["kr_times"]) for r in results]

def get_param(configs, paramkey):
    return [c[paramkey] for c in configs]

def read_it_all_in(globstring):
    print("Loading data...")
    mnpss = []
    kreuz = []
    synchs = []
    jitters = []
    inrates = []
    numin = []
    weights = []
    for npzfile in glob(globstring):
        print("Reading {}".format(npzfile))
        data = np.load(npzfile)
        configs = data["configs"]
        results = data["results"]
        mnpss.extend(get_mnpss(results))
        kreuz.extend(get_kreuz(results))
        synchs.extend(get_param(configs, "S_in"))
        jitters.extend(get_param(configs, "sigma"))
        inrates.extend(get_param(configs, "f_in"))
        numin.extend(get_param(configs, "N_in"))
        weights.extend(get_param(configs, "weight"))
    mnpss = np.array(mnpss)
    kreuz = np.array(kreuz)
    synchs = np.array(synchs)
    jitters = np.array(jitters)
    inrates = np.array(inrates)
    numin = np.array(numin)
    weights = np.array(weights)
    # sort based on increasing mnpss (makes plotting lines over mnpss simpler)
    sorted_idx = np.argsort(mnpss)
    mnpss = mnpss[sorted_idx]
    kreuz = kreuz[sorted_idx]
    jitters = jitters[sorted_idx]
    inrates = inrates[sorted_idx]
    numin = numin[sorted_idx]
    weights = weights[sorted_idx]
    return mnpss, kreuz, jitters, inrates, numin, weights



plt.rcParams['image.cmap'] = 'gray'
vth = 0.015  # threshold value used in simulations
tau = 0.01
mnpss, kreuz, jitters, inrates, numin, weights = read_it_all_in("../npzfiles/*.npz")

# drive and volley peaks
drive = (inrates*numin*weights*tau).astype("float")
peaks = (numin*weights).astype("float")

# get rid of excessively high drive
nd_idx = drive <= 0.04
mnpss = mnpss[nd_idx]
kreuz = kreuz[nd_idx]
jitters = jitters[nd_idx]
inrates = inrates[nd_idx]
numin = numin[nd_idx]
weights = weights[nd_idx]

drive = drive[nd_idx]
peaks = peaks[nd_idx]


# colour the four cases
cases = []
for d, p in zip(drive, peaks):
    case = 0
    if (p < vth) and (d < vth):
        case = 1
    elif (p >= vth) and (d < vth):
        case = 2
    elif (p < vth) and (d >= vth):
        case = 3
    elif (p >= vth) and (d >= vth):
        case = 4
    else:
        raise Exception("WAT")
    cases.append(case)
cases = np.array(cases)

# print cases as colours over the two variables
plt.figure("case colours")
plt.scatter(mnpss, kreuz, c=cases)
plt.xlabel(r"$\overline{M}$")
plt.ylabel(r"$S_{m}$")
cbar = plt.colorbar()
cbar.set_label("case")
plt.savefig("npss_kreuz_cases.pdf")
plt.savefig("npss_kreuz_cases.png")

# let's get a simplified view. Let's see how NPSS/S_m changes with
# different combinations of <V> and Delta_v
#fig = plt.figure("ratio")
#ax = fig.add_subplot(111, projection="3d")
#ax.scatter3D(drive, peaks, mnpss/kreuz)
#ax.set_xlabel(r"$\langle V \rangle$")
#ax.set_ylabel(r"$\Delta_{v}$")
#ax.set_zlabel(r"$\overline{M}/S_{m}$")
#plt.show()

# ignore the peak
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter3D(mnpss, kreuz, drive, c=drive)
ax.set_xlabel(r"$\overline{M}$")
ax.set_ylabel(r"$S_{m}$")
ax.set_zlabel(r"$\langle V \rangle$")
plt.show()
