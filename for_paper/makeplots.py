import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from glob import glob

def get_mnpss(results):
    return [np.mean(r["npss"]) for r in results]

def get_kreuz(results):
    return [np.trapz(r["kr_dists"], r["kr_times"]) for r in results]

def get_param(configs, paramkey):
    return [c[paramkey] for c in configs]

def cf3(x, a, b, c):
    return np.polyval([a, b, c], x)

def cf4(x, a, b, c, d):
    return np.polyval([a, b, c, d], x)

def cf5(x, a, b, c, d, e):
    return np.polyval([a, b, c, d, e], x)

def cf9(x, a, b, c, d, e, f, g, h, i):
    return np.polyval([a, b, c, d, e, f, g, h, i], x)

def calcerrors(x, y, coef):
    target = np.polyval(coef, x)
    return abs(target-y)

plt.rcParams['image.cmap'] = 'gray'

print("Loading data...")
mnpss = []
kreuz = []
synchs = []
jitters = []
inrates = []
numin = []
weights = []
for npzfile in glob("../npzfiles/*.npz"):
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

sorted_idx = np.argsort(mnpss)
mnpss = mnpss[sorted_idx]
kreuz = kreuz[sorted_idx]
jitters = jitters[sorted_idx]

print("Fitting curves...")
curvefunc = cf5
popt, pcov = curve_fit(curvefunc, mnpss, kreuz)
curvepts = curvefunc(mnpss, *popt)

njidx = jitters == 0
poptnj, pcov = curve_fit(curvefunc, mnpss[njidx], kreuz[njidx])
njcurvepts = curvefunc(mnpss[njidx], *poptnj)

pjidx = jitters > 0
poptpj, pcov = curve_fit(curvefunc, mnpss[pjidx], kreuz[pjidx])
pjcurvepts = curvefunc(mnpss[pjidx], *poptpj)

print("Number of items")
print("Without jitter: {}".format(np.count_nonzero(njidx)))
print("With jitter:    {}".format(np.count_nonzero(pjidx)))
print("Total:          {}".format(len(mnpss)))

print("Making plots...")
fig = plt.figure("NPSS vs SPIKE-distance", dpi=100, figsize=(8,6))
### All data points
plt.subplot2grid((11,11), (0,0), rowspan=4, colspan=10)
allpts = plt.scatter(mnpss, kreuz, c=jitters*1000)
plt.plot(mnpss, curvepts, color="black", linestyle="-", linewidth=5, alpha=0.5)
plt.xlabel(r"$\overline{M}$")
plt.ylabel(r"$D_S$")
plt.axis(xmin=0, xmax=1, ymin=0)

### Split jitter from no-jitter
plt.subplot2grid((11,11), (6,0), rowspan=4, colspan=4)
njpts = plt.scatter(mnpss[njidx], kreuz[njidx], c=jitters[njidx]*1000)
plt.plot(mnpss, curvepts, color="grey", linestyle="-", linewidth=5, alpha=0.5)
plt.plot(mnpss[njidx], njcurvepts, color="black", linestyle="-",
         linewidth=3, alpha=0.6)
plt.xlabel(r"$\overline{M}$")
plt.ylabel(r"$D_S$")
plt.axis(xmin=0, xmax=1, ymin=0)

plt.subplot2grid((11,11), (6,6), rowspan=4, colspan=4)
plt.scatter(mnpss[pjidx], kreuz[pjidx], c=jitters[pjidx]*1000)
plt.plot(mnpss, curvepts, color="grey", linestyle="-", linewidth=5, alpha=0.5)
plt.plot(mnpss[pjidx], pjcurvepts, color="black", linestyle="-",
         linewidth=3, alpha=0.6)
plt.xlabel(r"$\overline{M}$")
plt.ylabel(r"$D_S$")
plt.axis(xmin=0, xmax=1, ymin=0)

cax = fig.add_axes([0.9, 0.15, 0.03, 0.75])
cbar = plt.colorbar(cax=cax)
cbar.set_label(r"$\sigma_{in}$ (ms)")

plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.savefig("npss_v_dist.pdf")
plt.savefig("npss_v_dist.png")

print("Fitted curve coefficients: {}".format(", ".join(str(p) for p in popt)))

print("Calculating deviations...")
njerrors = calcerrors(mnpss[njidx], kreuz[njidx], popt)
pjerrors = calcerrors(mnpss[pjidx], kreuz[pjidx], popt)
plt.figure("Deviations")
plt.scatter(synchs[pjidx], pjerrors)
#cbar = plt.colorbar()
#cbar.set_label(r"$S_{in}$")
plt.xlabel(r"$\sigma_{in}$ (ms)")
plt.ylabel("Abs. error")
plt.savefig("deviations.pdf")
plt.savefig("deviations.png")
