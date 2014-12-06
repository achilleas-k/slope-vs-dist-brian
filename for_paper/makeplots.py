from __future__ import division, print_function
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
    allx = np.arange(0, 1.5, 0.0001)
    ally = np.polyval(coef, allx)
    errors = []
    for xi, yi in zip(x, y):
        distances = (xi-allx)**2+(yi-ally)**2
        errors.append(min(distances))
    return np.sqrt(errors)

def plot_errors(x, y, coef):
    allx = np.arange(0, 1.5, 0.0001)
    ally = np.polyval(coef, allx)

    plt.figure("deviations (on x)")
    plt.plot(allx, ally, 'k')
    for xi, yi in zip(x, y):
        plt.plot([xi, xi], [yi, np.polyval(coef, xi)], c='black',
                marker='.', linestyle='-')
    plt.savefig("deviations_x.png")

    plt.figure("deviations (from curve)")
    plt.plot(allx, ally, 'k')
    for xi, yi in zip(x, y):
        distances = (xi-allx)**2+(yi-ally)**2
        minidx = np.argmin(distances)
        minx = allx[minidx]
        miny = ally[minidx]
        plt.plot([xi, minx], [yi, miny], c='black',
                 marker='.', linestyle='-')
    plt.savefig("deviations_curve.png")

def clip(x, min, max):
    x[x < min] = min
    x[x > max] = max
    return x


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
inrates = inrates[sorted_idx]
numin = numin[sorted_idx]
weights = weights[sorted_idx]

drive = (inrates*numin*weights*0.01).astype("float")

print("Fitting curves...")
curvefunc = cf3
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
vmax = 1.0
colour = clip(jitters*10000, 0, vmax)
plt.subplot2grid((11,11), (0,0), rowspan=4, colspan=10)
plt.title("(a)"+" "*110)
allpts = plt.scatter(mnpss, kreuz, vmin=0, vmax=vmax, c=colour)# c=jitters*1000)
plt.plot(mnpss, curvepts, color="black", linestyle="-", linewidth=5, alpha=0.5)
plt.xlabel(r"$\overline{M}$")
plt.ylabel(r"$D_S$")
plt.axis(xmin=0, xmax=1, ymin=0)

### Split jitter from no-jitter
plt.subplot2grid((11,11), (6,0), rowspan=4, colspan=4)
plt.title("(b)"+" "*55)
njpts = plt.scatter(mnpss[njidx], kreuz[njidx], vmin=0, vmax=vmax, c=colour[njidx])# c=jitters[njidx]*1000)
plt.plot(mnpss, curvepts, color="black", linestyle="-", linewidth=5, alpha=0.5)
plt.xlabel(r"$\overline{M}$")
plt.ylabel(r"$D_S$")
plt.axis(xmin=0, xmax=1, ymin=0)

plt.subplot2grid((11,11), (6,6), rowspan=4, colspan=4)
plt.title("(c)"+" "*55)
plt.scatter(mnpss[pjidx], kreuz[pjidx], vmin=0, vmax=vmax, c=colour[pjidx])# c=jitters[pjidx]*1000)
plt.plot(mnpss, curvepts, color="black", linestyle="-", linewidth=5, alpha=0.5)
plt.xlabel(r"$\overline{M}$")
plt.ylabel(r"$D_S$")
plt.axis(xmin=0, xmax=1, ymin=0)

cax = fig.add_axes([0.9, 0.15, 0.03, 0.75])
cbar = plt.colorbar(allpts, cax=cax)
cbar.set_label(r"$\sigma_{in}$ (ms)")

plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.savefig("npss_v_dist.pdf")
plt.savefig("npss_v_dist.png")

print("Fitted curve coefficients: {}".format(", ".join(str(p) for p in popt)))

print("Calculating and plotting deviations...")
allerrors = calcerrors(mnpss, kreuz, popt)
njerrors = allerrors[njidx]
pjerrors = allerrors[pjidx]
plt.figure("Jitter vs Errors")
allerrs = plt.scatter(jitters, allerrors, c=drive)
cbar = plt.colorbar(allerrs)
cbar.set_label(r"$\langle V \rangle$")
plt.xlabel(r"$\sigma_{in}$ (ms)")
plt.ylabel("Abs. error")
plt.savefig("jitters_v_errors.pdf")
plt.savefig("jitters_v_errors.png")

plot_errors(mnpss, kreuz, popt)

print("Plotting error histograms (with jitter)...")
histbins = np.arange(0, 1.001, 0.2)
plt.figure("Histograms (jitter)")
njhy, njhx = np.histogram(njerrors, density=True, bins=histbins)
pjhy, pjhx = np.histogram(pjerrors, density=True, bins=histbins)
allhy, allhx = np.histogram(allerrors, density=True, bins=histbins)
plt.plot(njhx[:-1], njhy, c="black", label="$\sigma_{in} = 0$ ms")
plt.plot(pjhx[:-1], pjhy, c="grey", label="$\sigma_{in} > 0$ ms")
plt.plot(allhx[:-1], allhy, c="black", linestyle="--", label="All samples")
plt.legend(loc="best")
plt.savefig("error_hist_jitter.pdf")
plt.savefig("error_hist_jitter.png")

print("Average absolute errors")
print("Without jitter: {}".format(np.mean(njerrors)))
print("With jitter:    {}".format(np.mean(pjerrors)))
print("All:            {}".format(np.mean(allerrors)))

print("Plotting results with drive...")
fig = plt.figure("NPSS vs SPIKE-distance with drive", dpi=100, figsize=(8,6))
idx = drive > 0.02
allpts = plt.scatter(mnpss[idx], kreuz[idx], c=drive[idx])
plt.plot(mnpss, curvepts, color="black", linestyle="-", linewidth=5, alpha=0.5)
plt.xlabel(r"$\overline{M}$")
plt.ylabel(r"$D_S$")
plt.axis(xmin=0, xmax=1, ymin=0)
cbar = plt.colorbar(allpts)
cbar.set_label(r"$\langle V \rangle$")
plt.savefig("npss_v_dist_drive.pdf")
plt.savefig("npss_v_dist_drive.png")

plt.figure("Drive vs Errors")
plt.scatter(drive, allerrors, c="black")
plt.plot([0.015, 0.015], [0, 1], c="black", linestyle="--")
plt.plot([0.040, 0.040], [0, 1], c="black", linestyle="--")
plt.xlabel(r"$\langle V \rangle$")
plt.ylabel("Abs. error")
plt.savefig("drive_v_errors.pdf")
plt.savefig("drive_v_errors.png")

print("Plotting error histograms (with drive)...")
plt.figure("Histograms (drive)")
lowseg, highseg = 0.015, 0.04
hderrors = allerrors[drive>=highseg]
gderrors = allerrors[(drive>=lowseg) & (drive < highseg)]
lderrors = allerrors[drive<lowseg]
hdhy, hdhx = np.histogram(hderrors, density=True, bins=histbins)
gdhy, gdhx = np.histogram(gderrors, density=True, bins=histbins)
ldhy, ldhx = np.histogram(lderrors, density=True, bins=histbins)
allhy, allhx = np.histogram(allerrors, density=True, bins=histbins)
plt.plot(hdhx[:-1], hdhy, c="grey", linestyle="--", label=r"$\langle V \rangle \geq {} mV$".format(highseg*1000))
plt.plot(gdhx[:-1], gdhy, c="black", label=r"${} mV \leq \langle V \rangle < {} mV$".format(lowseg*1000, highseg*1000))
plt.plot(ldhx[:-1], ldhy, c="grey", label=r"$\langle V \rangle < {} mV$".format(lowseg*1000))
plt.plot(allhx[:-1], allhy, c="black", linestyle="--", label="All samples")
plt.legend(loc="best")
plt.savefig("error_hist_drive.pdf")
plt.savefig("error_hist_drive.png")

print("Average absolute errors")
print("High drive: {}".format(np.mean(hderrors)))
print("Good drive: {}".format(np.mean(gderrors)))
print("Low drive : {}".format(np.mean(lderrors)))
# Statistical significance?
print("Done")

#Make histograms for errors against all individual parameters
#Count the total number of errors in each bin and make a point about "higher probablility of large errors" for cases
