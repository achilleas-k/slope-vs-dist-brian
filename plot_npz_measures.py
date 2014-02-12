from __future__ import print_function
import os
import sys
import numpy as np
from matplotlib.pyplot import scatter, xlabel, ylabel, savefig, clf


measure_npz = np.load(sys.argv[1])
nitems = len(measure_npz.keys())
npss = []
modulus = []
vp = []
kr = []
corr = []
for k, v in measure_npz.iteritems():
    v = v.item()  # this again
    if np.any(np.isnan(v["npss"])):
        continue
    npss.append(np.mean(v["npss"]))
    modulus.append(v["modulus"][0])
    vp.append(v["vp"][0])
    kr.append(v["kreuz"][0])
    corr.append(np.random.random())

try:
    os.mkdir("plots")
except OSError:
    pass

imgtypes = ["eps", "png", "pdf"]
scatter(npss, corr)
xlabel("NPSS"); ylabel("Correlation coefficient")
for ext in imgtypes:
    savefig("plots/npss_corr.%s" % ext)
    print("Plotted npss_corr.%s" % ext)
clf()
scatter(npss, vp)
xlabel("NPSS"); ylabel("V-P")
for ext in imgtypes:
    savefig("plots/npss_vp.%s" % ext)
    print("Plotted npss_vp.%s" % ext)
clf()
scatter(npss, kr)
xlabel("NPSS"); ylabel("Kreuz")
for ext in imgtypes:
    savefig("plots/npss_kr.%s" % ext)
    print("Plotted npss_kr.%s" % ext)
clf()
scatter(npss, modulus)
xlabel("NPSS"); ylabel("Modulus")
for ext in imgtypes:
    savefig("plots/npps_modm.%s" % ext)
    print("Plotted npss_modm.%s" % ext)
clf()

print("--\nCorrelation between means:")
print("\tNPSS\tCorr\tV-P\tKreuz")
labels = ["NPSS", "Corr", "V-P", "Kreuz", "Modulus"]

#import IPython; IPython.embed()
for cc, l in zip(np.corrcoef((npss, corr,
                              vp, kr, modulus)),
                 labels):
    print(l, end="")
    for c in cc:
        print("\t%0.2f" % (c), end="")
    print("")

print("--\nDone!")


