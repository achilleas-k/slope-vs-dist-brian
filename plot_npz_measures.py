from __future__ import print_function
import os
import sys
import numpy as np
from matplotlib.pyplot import (scatter, xlabel, ylabel, savefig, clf, figure,
                               yticks, xticks, axis, subplot, subplots_adjust)


measure_npz = np.load(sys.argv[1])
npss = []
modulus = []
vp = []
kr = []
Sin = []
for k, v in measure_npz.iteritems():
    v = v.item()  # this again
    if np.any(np.isnan(v["npss"])):
        continue
    npss.append(np.mean(v["npss"]))
    modulus.append(v["modulus"][0])
    vp.append(v["vp"][0])
    kr.append(v["kreuz"][0])

try:
    os.mkdir("plots")
except OSError:
    pass

imgtypes = ["eps", "png", "pdf"]
textsize = 20

yt_pos = [0, max(npss)/2, max(npss)]
figure("comparison", figsize=(14,4), dpi=100)
subplot(131)
scatter(vp, npss, c='k')
ylabel("NPSS", size=textsize); xlabel("$D_V$", size=textsize)
xt = [0, round(max(vp))/2, round(max(vp))]
xticks(xt, xt, size=textsize)
yticks(yt_pos, yt_pos, size=textsize)
axis(ymin=0, ymax=max(npss))

subplot(132)
scatter(kr, npss, c='k')
xlabel("$D_S$", size=textsize)
xt = [0, 0.07, 0.13]  # set by hand
xticks(xt, xt, size=textsize)
yticks(yt_pos, [])
axis(ymin=0, ymax=max(npss))

subplot(133)
scatter(modulus, npss, c='k')
xlabel("$D_m$", size=textsize)
xt = [0, 0.15, 0.3]
xticks(xt, xt, size=textsize)
yticks(yt_pos, [])
axis(ymin=0, ymax=max(npss))

subplots_adjust(left=0.15, right=0.9, top=0.95, bottom=0.18, wspace=0.1)
for ext in imgtypes:
    savefig("plots/npss_comparison.%s" % ext)
    print("Plotted npss_comparison.%s" % ext)

print("--\nCorrelation between means:")
print("\tNPSS\t$D_V$\t$D_S$\t$D_m$")
labels = ["NPSS", "$D_V$", "$D_S$", "$D_m$"]

# import IPython; IPython.embed()
for cc, l in zip(np.corrcoef((npss, vp, kr, modulus)),
                 labels):
    print(l, end="")
    for c in cc:
        print("\t%0.2f" % (c), end="")
    print("")

print("--\nDone!")


