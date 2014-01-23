from __future__ import print_function
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np

pickle_file = sys.argv[1]  # screw input checking
if len(sys.argv) > 2:
    colour_attr = sys.argv[2]  # attribute/parameter to be used for third dim.
else:
    colour_attr = False


data = pickle.load(open(pickle_file))
results = data["results"]
config = data["config"]

npss = results["npss_mean"]
corr = results["corr_mean"]
vp_d = results["vp_d_mean"]
kr_d = results["kr_d_mean"]
mm_d = results["mm_d_mean"]

if colour_attr:
    try:
        param = config[colour_attr]
    except KeyError:
        print("No such attribute \"%s\". Parameter must be one of the following:")
        print(", ".join(config.keys()))
        sys.exit(1)
else:
    param = "b"  # blue dots


plt.scatter(npss, corr, c=param)
plt.xlabel("NPSS")
plt.ylabel("Correlation coefficient")
if colour_attr:
    cbar = plt.colorbar()
    cbar.set_label(colour_attr)
#plt.axis([0, 1, 0, 0.15])
plt.savefig("plots/npss_corr.png")
print("Plotted npss_corr.png")

plt.clf()
plt.scatter(npss, vp_d, c=param)
plt.xlabel("NPSS")
plt.ylabel("V-P")
if colour_attr:
    cbar = plt.colorbar()
    cbar.set_label(colour_attr)
#plt.axis([0, 1, 0, 2])
plt.savefig("plots/npss_vp.png")
print("Plotted npss_vp.png")

plt.clf()
plt.scatter(npss, kr_d, c=param)
plt.xlabel("NPSS")
plt.ylabel("Kreuz")
if colour_attr:
    cbar = plt.colorbar()
    cbar.set_label(colour_attr)
#plt.axis([0, 1, 0, 0.15])
plt.savefig("plots/npss_kr.png")
print("Plotted npss_kr.png")

plt.clf()
plt.scatter(npss,mm_d, c=param)
plt.xlabel("NPSS")
plt.ylabel("Modulus metric")
if colour_attr:
    cbar = plt.colorbar()
    cbar.set_label(colour_attr)
#plt.axis([0, 1, 0, 0.002])
plt.savefig("plots/npss_modm.png")
print("Plotted npss_modm.png")

print("Correlation between means:")
print("\tNPSS\tCorr\tV-P\tKreuz\tMod")
labels = ["NPSS", "Corr", "V-P", "Kreuz", "Mod"]
for cc, l in zip(np.corrcoef((npss, corr,
                              vp_d, kr_d, mm_d)),
                 labels):
    print(l, end="")
    for c in cc:
        print("\t%0.2f" % (c), end="")
    print("")

print("Done!")

