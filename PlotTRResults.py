import sys
import numpy as np
from matplotlib import pyplot as plt

filename = sys.argv[1]
print("Loading ...")
data = np.load(filename)

results = np.array([data["results"]])[0]  # yeah, I know it's weird
configs = data["configurations"]

slopes = [res["slopes"] for res in results.itervalues()]
npss = [res["npss"] for res in results.itervalues()]
vp_dists = [res["vp_dists"] for res in results.itervalues()]
kr_dists = [res["kr_dists"] for res in results.itervalues()]
correlations = [res["correlations"] for res in results.itervalues()]


avg_slope = np.array([np.mean(slps)
                   if len(slps) else 0
                   for slps in slopes])
avg_npss = np.array([np.mean(slps)
                  if len(slps) else 0
                  for slps in npss])
avg_vp = np.array([np.mean(vp)
                if len(vp) else 0
                for vp in vp_dists])
avg_kr = np.array([np.mean(kr)
                if len(kr) else 0
                for kr in kr_dists])
avg_corr = np.array([np.mean(cr)
                  if len(cr) else 0
                  for cr in correlations])

std_slope = np.array([np.std(slps)
                   if len(slps) else 0
                   for slps in slopes])
std_npss = np.array([np.std(slps)
                  if len(slps) else 0
                  for slps in npss])
std_vp = np.array([np.std(vp)
                if len(vp) else 0
                for vp in vp_dists])
std_kr = np.array([np.std(kr)
                if len(kr) else 0
                for kr in kr_dists])
std_corr = np.array([np.std(cr)
                  if len(cr) else 0
                  for cr in correlations])

print("Plotting ...")
plt.figure()
plt.errorbar(avg_slope, avg_vp, xerr=std_slope, yerr=std_vp, fmt="k.")
plt.title("Average slope vs average V-P")
plt.xlabel("Slope")
plt.ylabel("V-P")
plt.savefig("slope_vs_vp2.png")

plt.clf()
plt.figure()
plt.errorbar(avg_npss, avg_vp, xerr=std_npss, yerr=std_vp, fmt="k.")
plt.title("Average normalised slope vs average V-P")
plt.xlabel("Normalised slope")
plt.ylabel("V-P")
plt.savefig("npss_vs_vp2.png")

plt.figure()
plt.errorbar(avg_slope, avg_kr, xerr=std_slope, yerr=std_kr, fmt="k.")
plt.title("Average slope vs average Kreuz")
plt.xlabel("Slope")
plt.ylabel("Kreuz")
plt.savefig("slope_vs_kr2.png")

plt.clf()
plt.figure()
plt.errorbar(avg_npss, avg_kr, xerr=std_npss, yerr=std_kr, fmt="k.")
plt.title("Average normalised slope vs average Kreuz")
plt.xlabel("Normalised slope")
plt.ylabel("Kreuz")
plt.savefig("npss_vs_kr2.png")

plt.clf()
plt.figure()
plt.errorbar(avg_slope, avg_corr, xerr=std_slope, yerr=std_corr, fmt="k.")
plt.title("Average slope vs average correlation coefficient")
plt.xlabel("Slope")
plt.ylabel("Corr coef")
plt.savefig("slope_vs_cc2.png")

plt.clf()
plt.figure()
plt.errorbar(avg_npss, avg_corr, xerr=std_npss, yerr=std_corr, fmt="k.")
plt.title("Average normalised slope vs average correlation coefficient")
plt.xlabel("Noormalised slope")
plt.ylabel("Corr coef")
plt.savefig("npss_vs_cc2.png")

print("Done!")

