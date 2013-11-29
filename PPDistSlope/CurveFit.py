"""
Fit a curve to the slope vs dist scatter
"""
import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def curve(x, a, b, c):
    """
    The function to fit to the data
    """
    return a*x**2+b*x+c

filename = sys.argv[1]
data = np.load(filename)

slopes = data["slope"]
distances = data["dist"]
nrand = data["Nrand"]

# strip out data with no random spike trains
idx = nrand > 0
slopes = slopes[idx]
distances = distances[idx]

# sort by x-axis (slopes)
srtidx = np.argsort(slopes)
slopes = slopes[srtidx]
distances = distances[srtidx]

# fit the curve
popt, pcov = curve_fit(curve, slopes, distances)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(slopes, distances, c="gray")
ax.plot(slopes, np.polyval(popt, slopes), "r-")
ax.grid()
plt.xlabel("Slope (V/s)")
plt.ylabel("Spike time distance")
plt.savefig("svd_trend.png")


