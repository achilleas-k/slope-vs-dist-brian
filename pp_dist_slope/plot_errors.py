"""
- Fit curve to data
- Plot absolute deviations from curve vs:
    - Sigma
    - Nrand
    - frand
    - Nrand*frand
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
frand = data["randrate"]
sigma = data["sigma"]

# strip data with no random spike trains
idx = nrand > 0
slopes = slopes[idx]
distances = distances[idx]
nrand = nrand[idx]
frand = frand[idx]
sigma = sigma[idx]

# fit the curve
popt, pcov = curve_fit(curve, slopes, distances)
curve_points = np.polyval(popt, slopes)
errors = abs(distances-curve_points)/curve_points

plt.subplot(2,2,1)
plt.scatter(sigma, errors)
plt.xlabel("sigma")
plt.ylabel("error")
plt.subplot(2,2,2)
plt.scatter(nrand, errors)
plt.xlabel("Nrand")
plt.ylabel("error")
plt.subplot(2,2,3)
plt.scatter(frand, errors)
plt.xlabel("f rand")
plt.ylabel("error")
plt.subplot(2,2,4)
plt.scatter(frand*nrand, errors)
plt.xlabel("f*n")
plt.ylabel("error")

plt.show()

