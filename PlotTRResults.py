import sys
import numpy as np

filename = sys.argv[1]
data = np.load(filename)

results = np.array(data["results"])[0]  # yeah, I know it's weird
configs = data["configurations"]

slopes = results["slopes"]
npss = results["npss"]
vp_dists = results["np_dists"]
kr_dists = results["kr_dists"]
correlations = results["correlations"]


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
