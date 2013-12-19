import numpy as np



avg_slope = np.array([np.mean(slps)
                   if len(slps) else 0
                   for slps in slopes_collection])
avg_npss = np.array([np.mean(slps)
                  if len(slps) else 0
                  for slps in npss_collection])
avg_vp = np.array([np.mean(vp)
                if len(vp) else 0
                for vp in vp_dist_collection])
avg_kr = np.array([np.mean(kr)
                if len(kr) else 0
                for kr in kr_dist_collection])
avg_corr = np.array([np.mean(cr)
                  if len(cr) else 0
                  for cr in corr_collection])


