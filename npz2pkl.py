"""
Load all the results and configuration data from directories containing
multiple .npz files and save them in a pickle. Strict data structure is
assumed, as created from DataGenTaskFarmMC.py.
"""

from __future__ import print_function
import os
import sys
import pickle
import numpy as np
import time

directories = sys.argv[1:]

results = []
configs = []
for datadir in directories:
    print("Loading files from %s ..." % (datadir))
    filesindir = os.listdir(datadir)
    npzindir = [fid for fid in filesindir if fid.endswith("npz")]
    npzcount = len(npzindir)
    for idx, npzfile in enumerate(npzindir):
        npzdata = np.load(os.path.join(datadir, npzfile))
        res = npzdata["results"].item()
        conf = npzdata["config"].item()
        results.append(res)
        configs.append(conf)
        print("Finished reading %s (%i/%i)" % (npzfile, idx+1, npzcount),
              end="\r")
    print("")


# Load the data from the old pickle
oldpickle_file = "pkl/metric_comp_results.pkl"
if not os.path.exists(oldpickle_file):
    print("Old pickle data not found. Leaving results and configs as is.")
else:
    print("Loading data from old pickle file: %s" % (oldpickle_file))
    oldpickle_data = pickle.load(open(oldpickle_file))
    results = oldpickle_data
    print("Organising configs into lists ...")
    N_in = [conf["N_in"] for conf in configs]
    S_in = [conf["S_in"] for conf in configs]
    f_in = [conf["f_in"] for conf in configs]
    uuid = [conf["id"]   for conf in configs]
    sigma = [conf["sigma"] for conf in configs]
    weight = [conf["weight"] for conf in configs]

    configs = {"N_in": N_in,
               "S_in": S_in,
               "f_in": f_in,
               "id": uuid,
               "sigma": sigma,
               "weight": weight, }

conf_res = {"config": configs,
            "results": results, }

datenow = time.localtime()
isodate = "%i-%02i-%02i" % (datenow.tm_year, datenow.tm_mon, datenow.tm_mday)
newpickle_file = "pkl/metric_comp_results_%s.pkl" % (isodate)
pickle.dump(conf_res, open(newpickle_file, "w"))

print("Saved everything to %s" % (newpickle_file))
print("Done!")

