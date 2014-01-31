"""
Load all the results and configuration data from directories containing
multiple .npz files and save them in a pickle. Strict data structure is
assumed, as created from DataGenTaskFarmMC.py.
"""

from __future__ import print_function
import os
import sys
import numpy as np
import time

directories = sys.argv[1:]

configs = []
recorded = []
npss = []
for datadir in directories:
    print("Loading files from %s ..." % (datadir))
    filesindir = os.listdir(datadir)
    npzindir = [fid for fid in filesindir if fid.endswith("npz")]
    npzcount = len(npzindir)
    for idx, npzfile in enumerate(npzindir):
        npzdata = np.load(os.path.join(datadir, npzfile))
        res_all = npzdata["results"].item()
        conf = npzdata["config"].item()
        id = res_all["id"]
        rec = {
            "id": res_all["id"],
            "inspikes": res_all["inspikes"],
            "outspikes": res_all["outspikes"],
            "mem": res_all["mem"]
        }
        conf["id"] = id
        npss_i = {"id": id,
                  "npss": res_all["npss"]}
        configs.append(conf)
        recorded.append(rec)
        npss.append(npss_i)
        print("Finished reading %s (%i/%i)" % (npzfile, idx+1, npzcount),
              end="\r")
    print("")

datenow = time.localtime()
isodate = "%i-%02i-%02i" % (datenow.tm_year, datenow.tm_mon, datenow.tm_mday)
configs_npz = "configs_%s.npz" % isodate
np.savez(configs_npz, configs)
print("Saved %s." % (configs_npz))
recorded_npz = "monitors_%s.npz" % isodate
np.savez(recorded_npz, recorded)
print("Saved %s." % (recorded_npz))
npss_npz = "npzz_%s.npz" % isodate
np.savez(npss_npz, npss)
print("Saved %s." % (npss_npz))


print("Done!")

