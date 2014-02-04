import sys
from spikerlib.metrics import victor_purpura as vp
import numpy as np
import multiprocessing as mp


def call_interval_vp(record):
    id = record["id"]
    inspikes = record["inspikes"]
    outspikes = record["outspikes"]
    dist_vp = np.mean(vp.interval(inspikes, outspikes, 1/0.002, mp=False))
    return {"id": id, "vp": dist_vp}

spikes = np.load("2014-01-31_spikes.npz")["data"]
_num_items = len(spikes)
pool = mp.Pool()
dist_vp_iter = pool.imap(call_interval_vp, spikes)
print("Starting calculation of victor-purpura ...")
dist_vp = []
for idx, vp_item in enumerate(dist_vp_iter, 1):
    dist_vp.append(vp_item)
    print("%i/%i complete ..." % (idx, _num_items))
print("Calculation complete. Saving to victor_purpura.npz")
np.savez("victor_purpura.npz", data=dist_vp)
print("DONE!")


