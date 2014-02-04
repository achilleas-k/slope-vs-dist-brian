import sys
sys.path.append("./spikerlib.egg")
from spikerlib.metrics import kreuz as kr
import numpy as np
import multiprocessing as mp


def call_interval_kr(record):
    id = record["id"]
    inspikes = record["inspikes"]
    outspikes = record["outspikes"]
    dist_kr = np.mean(kr.interval(inspikes, outspikes, mp=False))
    return {"id": id, "kr": dist_kr}

spikes = np.load("2014-01-31_spikes.npz")["data"]
_num_items = len(spikes)
pool = mp.Pool()
kreuz_iter = pool.imap(call_interval_kr, spikes)
print("Starting calculation of kreuz ...")
dist_kr = []
for idx, kr_item in enumerate(kreuz_iter, 1):
    dist_kr.append(kr_item)
    print("%i/%i complete ..." % (idx, _num_items))
print("Calculation complete. Saving to kreuz.npz")
np.savez("kreuz.npz", data=dist_kr)
print("DONE!")


