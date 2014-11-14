import sys
sys.path.append("./spikerlib.egg")
from spikerlib.metrics import kreuz as kr
import numpy as np
import multiprocessing as mp


def call_mulitvar_kr(record):
    id = record["id"]
    inspikes = record["inspikes"]
    start, end = 0, max([max(insp) for insp in inspikes])
    samples = int(end/0.001)  # 1 sample per millisecond
    dist_kr = kr.multivariate(inspikes, start, end, samples)
    return {"id": id, "kr": dist_kr}

spikes = np.load("2014-01-31_spikes.npz")["data"]
_num_items = len(spikes)
pool = mp.Pool()
kreuz_iter = pool.imap(call_mulitvar_kr, spikes)
print("Starting calculation of kreuz ...")
dist_kr = []
for idx, kr_item in enumerate(kreuz_iter, 1):
    dist_kr.append(kr_item)
    print("%i/%i complete ..." % (idx, _num_items))
print("Calculation complete. Saving to kreuz.npz")
np.savez("kreuz.npz", data=dist_kr)
print("DONE!")


