import sys
from spikerlib.metrics import corrcoef as cc
import numpy as np
import multiprocessing as mp


def call_interval_cc(record):
    id = record["id"]
    inspikes = record["inspikes"]
    outspikes = record["outspikes"]
    corrcoef = np.mean(cc.interval(inspikes, outspikes, b=0.002, duration=5))
    return {"id": id, "corr": corrcoef}

spikes = np.load("2014-01-31_spikes.npz")["data"]
_num_items = len(spikes)
pool = mp.Pool()
corrcoef_iter = pool.imap(call_interval_cc, spikes)
print("Starting calculation of correlation ...")
corrcoef = []
for idx, cc_item in enumerate(corrcoef_iter, 1):
    corrcoef.append(cc_item)
    print("%i/%i complete ..." % (idx, _num_items))
print("Calculation complete. Saving to correlation.npz")
np.savez("correlation.npz", data=corrcoef)
print("DONE!")


