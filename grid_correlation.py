import sys
sys.path.append("./spikerlib.egg")
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


pool = mp.Pool()
corrcoef = pool.map(call_interval_cc, spikes)
np.savez("correlation.npz", data=corrcoef)


