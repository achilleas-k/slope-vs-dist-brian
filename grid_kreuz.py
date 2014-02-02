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


pool = mp.Pool()
dist_kr = pool.map(call_interval_kr, spikes)
np.savez("kreuz.npz", data=dist_kr)


