import sys
sys.path.append("./spikerlib.egg")
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


pool = mp.Pool()
dist_vp = pool.map(call_interval_vp, spikes)
np.savez("victor_purpura.npz", data=dist_vp)


