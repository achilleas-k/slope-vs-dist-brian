import sys
sys.path.append("./spikerlib.egg")
from spikerlib.metrics import modulus_metric as mm
import numpy as np
import multiprocessing as mp


def call_interval_mm(record):
    id = record["id"]
    inspikes = record["inspikes"]
    outspikes = record["outspikes"]
    modulus = np.mean(mm.interval(inspikes, outspikes, 0, 5, mp=False))
    return {"id": id, "modulus": modulus}

spikes = np.load("2014-01-31_spikes.npz")["data"]


pool = mp.Pool()
modulus = pool.map(call_interval_mm, spikes)
np.savez("modulus.npz", data=modulus)


