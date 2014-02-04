import sys
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
_num_items = len(spikes)
pool = mp.Pool()
modulus_iter = pool.imap(call_interval_mm, spikes)
print("Starting calculation of modulus ...")
modulus = []
for idx, mod_item in enumerate(modulus_iter, 1):
    modulus.append(mod_item)
    print("%i/%i complete ..." % (idx, _num_items))
print("Calculation complete. Saving to modulus.npz")
np.savez("modulus.npz", data=modulus)
print("DONE!")


