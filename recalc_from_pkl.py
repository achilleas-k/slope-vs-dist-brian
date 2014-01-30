from __future__ import print_function
import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
import spikerlib.metrics as metrics
from spikerlib.tools import normalised_pre_spike_slopes
import multiprocessing as mp


def calc_metrics(conres):
    config = conres[0]
    result = conres[1]
    _id = result["id"]
    inspikes = result["inspikes"]
    outspikes = result["outspikes"]
    mem = result["mem"]
    npss = result["npss"]  # trusting this one - the rest will be recalced
    vp_dists = metrics.victor_purpura.interval(inspikes, outspikes,
                                               cost=1.0/0.002, mp=True)
    kr_dists = metrics.kreuz.interval(inspikes, outspikes)
    corrs = metrics.corrcoef.interval_corr(inspikes, outspikes, 0.002)
    return {"id": _id,
            "vp_dists": vp_dists,
            "kr_dists": kr_dists,
            "correlations": corrs}




pickle_file = sys.argv[1]
data = pickle.load(open(pickle_file))
results = data["results"]
config = data["config"]

_numitems = len(results)

pool = mp.Pool()
metric_calc_iter = pool.imap(calc_metrics, zip(config[:3], results[:3]))
recalculated_metrics = []
for idx, recmet in enumerate(metric_calc_iter, 1):
    recalculated_metrics.append(recmet)
    print("\r%i/%i ..." % (idx, _numitems))
    sys.stdout.flush()
print("Calcultion complete")

# just dump for now - I need to get home
pickle.dump(recalculated_metrics, open("recalculated_metrics.pkl", "w"))


