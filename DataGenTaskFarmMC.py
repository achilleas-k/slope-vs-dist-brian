"""
Monte-Carlo version of DataGenTaskFarm.py
The simulation parameters are specified by range bounds (min, max) and number
of simulations.
"""
from brian import *
from brian.tools.taskfarm import *
from brian.tools.datamanager import *
import warnings
from spike_distance import mean_pairwise_distance
from neurotools import (gen_input_groups, pre_spike_slopes,
                        normalised_pre_spike_slopes, corrcoef_spiketrains)
from spike_distance_kreuz import multivariate_spike_distance
import itertools as itt
import gc
from time import time
import os
import zipfile

defaultclock.dt = dt = 0.1*ms
duration = 5*second

warnings.simplefilter("always")
slope_w = 2*msecond
dcost = float(2/slope_w)


def interval_VP(inputspikes, outputspikes, cost, dt=0.1*ms):
    dt = float(dt)
    vpdists = []
    for prv, nxt in zip(outputspikes[:-1], outputspikes[1:]):
        interval_inputs = []
        for insp in inputspikes:
            interval_inputs.append(insp[(prv < insp) & (insp < nxt+dt)])
        vpd = mean_pairwise_distance(interval_inputs, cost)
        vpdists.append(vpd)
    return vpdists


def interval_Kr(inputspikes, outputspikes, dt=0.1*ms):
    dt = float(dt)
    krdists = []
    for prv, nxt in zip(outputspikes[:-1], outputspikes[1:]):
        krd = multivariate_spike_distance(inputspikes, prv, nxt+dt, 1)
        krdists.append(krd[1])
    return krdists


def interval_corr(inputspikes, outputspikes, dt=0.1*ms, duration=None):
    dt = float(dt)
    corrs = []
    for prv, nxt in zip(outputspikes[:-1], outputspikes[1:]):
        interval_inputs = []
        for insp in inputspikes:
            interval_spikes = insp[(prv < insp) & (insp < nxt+dt)]-prv
            if len(interval_spikes):
                interval_inputs.append(interval_spikes)
        if len(interval_inputs):
            corrs_i = mean(corrcoef_spiketrains(interval_inputs, dt, duration))
        else:
            corrs.append(0)
        corrs.append(corrs_i)
    return corrs


def lifsim(sync, sgm, inrate, n_in, weight):
    clear(True)
    gc.collect()
    reinit_default_clock()
    seed = int(time()+(sync+sgm+inrate+n_in+weight)*1e3)
    sgm *= second
    inrate *= Hz
    weight *= volt
    Vth = 15*mV
    tau = 10*ms

    # neuron
    neuron = NeuronGroup(1, "dV/dt = -V/tau : volt",
                         threshold="V>Vth", reset="V=0*mvolt", refractory=1*ms)
    neuron.V = 0*mvolt
    netw = Network(neuron)
    config = {'id': seed,  # can use seed as id instead of uuid
              'S_in': sync,
              'sigma': sgm,
              'f_in': inrate,
              'N_in': n_in,
              'weight': weight}
    # inputs
    syncInp, randInp = gen_input_groups(n_in, inrate, sync, sgm, duration, dt)
    netw.add(syncInp, randInp)
    # connections
    if len(syncInp):
        syncCon = Connection(syncInp, neuron, "V", weight=weight)
        netw.add(syncCon)
    if len(randInp):
        randCon = Connection(randInp, neuron, "V", weight=weight)
        netw.add(randCon)
    # monitors
    syncmon = SpikeMonitor(syncInp)
    randmon = SpikeMonitor(randInp)
    netw.add(syncmon, randmon)
    vmon = StateMonitor(neuron, "V", record=True)
    outmon = SpikeMonitor(neuron)
    netw.add(vmon, outmon)
    # run
    netw.run(duration, report=None)

    if outmon.nspikes < 2:
        return
    vmon.insert_spikes(outmon, Vth)
    input_spiketrains = syncmon.spiketimes.values()
    input_spiketrains += randmon.spiketimes.values()
    input_spiketrains = array(input_spiketrains)

    # calculate slopes (no normalisation)
    slopes = pre_spike_slopes(vmon[0], outmon[0], Vth, slope_w, dt)
    # normalised slopes
    npss = normalised_pre_spike_slopes(vmon[0], outmon[0], 0*mV, Vth, tau,
                                       slope_w, dt)
    # calculate mean pairwise V-P distance for each interval
    vp_dists = interval_VP(input_spiketrains, outmon[0], dcost)
    # calculate multivariate Kreuz spike distance
    kr_dists = interval_Kr(input_spiketrains, outmon[0])

    # calculate mean correlation coefficient
    corrs = interval_corr(input_spiketrains, outmon[0], dt, duration)

    results = {'id': seed,
               'mem': vmon[0],
               'outspikes': outmon[0],
               'inspikes': input_spiketrains,
               'slopes': slopes,
               'npss': npss,
               'vp_dists': vp_dists,
               'kr_dists': kr_dists,
               'correlations': corrs}
    np.savez("%i.npz" % seed, config=config, results=results)
    return


if __name__=='__main__':
    print("Setting up ...")
    data = DataManager("meindata")
    num_sims = 100
    num_inputs = 50+randint(150, size=num_sims)  # 50, 200
    input_frequencies = 10+randint(90, size=num_sims)  # 10, 100
    input_weights = np.round(5e-4*random(num_sims)+1e-4, 5)  # 1e-4, 5e-4
    input_synchronies = np.round(random(num_sims), 2)  # 0, 1
    input_jitters = np.round(4e-3*random(num_sims), 4)  # 0, 4e-3
    params = zip(input_synchronies, input_jitters, input_frequencies,
                 num_inputs, input_weights)
    print("Simulations configured. Running ...")
    run_tasks(data, lifsim, params, gui=False, poolsize=0,
              numitems=num_sims)
    print("Simulations done!\nGathering npz files ...")
    zf = zipfile.ZipFile("results.zip", mode="w")
    for filename in os.listdir("."):
        if filename.endswith("npz"):
            print("Adding %s to zipfile ..." % (filename))
            zf.write(filename)
    print("npz files stored in results.zip")
    print("DONE!")




