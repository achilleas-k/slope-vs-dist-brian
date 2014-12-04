"""
Monte-Carlo version of DataGenTaskFarm.py
The simulation parameters are specified by range bounds (min, max) and number
of simulations.
"""
from __future__ import print_function, division
from brian import *
from brian.tools.taskfarm import *
from brian.tools.datamanager import *
import spikerlib as sl
import gc
from time import time
import os
from glob import glob
from datetime import datetime

defaultclock.dt = dt = 0.1*ms
duration = 5*second
slope_w = 2*msecond


def lifsim(sync, sgm, n_in, inrate, weight):
    clear(True)
    gc.collect()
    reinit_default_clock()
    seed = int(time()+(sync+sgm+inrate+n_in+weight)*1e3)
    sgm = sgm*second
    inrate = inrate*Hz
    weight = weight*volt
    Vth = 15*mV
    tau = 10*ms

    # neuron
    neuron = NeuronGroup(1, "dV/dt = -V/tau : volt",
                         threshold="V>Vth", reset="V=0*mvolt", refractory=1*ms)
    neuron.V = 0*mvolt
    netw = Network(neuron)
    config = {'id': seed,
              'S_in': sync,
              'sigma': sgm,
              'f_in': inrate,
              'N_in': n_in,
              'weight': weight}
    # inputs
    inputgroup = sl.tools.fast_synchronous_input_gen(n_in, inrate,
                                                     sync, sgm, duration)
    netw.add(inputgroup)
    # connectio
    inputconn = Connection(inputgroup, neuron, "V", weight=weight)
    netw.add(inputconn)
    # monitors
    inputmon = SpikeMonitor(inputgroup)
    netw.add(inputmon)
    vmon = StateMonitor(neuron, "V", record=True)
    outmon = SpikeMonitor(neuron)
    netw.add(vmon, outmon)
    # run
    netw.run(duration, report=None)
    if outmon.nspikes < 2:
        return
    vmon.insert_spikes(outmon, Vth)
    input_spiketrains = inputmon.spiketimes.values()
    kr_times, kr_dists = sl.metrics.kreuz.multivariate(input_spiketrains,
                                                       0*second, duration,
                                                       int(duration/dt))
    npss = sl.tools.npss(vmon[0], outmon[0], 0*mV, Vth, tau, slope_w)

    results = {'id': seed,
               'mem': vmon[0],
               'outspikes': outmon[0],
               'inspikes': input_spiketrains,
               'npss': npss,
               'kr_times': kr_times,
               'kr_dists': kr_dists}
    np.savez("%i.npz" % seed, config=config, results=results)
    return


if __name__=='__main__':
    print("Setting up ...")
    today = datetime.now()
    today_str = "{}{}{}".format(today.year, today.month, today.day)
    data = DataManager(today_str)
    num_sims = 200
    num_inputs = 50+randint(150, size=num_sims)  # 50, 200
    input_frequencies = 10+np.round(90*random(num_sims), 0)  # 10, 100
    input_weights = np.round(5e-4*random(num_sims)+1e-4, 5)  # 1e-4, 5e-4
    input_synchronies = np.round(random(num_sims), 2)  # 0, 1
    input_jitters = np.round(4e-3*random(num_sims), 4)  # 0, 4e-3
    params = zip(input_synchronies, input_jitters, num_inputs,
                 input_frequencies, input_weights)
    print("Simulations configured. Running ...")
    run_tasks(data, lifsim, params, gui=False, poolsize=0,
              numitems=num_sims)
    print("Simulations done!\nGathering npz files ...")
    npzfiles = glob("*.npz")
    results = []
    configs = []
    for npz in npzfiles:
        print("Adding {}...".format(npz))
        data = np.load(npz)
        conf = data['config'].item()
        resu = data['results'].item()
        newconf = {}
        for k in conf:
            newconf[k] = conf[k]
        newresu = {}
        for k in resu:
            newresu[k] = resu[k]
        configs.append(newconf)
        results.append(newresu)
    os.makedirs("collectedresults")
    np.savez("collectedresults/"+today_str+".npz",
             configs=configs,
             results=results)
    print("npz files stored in collectedrsults/{}.npz".format(today_str))
    print("DONE!")




