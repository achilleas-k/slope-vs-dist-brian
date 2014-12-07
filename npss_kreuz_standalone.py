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
import itertools as it

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

def get_random_params(nsims):
    num_sims = nsims
    num_inputs = 50+randint(150, size=num_sims)  # 50, 200
    input_frequencies = 10+np.round(190*random(num_sims), 0)  # 10, 200
    input_weights = np.round(5e-4*random(num_sims)+1e-4, 5)  # 1e-4, 5e-4
    input_synchronies = np.round(random(num_sims), 2)  # 0, 1
    input_jitters = np.round(4e-3*random(num_sims), 4)  # 0, 4e-3
    return zip(input_synchronies, input_jitters, num_inputs,
               input_frequencies, input_weights)

def get_fullrange_params():
    num_inputs = np.arange(50, 210, 10)
    input_frequencies = np.arange(10.0, 110.0, 10.0)
    input_weights = np.arange(1e-4, 6e-4, 1e-4)
    input_synchronies = np.arange(0, 1.1, 0.1)
    input_jitters = np.arange(0, 5e-3, 1e-3)
    return [p for p in it.product(input_synchronies, input_jitters, num_inputs,
                                  input_frequencies, input_weights)]

if __name__=='__main__':
    print("Setting up ...")
    today = datetime.now()
    today_str = "{}{:02}{:02}".format(today.year, today.month, today.day)
    data = DataManager(today_str)
    #params = get_fullrange_params()
    params = get_random_params(1000)
    num_sims = len(params)
    print("Simulations configured. Running ...")
    run_tasks(data, lifsim, params, gui=False, poolsize=0, numitems=num_sims)
    print("Simulations done!\nGathering npz files ...")
    npzfiles = glob("*.npz")
    try:
        os.makedirs("collectedresults")
    except OSError:
        pass
    results = []
    configs = []
    outcount = 0
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
        if len(configs) >= 500:
            outcount+=1
            filename = "collectedresults/"+today_str+"-"+str(outcount)+".npz"
            np.savez(filename,
                     configs=configs,
                     results=results)
            print("Saved "+filename)
            configs = []
            results = []
    outcount+=1
    filename = "collectedresults/"+today_str+"-"+str(outcount)+".npz"
    np.savez(filename,
             configs=configs,
             results=results)

    print("npz files stored in collectedrsults/{}.npz".format(today_str))
    print("Removing leftover npz files...")
    for npz in npzfiles:
        try:
            os.remove(npz)
        except IOError:
            pass
    print("DONE!")
