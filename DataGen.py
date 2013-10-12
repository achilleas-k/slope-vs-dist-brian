from brian import *
from brian.tools.taskfarm import *
from brian.tools.datamanager import *
import sys
import gc
from time import time
import itertools
import random as rnd

duration = 2*second
defaultclock.dt = dt = 0.1*ms
V_th = 15*mV
tau = 10*ms
t_refr = 2*ms
v_reset = 0*mV
V0 = 0*mV


def genInputGroups(N_in, f_in, sync, jitter, duration):
    N_sync = int(N_in*sync)
    N_rand = N_in-N_sync
    syncGroup = PoissonGroup(0, 0)  # dummy nrngrp
    randGroup = PoissonGroup(0, 0)
    if N_sync:
        pulse_intervals = []
        while sum(pulse_intervals)*second < duration:
            interval = rnd.expovariate(f_in)+dt
            pulse_intervals.append(interval)
        pulse_times = cumsum(pulse_intervals[:-1])  # drop last one
        sync_spikes = []
        pp = PulsePacket(0*second, 1, 0*second)  # dummy pp
        for pt in pulse_times:
            try:
                pp.generate(t=pt*second, n=N_sync, jitter=jitter*ms)
                sync_spikes.extend(pp.spiketimes)
            except ValueError:
                continue
        syncGroup = SpikeGeneratorGroup(N=N_sync, spiketimes=sync_spikes)

    if N_rand:
        randGroup = PoissonGroup(N_rand, rates=f_in)

    return syncGroup, randGroup


def lifsim(N_in, f_in, w_in, sync, jitter, report):
    clear(True)
    gc.collect()
    reinit_default_clock()
    seed = int(time()+(N_in+f_in+w_in+sync+jitter)*1e3)
    # add units to parameters
    f_in = f_in*Hz
    w_in = w_in*mV
    jitter = jitter*ms
    # seed the numpy prng
    np.random.seed(seed)
    # set up the network
    eqs = Equations('dV/dt = -(V-V0)/tau : volt')
    eqs.prepare()
    nrngrp = NeuronGroup(1, eqs, threshold='V>V_th',
                                                    refractory=t_refr,
                                                    reset='V=v_reset')
    nrngrp.V = V0
    simnetwork = Network(nrngrp)
    # set up the inputs - the important part
    sg, rg = genInputGroups(N_in, f_in, sync, jitter, duration)
    sm = SpikeMonitor(sg)
    rm = SpikeMonitor(rg)
    simnetwork.add(sm, rm)
    if len(sg):
        sConn = Connection(sg, nrngrp[idx], state='V', weight=w_in)
        simnetwork.add(sg, sConn)
    if len(rg):
        rConn = Connection(rg, nrngrp[idx], state='V', weight=w_in)
        simnetwork.add(rg, rConn)

    mem_mon = StateMonitor(nrngrp, 'V', record=True)
    output_mon = SpikeMonitor(nrngrp)
    simnetwork.add(mem_mon, st_mon)
    # all ready - run the simulation
    simnetwork.run(duration, report=report)
    mem_mon.insert_spikes(st_mon, value=V_th)
    # collect input spikes
    input_spikes = []
    for sm_idx in sm.spiketimes.iterkeys():
        input_spikes.append(sm.spiketimes[sm_idx])
    for rm_idx in rm.spiketimes.iterkeys():
        input_spikes.append(rm.spiketimes[rm_idx])
    output_spikes = []
    for om_idx in output_mon.spiketimes.iterkeys():
        output_spikes.append(output_mon.spiketimes[om_idx])
    return {
            'N_in': N_in,
            'f_in': f_in,
            'w_in': w_in,
            'sync': sync,
            'jitter': jitter,
            'mem': mem_mon.values,
            'output_spikes': output_spikes,
            'duration': defaultclock.t,
            'input_spikes': input_spikes,
            'seed': seed,
            }

if __name__=='__main__':
    data_dir = 'data_for_metric_comparison'
    data = DataManager(data_dir)
    print('\n')
    N_in = frange(50, 200, 50)
    f_in = frange(0.5, 1.5, 0.25)
    w_in = frange(0, 1.5, 0.5)
    sync = frange(0, 1, 0.2)
    jitt = frange(0, 4, 1.0)
    params_prod = itertools.product(N_in, f_in, w_in, sync, jitt)
    nsims = len(N_in)*len(f_in)*len(w_in)*len(sync)*len(jitt)
    print("Simulations configured. Running ...")
    run_tasks(data, lifsim, params_prod, gui=False,
                                    poolsize=0, numitems=nsims)
    print("Simulations done!")

