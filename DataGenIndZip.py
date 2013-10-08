from brian import *
from brian.tools.taskfarm import *
from brian.tools.datamanager import *
import sys
import gc
from time import time
import itertools
import random as rnd
import zipfile

duration = 3*second
defaultclock.dt = dt = 0.1*ms
V_th = 15*mV
tau = 10*ms
t_refr = 2*ms
v_reset = 0*mV
V0 = 0*mV

def genInputGroups(N_in, f_in, S_in, sigma, duration):
    N_sync = int(N_in*S_in)
    N_rand = N_in-N_sync
    syncGroup = PoissonGroup(0, 0) # dummy nrngrp
    randGroup = PoissonGroup(0, 0)
    if N_sync:
        pulse_intervals = []
        while sum(pulse_intervals)*second < duration:
            interval = rnd.expovariate(f_in)+dt
            pulse_intervals.append(interval)
        pulse_times = cumsum(pulse_intervals[:-1]) # ignore last one
        sync_spikes = []
        pp = PulsePacket(0*second, 1, 0*second) # dummy pp
        for pt in pulse_times:
            try:
                pp.generate(t=pt*second, n=N_sync, sigma=sigma*ms)
                sync_spikes.extend(pp.spiketimes)
            except ValueError:
                continue
        syncGroup = SpikeGeneratorGroup(N=N_sync, spiketimes=sync_spikes)

    if N_rand:
        randGroup = PoissonGroup(N_rand, rates=f_in)

    return (syncGroup, randGroup)

def lifsim(N_in, f_out, w_in, report):
    clear(True)
    gc.collect()
    reinit_default_clock()
    seed = int(time()+(N_in+f_out+w_in)*1e3)
    f_out = f_out*Hz
    w_in = w_in*mV
    input_configs = []
    for sigma in frange(0, 4, 0.5):
        for S_in in frange(0, 1, 0.2):
            input_configs.append((S_in, sigma))
    np.random.seed(seed)
    eqs = Equations('dV/dt = -(V-V0)/tau : volt')
    eqs.prepare()
    nrngrp = NeuronGroup(len(input_configs), eqs, threshold='V>V_th',
                                                    refractory=t_refr,
                                                    reset='V=v_reset')
    nrngrp.V = V0
    simnetwork = Network(nrngrp)
    syncGroups = []
    randGroups = []
    syncConns = []
    randConns = []
    syncMons = []
    randMons = []
    idx = 0
    # TODO: estimate input rate
    f_in = f_out
    for S_in, sigma in input_configs:
            sg, rg = genInputGroups(N_in, f_in, S_in, sigma, duration)
            syncGroups.append(sg)
            randGroups.append(rg)
            sm = SpikeMonitor(sg)
            rm = SpikeMonitor(rg)
            syncMons.append(sm)
            randMons.append(rm)
            simnetwork.add(sm, rm)
            if len(sg):
                sConn = Connection(sg, nrngrp[idx], state='V', weight=w_in)
                syncConns.append(sConn)
                simnetwork.add(sg, sConn)
            if len(rg):
                rConn = Connection(rg, nrngrp[idx], state='V', weight=w_in)
                randConns.append(rConn)
                simnetwork.add(rg, rConn)
            idx += 1

    mem_mon = StateMonitor(nrngrp, 'V', record=True)
    st_mon = SpikeMonitor(nrngrp)
    simnetwork.add(mem_mon, st_mon)
    simnetwork.run(duration, report=report)
    mem_mon.insert_spikes(st_mon, value=V_th)
    # collect input spikes
    input_spikes = []
    for sm, rm in zip(syncMons, randMons):
        input_spikes_idx = []
        for sm_idx in sm.spiketimes:
            input_spikes_idx.append(sm.spiketimes[sm_idx])
        for rm_idx in rm.spiketimes:
            input_spikes_idx.append(rm.spiketimes[rm_idx])
        input_spikes.append(input_spikes_idx)
    filename = os.path.join('data', 'data_%i_%i_%f.npz' % (N_in, f_in, w_in))
    np.savez(filename,
            N_in=N_in,
            f_in=f_in,
            f_out=f_out,
            w_in=w_in,
            mem=mem_mon.values,
            spikes=st_mon.spiketimes,
            duration=defaultclock.t,
            seed=seed,
            input_configs=input_configs,
            input_spikes=input_spikes,
            )

    return {
            'N_in': N_in,
            'f_in': f_in,
            'f_out': f_out,
            'w_in': w_in,
            'mem': mem_mon.values,
            'spikes': st_mon.spiketimes,
            'duration': defaultclock.t,
            'seed': seed,
            'input_configs': input_configs,
            'input_spikes': input_spikes,
            }

if __name__=='__main__':
    data_dir = 'data_for_metric_comparison'
    data = DataManager(data_dir)
    os.mkdir('data')
    print('\n')
    N_in = [50, 100, 150]
    f_out = [70, 150, 200, 250, 300, 350]
    w_in = [0.5, 1.0, 1.5]
    params_prod = itertools.product(N_in, f_out, w_in)
    nsims = len(N_in)*len(f_out)*len(w_in)
    print("Simulations configured. Running ...")
    run_tasks(data, lifsim, params_prod, gui=False,
                                    poolsize=0, numitems=nsims)
    print("Simulations done!")
    print("Creating data zip ...")
    zip = zipfile.ZipFile('data.zip', 'w')
    for datafile in os.listdir('data'):
        zip.write(os.path.join('data', datafile))
    zip.close()
    print("DONE!")



