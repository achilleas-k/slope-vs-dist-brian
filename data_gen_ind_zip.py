from brian import *
from brian.tools.taskfarm import *
from brian.tools.datamanager import *
import os
import gc
from time import time
import itertools
import random as rnd
import zipfile
import neurotools as nt

duration = 3*second
defaultclock.dt = dt = 0.1*ms
V_th = 15*mV
tau = 10*ms
t_refr = 2*ms
v_reset = 0*mV
V0 = 0*mV


def lifsim(N_in, f_out, w_in, report):
    clear(True)
    gc.collect()
    reinit_default_clock()
    seed = int(time()+(N_in+f_out+w_in)*1e3)
    f_out = f_out*Hz
    w_in = w_in*mV
    input_configs = []
    for sigma in frange(0, 4, 2.0):
        for S_in in frange(0, 1, 0.2):
            input_configs.append((S_in, sigma))
    np.random.seed(seed)
    eqs = Equations('dV/dt = -(V-V0)/tau : volt')
    eqs.prepare()
    neurondef = {
            'eqs': eqs,
            'V_th': V_th,
            'refr': t_refr,
            'reset': v_reset
            }
    f_in = nt.calibrate_frequencies(neurondef, N_in, w_in, input_configs, f_out)
    print("Input calibration complete. Starting simulation ...")
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
    for idx, (S_in, sigma) in enumerate(input_configs):
            sg, rg = nt.gen_input_groups(N_in, f_in[idx], S_in, sigma, duration)
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
    #os.mkdir('data')
    print('\n')
    N_in = [100]
    f_out = [50]
    w_in = [0.5]
    params_prod = itertools.product(N_in, f_out, w_in)
    nsims = len(N_in)*len(f_out)*len(w_in)
    print("Simulations configured. Running ...")
    #run_tasks(data, lifsim, params_prod, gui=False,
    #                                poolsize=1, numitems=nsims)
    lifsim(100, 100, 0.2, None)
    print("Simulations done!")
    print("Creating data zip ...")
    zip = zipfile.ZipFile('data.zip', 'w')
    for datafile in os.listdir('data'):
        zip.write(os.path.join('data', datafile))
    zip.close()
    print("DONE!")



