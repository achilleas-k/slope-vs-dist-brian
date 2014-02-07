import brian
from brian.units import msecond, second, volt, mvolt, hertz
import spikerlib as st
import warnings
from time import time


ms = msecond
brian.defaultclock.dt = dt = 0.1*ms
duration = 2*second
Vth = 15*mvolt
tau = 10*msecond

warnings.simplefilter("always")
slope_w = 2*ms
dcost = float(2/slope_w)

measures = []


def lif_measures(synchrony, jitter, num_inp, freq, weight):
    # initializing (could go in a separate initializer)
    brian.reinit_default_clock(0*second)
    if brian.defaultclock.t > 0:
        warnings.warn("Clock start value > 0 : %f" % (
            brian.defaultclock.t))
    seed = int(time()+(synchrony+jitter+num_inp+freq+weight)*1e3)
    jitter = jitter*second
    freq = freq*hertz
    weight = weight*volt

    # neuron
    neuron = brian.NeuronGroup(1, "dV/dt = -V/tau : volt",
                               threshold="V>Vth", reset="V=0*mvolt",
                               refractory=1*msecond)
    neuron.V = 0*mvolt
    netw = brian.Network(neuron)
    config = {"id": seed,
              "S_in": synchrony,
              "sigma": jitter,
              "f_in": freq,
              "N_in": num_inp,
              "weight": weight}
    # inputs
    sync_inp, rand_inp = st.tools.gen_input_groups(num_inp, freq,
                                                 synchrony, jitter,
                                                 duration, dt)
    netw.add(sync_inp, rand_inp)
    # connections
    if len(sync_inp):
        sync_con = brian.Connection(sync_inp, neuron, "V", weight=weight)
        netw.add(sync_con)
    if len(rand_inp):
        rand_con = brian.Connection(rand_inp, neuron, "V", weight=weight)
        netw.add(rand_con)
    # monitors
    sync_mon = brian.SpikeMonitor(sync_inp)
    rand_mon = brian.SpikeMonitor(rand_inp)
    netw.add(sync_mon, rand_mon)
    v_mon = brian.StateMonitor(neuron, "V", record=True)
    out_mon = brian.SpikeMonitor(neuron)
    netw.add(v_mon, out_mon)
    # network operator for calculating measures every time spike is fired
    @brian.network_operation(when="end")
    def calc_measures():
        if len(out_mon[0]) > 1 and out_mon[0][-1]*second == brian.defaultclock.t:
            isi_edges = (out_mon[0][-2], out_mon[0][-1])
            isi_dura = isi_edges[1]-isi_edges[0]
            isi_dt = (int(isi_edges[0]/dt), int(isi_edges[1]/dt))
            interval_trace = v_mon[0][isi_dt[0]:isi_dt[1]+2]
            slope = (float(Vth)-interval_trace[-int(slope_w/dt)])/float(slope_w)
            # grab interval inputs
            interval_inputs = []
            for inp in sync_mon.spiketimes.itervalues():
                _idx = (isi_edges[0] < inp) & (inp < isi_edges[1])
                interval_inputs.append(inp[_idx]-isi_edges[0])
            for inp in rand_mon.spiketimes.itervalues():
                _idx = (isi_edges[0] < inp) & (inp < isi_edges[1])
                interval_inputs.append(inp[_idx]-isi_edges[0])
            kreuz =  st.metrics.kreuz.pairwise(interval_inputs,
                                              isi_edges[0], isi_edges[1],
                                              2)
            kreuz = kreuz[1][0]
            #if len(kreuz[1]) == 1:
            #    kreuz = kreuz[1][0]
            #else:
            #    warnings.warn("Kreuz metric returned multiple values.")
            #    print(kreuz[1])
            #    kreuz = brian.mean(kreuz[1])
            vp =  st.metrics.victor_purpura.pairwise(interval_inputs,
                                                    dcost)
            mod = st.metrics.modulus_metric.pairwise(interval_inputs,
                                                      isi_edges[0],
                                                      isi_edges[1]).item()
            #corr = st.metrics.corrcoef.corrcoef_spiketrains(interval_inputs,
            #                                                duration=isi_dura)
            measures.append({"slope": slope,
                             #"npss": npss,
                             "kreuz": kreuz,
                             "vp": vp,
                             "mod": mod})

    netw.add(calc_measures)
    # run
    netw.run(duration, report=None)
    brian.plot(v_mon.times, v_mon[0])
    brian.show()
    return {"id": seed, "config": config, "measures": measures}


