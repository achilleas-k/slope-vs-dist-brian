from brian import *
import neurotools as nt

def interval_corr(inputspikes, outputspikes, b=0.1*ms, duration=None):
    b = float(b)
    corrs = []
    for prv, nxt in zip(outputspikes[:-1], outputspikes[1:]):
        interval_inputs = []
        for insp in inputspikes:
            interval_spikes = insp[(prv < insp) & (insp < nxt+b)]-prv
            if len(interval_spikes):
                interval_inputs.append(interval_spikes)
        if len(interval_inputs):
            corrs_i = mean(nt.corrcoef_spiketrains(interval_inputs, b, duration))
        else:
            corrs.append(0)
        corrs.append(corrs_i)
    return corrs



