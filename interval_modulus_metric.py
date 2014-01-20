import numpy as np
import modulus_metric as modm

def interval_modm(inspikes, outspikes, start, end):
    distances = []
    for prv, nxt in zip(outspikes[:-1], outspikes[1:]):
        interval_inputs = []
        for insp in inspikes:
            interval_spikes = insp[(prv < insp) & (insp <= nxt)]-prv
            if len(interval_spikes):
                interval_inputs.append(interval_spikes)
        dist_i = np.mean(modm.avg_pairwise_modulus(interval_inputs, 0, nxt-prv))
        distances.append(dist_i)
    return distances

