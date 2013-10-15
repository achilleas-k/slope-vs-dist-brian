from brian import *
from brian.tools.datamanager import *
import sys
import neurotools as nt


def load_data(dirname):
    dataman = DataManager(dirname)
    voltage = []
    output_spikes = []
    input_rate = []
    jitter = []
    sync = []
    num_inputs = []
    weight = []
    for data in dataman.itervalues():
        if not len(data):
            continue
        voltage.append(data['mem'])
        output_spikes.append(data['output_spikes'])
        input_rate.append(data['f_in'])
        jitter.append(data['jitter'])
        sync.append(data['sync'])
        num_inputs.append(data['N_in'])
        weight.append(data['w_in'])
    flatdata = {
            'mem': voltage,
            'output_spikes': output_spikes,
            'input_rate': input_rate,
            'jitter': jitter,
            'sync': sync,
            'num_inputs': num_inputs,
            'input_weight': weight,
            }
    return flatdata

def calculate_slopes(data):
    npss = []
    mems = data['mem']
    output_spikes = data['output_spikes']
    for v, spikes in zip(mems, output_spikes):
        npss.append(nt.norm_firing_slope(v, spikes, th=15*mV, tau=10*ms, dt=0.1*ms, w=2*ms)[0])
    return npss


if __name__=='__main__':
    data_dir = sys.argv[1]
    data = load_data(data_dir)
    npss = calculate_slopes(data)




