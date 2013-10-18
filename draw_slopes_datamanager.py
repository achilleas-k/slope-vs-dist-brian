from brian import *
from brian.tools.datamanager import *
import sys
import spike_distance_kreuz as sd
import neurotools as nt


def load_data(dirname):
    dataman = DataManager(dirname)
    voltage = []
    output_spikes = []
    input_rate = []
    input_spikes = []
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
        input_spikes.append(data['input_spikes'])
    flatdata = {
            'mem': voltage,
            'output_spikes': output_spikes,
            'input_rate': input_rate,
            'jitter': jitter,
            'sync': sync,
            'num_inputs': num_inputs,
            'input_weight': weight,
            'input_spikes': input_spikes,
            }
    return flatdata

def calculate_slopes(data):
    npss = []
    mems = data['mem']
    output_spikes = data['output_spikes']
    for v, spikes in zip(mems, output_spikes):
        npss.append(nt.norm_firing_slope(v[0], spikes[0],
            th=15*mV, tau=10*ms, dt=0.1*ms, w=2*ms)[0])
    return npss

def calculate_dist(data):
    '''
    Calculate spike train distance using Kreuz spike distance
    '''
    sync = []
    jitter = []
    dist = []
    itemcount = d.itemcount()
    for idx, d in enumerate(data.itervalues()):
        if not d: continue
        inputs = d['input_spikes']
        sync.append(d['sync'])
        jitter.append(d['jitter'])
        t, sd_i = sd.multivariate_spike_distance(inputs, 0, 1.9, 10)
        dist.append(mean(sd_i))
        print "%i/%i -> S: %f, J: %f, D: %f" % (
                idx, itemcount,
                sync[-1], jitter[-1], dist[-1])
    return dist

def aggregate_slopes(data):
    npss = data['slopes']
    sync = data['sync']
    jitter = data['jitter']
    sync_range = sorted(unique(sync))
    jitter_range = sorted(unique(jitter))
    img = zeros((len(jitter_range), len(sync_range)))
    for si, ji, slp in zip(sync, jitter, npss):
        j = jitter_range.index(float(ji))
        s = sync_range.index(si)
        img[j,s] += slp
    return sync_range, jitter_range, img

if __name__=='__main__':
    data_dir = sys.argv[1]
    data = load_data(data_dir)
    npss = calculate_slopes(data)
    data['slopes'] = npss
    sync, jitter, img = aggregate_slopes(data)
    filename = "aggregated_slopes.npz"
    print("Saving aggregated data to %s" % (filename))
    np.savez(filename, slopes=img)
    # draw it!
    # extent = (min(sync), max(sync), min(jitter), max(jitter))
    # imshow(img, extent=extent, origin='lower', aspect='auto')
    # group simulations by input params & firing rate
    imgdata = []
    for slope, s, j, n, w, out in zip(
            npss, data['sync'], data['jitter'], data['num_inputs'],
            data['input_weight'], data['output_spikes']):
        fout = len(out[0])
        if 90 < fout < 110 and 0.00015*volt < w < 0.00025*volt and n == 100:
            imgdata.append((s, j, slope))




