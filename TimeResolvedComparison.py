from brian import *
import warnings
from spike_distance_mp import mean_pairwise_distance
from neurotools import norm_firing_slope, genInputGroups, npss
from spike_distance_kreuz import multivariate_spike_distance

def get_win_start(trace, spikes, w=2*ms, dt=0.1*ms):
    if len(spikes) == 0:
        return array([])
    spikes = array(spikes)
    spikes_dt = spikes/(0.1*ms)
    spikes_dt = spikes_dt.astype(int)
    w_dt = int(w/dt)  # 2ms/dt
    win_start_dt = spikes_dt-w_dt
    return trace[win_start_dt]


def interval_VP(inputspikes, outputspikes, cost, dt=0.1*ms):
    dt = float(dt)
    vpdists = []
    for prv, nxt in zip(outputspikes[:-1], outputspikes[1:]):
        interval_inputs = []
        for insp in inputspikes:
            interval_inputs.append(insp[(prv < insp) & (insp < nxt+dt)])
        vpd = mean_pairwise_distance(interval_inputs, cost)
        vpdists.append(vpd)
    return vpdists


def interval_Kr(inputspikes, outputspikes, dt=0.1*ms):
    dt = float(dt)
    krdists = []
    for prv, nxt in zip(outputspikes[:-1], outputspikes[1:]):
        krd = multivariate_spike_distance(inputspikes, prv, nxt+dt, 1)
        krdists.append(krd[1])
    return krdists


defaultclock.dt = dt = 0.1*ms
duration = 5*second

warnings.simplefilter("always")
slope_w = 2*msecond
dcost = float(2/slope_w)

N_in = 100
f_in = 30*Hz
S_in = frange(0, 1, 0.1)
Nsims = len(S_in)

# neuron
neuron = NeuronGroup(Nsims, "dV/dt = -V/(10*ms) : volt",
        threshold="V>15*mvolt", reset="V=0*mvolt")
neuron.V = 0*mvolt
netw = Network(neuron)

syncGens = []
randGens = []
syncCons = []
randCons = []
syncMons = []
randMons = []
for idx, sync in enumerate(S_in):
    # inputs
    syncInp, randInp = genInputGroups(N_in, f_in, sync, 0*ms, duration, dt)
    syncGens.append(syncInp)
    randGens.append(randInp)
    netw.add(syncInp, randInp)
    # connections
    syncCon = Connection(syncInp, neuron[idx], "V", weight=0.3*mV)
    randCon = Connection(randInp, neuron[idx], "V", weight=0.3*mV)
    syncCons.append(syncCon)
    randCons.append(randCon)
    netw.add(syncCon, randCon)
    # monitors
    syncmon = SpikeMonitor(syncInp)
    randmon = SpikeMonitor(randInp)
    syncMons.append(syncmon)
    randMons.append(randmon)
    netw.add(syncmon, randmon)

# global monitors
vmon = StateMonitor(neuron, "V", record=True)
outmon = SpikeMonitor(neuron)
netw.add(vmon, outmon)
# run
print("Running ...")
netw.run(duration, report="stdout")

if outmon.nspikes == 0:
    print("No spikes fired. Aborting!")
    sys.exit(0)

# TODO: correlation coefficient on binned spike trains.

# collect input spikes into array of arrays
print("Collecting input spikes ...")
input_spiketrain_collection = []
for syncmon, randmon in zip(syncMons, randMons):
    input_spiketrains = syncmon.spiketimes.values()
    input_spiketrains += randmon.spiketimes.values()
    input_spiketrains = array(input_spiketrains)
    input_spiketrain_collection.append(input_spiketrains)

mslope_collection = []
slopes_collection = []
oldslopes_collection = []
vp_dist_collection = []
kr_dist_collection = []
for idx in range(Nsims):
    # calculate npss
    print("%i: Calculating slope measure ..." % idx)
    mslope, slopes = norm_firing_slope(vmon[idx], outmon[idx],
            15*mV, 2*ms, dt)
    _, oldslopes = npss(vmon[idx], outmon[idx], 15*mV, 2*ms, dt)
    mslope_collection.append(mslope)
    slopes_collection.append(slopes)
    oldslopes_collection.append(oldslopes)

    input_spiketrains = input_spiketrain_collection[idx]
    # calculate mean pairwise V-P distance for each interval
    print("%i: Calculating VP distance ..." % idx)
    vp_dists = interval_VP(input_spiketrains, outmon[idx], dcost)
    vp_dist_collection.append(vp_dists)

    # calculate multivariate Kreuz spike distance
    print("%i: Calculating Kreuz distance ..." % idx)
    kr_dists = interval_Kr(input_spiketrains, outmon[idx])
    kr_dist_collection.append(kr_dists)

print("Plotting ...")
subplot2grid((5,1), (0,0), rowspan=1, colspan=1)
plot(outmon[0][1:], slopes)
axis(xmin=0, xmax=float(duration))

subplot2grid((5,1), (1,0), rowspan=1, colspan=1)
plot(outmon[0][1:], oldslopes)
axis(xmin=0, xmax=float(duration))

subplot2grid((5,1), (2,0), rowspan=1, colspan=1)
plot(outmon[0][1:], vp_dists)
axis(xmin=0, xmax=float(duration))

subplot2grid((5,1), (3,0), rowspan=1, colspan=1)
plot(outmon[0][1:], kr_dists)
axis(xmin=0, xmax=float(duration))

subplot2grid((5,1), (4,0), rowspan=1, colspan=1)
raster_plot(syncmon, randmon)
xmin, xmax, ymin, ymax = axis()
for ospk in outmon[0]:
    plot([ospk*1000, ospk*1000], [ymin, ymax], 'r-')
axis(xmin=0, xmax=float(duration)*1000)

ion()
show()


