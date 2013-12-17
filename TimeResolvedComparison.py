from brian import *
import warnings
from spike_distance_mp import mean_pairwise_distance
from neurotools import (gen_input_groups, pre_spike_slope,
                        normalised_pre_spike_slope, corrcoef_spiketrains)
from spike_distance_kreuz import multivariate_spike_distance


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


def interval_corr(inputspikes, outputspikes, dt=0.1*ms, duration=None):
    dt = float(dt)
    corrs = []
    for prv, nxt in zip(outputspikes[:-1], outputspikes[1:]):
        interval_inputs = []
        for insp in inputspikes:
            interval_inputs.append(insp[(prv < insp) & (insp < nxt+dt)])
            # TODO: spike times should be shifted to time = 0
        corrs_i = mean(corrcoef_spiketrains(interval_inputs, dt, duration))
        corrs.append(corrs_i)
    return corrs

defaultclock.dt = dt = 0.1*ms
duration = 1*second

warnings.simplefilter("always")
slope_w = 2*msecond
dcost = float(2/slope_w)

Vth = 15*mV
N_in = 100
f_in = 30*Hz
S_in = frange(0, 1, 0.1)
sigma_in = 1*ms
Nsims = len(S_in)

# neuron
neuron = NeuronGroup(Nsims, "dV/dt = -V/(10*ms) : volt",
        threshold="V>Vth", reset="V=0*mvolt")
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
    syncInp, randInp = gen_input_groups(N_in, f_in, sync, sigma_in, duration, dt)
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

vmon.insert_spikes(outmon, Vth)

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
winstart_collection = []
vp_dist_collection = []
kr_dist_collection = []
corr_collection = []
for idx in range(Nsims):
    # calculate npss
    print("%i: Calculating slope measure ..." % idx)
    mslope, slopes = norm_firing_slope(vmon[idx], outmon[idx],
            Vth, 2*ms, dt)
    _, oldslopes = npss(vmon[idx], outmon[idx], Vth, 2*ms, dt)
    mslope_collection.append(mslope)
    slopes_collection.append(slopes)
    oldslopes_collection.append(oldslopes)

    # calculate window start
    winstart = get_win_start(vmon[idx], outmon[idx])
    winstart_collection.append(winstart)

    input_spiketrains = input_spiketrain_collection[idx]
    # calculate mean pairwise V-P distance for each interval
    print("%i: Calculating VP distance ..." % idx)
    vp_dists = interval_VP(input_spiketrains, outmon[idx], dcost)
    vp_dist_collection.append(vp_dists)

    # calculate multivariate Kreuz spike distance
    print("%i: Calculating Kreuz distance ..." % idx)
    kr_dists = interval_Kr(input_spiketrains, outmon[idx])
    kr_dist_collection.append(kr_dists)

    # calculate mean correlation coefficient
    print("%i: Calculating binned correlation coefficient ..." % idx)
    corrs = interval_corr(input_spiketrains, outmon[idx], dt, duration)
    corr_collection.append(corrs)

print("Saving data before plotting ...")
np.savez("svd_fullsync_nojitt.npz",
        slopes=slopes_collection,
        oldslopes=oldslopes_collection,
        winstart=winstart_collection,
        vp_dists=vp_dist_collection,
        kr_dists=kr_dist_collection,
        input_corrs=corr_collection,
        traces=vmon.values,
        outspikes=outmon.spiketimes.values(),
        inspikes=input_spiketrain_collection)

print("Calculating averages ...")
# slope averages is already in ``mslope_collection`` but let's calc it anyway
avg_slope = array([mean(slps)
                    if len(slps) else 0
                    for slps in slopes_collection])
avg_oldslope = array([mean(oldslp)
                    if len(oldslp) else 0
                    for oldslp in oldslopes_collection])
avg_winstart = array([mean(winstart)
                    if len(winstart) else 0
                    for winstart in winstart_collection])
avg_vp = array([mean(vp)
                if len(vp) else 0
                for vp in vp_dist_collection])
avg_kr = array([mean(kr)
                if len(kr) else 0
                for kr in kr_dist_collection])
avg_corr = array([mean(cr)
                if len(cr) else 0
                for cr in corr_collection])


print("Plotting ...")
figure()
scatter(avg_slope, avg_vp)
title("Average normalised slope vs average V-P")
xlabel("Slope")
ylabel("V-P")
savefig("slope_vs_vp.png")

clf()
figure()
scatter(avg_oldslope, avg_vp)
title("Average normalised slope (old calc) vs average V-P")
xlabel("Slope")
ylabel("V-P")
savefig("oldslope_vs_vp.png")

clf()
figure()
scatter(avg_winstart, avg_vp)
title("Average V(t-w) vs average V-P")
xlabel("V(t-w)")
ylabel("V-P")
savefig("ws_vs_vp.png")

figure()
scatter(avg_slope, avg_kr)
title("Average normalised slope vs average Kreuz")
xlabel("Slope")
ylabel("Kreuz")
savefig("slope_vs_kr.png")

clf()
figure()
scatter(avg_oldslope, avg_kr)
title("Average normalised slope (old calc) vs average Kreuz")
xlabel("Slope")
ylabel("Kreuz")
savefig("oldslope_vs_kr.png")

clf()
figure()
scatter(avg_winstart, avg_kr)
title("Average V(t-w) vs average Kreuz")
xlabel("V(t-w)")
ylabel("Kreuz")
savefig("ws_vs_kr.png")

clf()
figure()
scatter(avg_winstart, avg_corr)
title("Average V(t-w) vs average correlation coefficient")
xlabel("V(t-w)")
ylabel("Corr coef")
savefig("ws_vs_cc.png")

print("DONE!")




