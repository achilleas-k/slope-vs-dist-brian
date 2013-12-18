from brian import *
import warnings
from spike_distance_mp import mean_pairwise_distance
from neurotools import (gen_input_groups, pre_spike_slopes,
                        normalised_pre_spike_slopes, corrcoef_spiketrains)
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
            interval_spikes = insp[(prv < insp) & (insp < nxt+dt)]-prv
            if len(interval_spikes):
                interval_inputs.append(interval_spikes)
        if len(interval_inputs):
            corrs_i = mean(corrcoef_spiketrains(interval_inputs, dt, duration))
        else:
            corrs.append(0)
        corrs.append(corrs_i)
    return corrs

defaultclock.dt = dt = 0.1*ms
duration = 1*second

warnings.simplefilter("always")
slope_w = 2*msecond
dcost = float(2/slope_w)

Vth = 15*mV
tau = 10*ms
N_in = 100
f_in = 30*Hz
S_in = frange(0, 1, 0.1)
sigma_in = 1*ms
Nsims = len(S_in)

# neuron
neuron = NeuronGroup(Nsims, "dV/dt = -V/tau : volt",
                     threshold="V>Vth", reset="V=0*mvolt", refractory=1*ms)
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

slopes_collection = []
npss_collection = []
vp_dist_collection = []
kr_dist_collection = []
corr_collection = []
for idx in range(Nsims):
    if len(outmon[idx]) <= 1:
        print("Simulation %i fired %i spikes. Discarding ..." % (
            idx, len(outmon[idx])))
        continue  # drop it
    print("%i: Calculating slopes ..." % idx)
    slopes = pre_spike_slopes(vmon[idx], outmon[idx], Vth, slope_w, dt)
    slopes_collection.append(slopes)

    print("%i: Calculating normalised slopes ..." % idx)
    npss = normalised_pre_spike_slopes(vmon[idx], outmon[idx], 0*mV, Vth, tau,
                                       slope_w, dt)
    npss_collection.append(npss)

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
         npss=npss_collection,
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
avg_npss = array([mean(slps)
                  if len(slps) else 0
                  for slps in npss_collection])
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
title("Average slope vs average V-P")
xlabel("Slope")
ylabel("V-P")
savefig("slope_vs_vp.png")

clf()
figure()
scatter(avg_npss, avg_vp)
title("Average normalised slope vs average V-P")
xlabel("Normalised slope")
ylabel("V-P")
savefig("npss_vs_vp.png")

figure()
scatter(avg_slope, avg_kr)
title("Average slope vs average Kreuz")
xlabel("Slope")
ylabel("Kreuz")
savefig("slope_vs_kr.png")

clf()
figure()
scatter(avg_npss, avg_kr)
title("Average normalised slope vs average Kreuz")
xlabel("Normalised slope")
ylabel("Kreuz")
savefig("npss_vs_kr.png")

clf()
figure()
scatter(avg_slope, avg_corr)
title("Average slope vs average correlation coefficient")
xlabel("Slope")
ylabel("Corr coef")
savefig("slope_vs_cc.png")

clf()
figure()
scatter(avg_npss, avg_corr)
title("Average normalised slope vs average correlation coefficient")
xlabel("Noormalised slope")
ylabel("Corr coef")
savefig("npss_vs_cc.png")

print("DONE!")




