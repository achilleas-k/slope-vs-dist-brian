from brian import *
import warnings
from spike_distance_mp import mean_pairwise_distance
from neurotools import (gen_input_groups, pre_spike_slopes,
                        normalised_pre_spike_slopes, corrcoef_spiketrains)
from spike_distance_kreuz import multivariate_spike_distance
import itertools as itt
import uuid


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
duration = 5*second

warnings.simplefilter("always")
slope_w = 2*msecond
dcost = float(2/slope_w)

Vth = 15*mV
tau = 10*ms
num_inputs = [50, 100, 200]
input_frequencies = [10*Hz, 30*Hz, 50*Hz, 100*Hz]
input_weights = [0.1*mV, 0.3*mV, 0.6*mV]
input_synchronies = frange(0, 1, 0.1)
input_jitters = frange(0, 4, 1)*ms
num_simulations = (len(num_inputs)*len(input_frequencies)*len(input_weights)*
                   len(input_synchronies)*len(input_jitters))

# neuron
neuron = NeuronGroup(num_simulations, "dV/dt = -V/tau : volt",
                     threshold="V>Vth", reset="V=0*mvolt", refractory=1*ms)
neuron.V = 0*mvolt
netw = Network(neuron)

syncGens = []
randGens = []
syncCons = []
randCons = []
syncMons = []
randMons = []
configurations = []
uuids = []
for idx, (sync, sgm, inrate, n_in, weight) in enumerate(itt.product(
        input_synchronies, input_jitters,
        input_frequencies, num_inputs, input_weights)):
    print("%i/%i - Setting up simulation with the following parameters:" % (
        idx, num_simulations))
    print("\tS_in: %f" % (sync))
    print("\tsigma: %f" % (sgm))
    print("\tf_in: %f" % (inrate))
    print("\tN_in: %f" % (n_in))
    print("\tweight: %f" % (weight))
    id = uuid.uuid4().hex
    uuids.append(id)  # ordered list of uuids to be used by the results
    configurations.append({'id': id,
                           'S_in': sync,
                           'sigma': sgm,
                           'f_in': inrate,
                           'N_in': n_in,
                           'weight': weight})
    # inputs
    syncInp, randInp = gen_input_groups(n_in, inrate, sync, sgm*second, duration, dt)
    syncGens.append(syncInp)
    randGens.append(randInp)
    netw.add(syncInp, randInp)
    # connections
    if len(syncInp):
        syncCon = Connection(syncInp, neuron[idx], "V", weight=weight)
        syncCons.append(syncCon)
        netw.add(syncCon)
    if len(randInp):
        randCon = Connection(randInp, neuron[idx], "V", weight=weight)
        randCons.append(randCon)
        netw.add(randCon)
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
results = {}
for idx in range(num_simulations):
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

    results[uuids[idx]] = { 'mem': vmon[idx],
                            'outspikes': outmon[idx],
                            'inspikes': input_spiketrains,
                            'slopes': slopes,
                            'npss': npss,
                            'vp_dists': vp_dists,
                            'kr_dists': kr_dists,
                            'correlations': corrs}

print("Saving data before plotting ...")
#np.savez("data_breakdown.npz",
#         slopes=slopes_collection,
#         npss=npss_collection,
#         vp_dists=vp_dist_collection,
#         kr_dists=kr_dist_collection,
#         input_corrs=corr_collection,
#         traces=vmon.values,
#         outspikes=outmon.spiketimes.values(),
#         inspikes=input_spiketrain_collection)

np.savez("svd_sin_sigma_range.npz",
         configurations=configurations,
         results=results)

#print("Calculating averages ...")
#avg_slope = array([mean(slps)
#                   if len(slps) else 0
#                   for slps in slopes_collection])
#avg_npss = array([mean(slps)
#                  if len(slps) else 0
#                  for slps in npss_collection])
#avg_vp = array([mean(vp)
#                if len(vp) else 0
#                for vp in vp_dist_collection])
#avg_kr = array([mean(kr)
#                if len(kr) else 0
#                for kr in kr_dist_collection])
#avg_corr = array([mean(cr)
#                  if len(cr) else 0
#                  for cr in corr_collection])
#
#
#print("Plotting ...")
#figure()
#scatter(avg_slope, avg_vp)
#title("Average slope vs average V-P")
#xlabel("Slope")
#ylabel("V-P")
#savefig("slope_vs_vp.png")
#
#clf()
#figure()
#scatter(avg_npss, avg_vp)
#title("Average normalised slope vs average V-P")
#xlabel("Normalised slope")
#ylabel("V-P")
#savefig("npss_vs_vp.png")
#
#figure()
#scatter(avg_slope, avg_kr)
#title("Average slope vs average Kreuz")
#xlabel("Slope")
#ylabel("Kreuz")
#savefig("slope_vs_kr.png")
#
#clf()
#figure()
#scatter(avg_npss, avg_kr)
#title("Average normalised slope vs average Kreuz")
#xlabel("Normalised slope")
#ylabel("Kreuz")
#savefig("npss_vs_kr.png")
#
#clf()
#figure()
#scatter(avg_slope, avg_corr)
#title("Average slope vs average correlation coefficient")
#xlabel("Slope")
#ylabel("Corr coef")
#savefig("slope_vs_cc.png")
#
#clf()
#figure()
#scatter(avg_npss, avg_corr)
#title("Average normalised slope vs average correlation coefficient")
#xlabel("Noormalised slope")
#ylabel("Corr coef")
#savefig("npss_vs_cc.png")

print("DONE!")




