from brian import *
from neurotools import CV, CV2, IR, LV, SI

duration = 4*second
rate = 30*Hz
N = 50
rate_sigma = 80*ms
rate_loc = 2*second
rate_baseline = 100*Hz
rate_peak = 300*Hz
rate_peak_multiplier = (rate_peak-rate_baseline)/(1.0/(rate_sigma*sqrt(2*pi)))


def poissonrate(t):
    return rate_baseline+rate_peak_multiplier*1.0/(rate_sigma*sqrt(2*pi))*exp(-0.5*((t-rate_loc)/rate_sigma)**2)

poisson_neurons = PoissonGroup(N, rates=poissonrate)
poisson_spikes = SpikeMonitor(poisson_neurons)

print("Generating poisson spike trains ...")
run(duration)

print("Calculating variability measures ...")
window = 300*ms
window_step = 5*ms
window_starts = arange(0*second, duration-window, 5*ms)
sliding_window_bounds = [(start*second, start*second+window)\
        for start in window_starts]
cv_all = []
cv2_all = []
ir_all = []
lv_all = []
si_all = []
spikerates = []
for start, end in sliding_window_bounds:
    window_spikes = [st[(st > start) & (st < end)]\
            for st in poisson_spikes.spiketimes.itervalues()]
    cv_w  =  [CV(st) for st in window_spikes]
    cv2_w  = [CV2(st) for st in window_spikes]
    ir_w  =  [IR(st) for st in window_spikes]
    lv_w  =  [LV(st) for st in window_spikes]
    si_w  =  [SI(st) for st in window_spikes]
    cv_all.append(cv_w)
    cv2_all.append(cv2_w)
    ir_all.append(ir_w)
    lv_all.append(lv_w)
    si_all.append(si_w)
    spikerates.append(mean([len(ws) for ws in window_spikes])/window)

cv_mean = mean(cv_all, axis=1)
cv2_mean = mean(cv2_all, axis=1)
ir_mean = mean(ir_all, axis=1)
lv_mean = mean(lv_all, axis=1)
si_mean = mean(si_all, axis=1)

t = window_starts+window/2
baserates = [poissonrate(ti*second) for ti in t]

subplot(311)
raster_plot(poisson_spikes)
subplot(312)
plot(t, baserates, 'k--', label="Poisson rates")
plot(t, spikerates, label="Actual spike rates")
legend(loc='upper right')
subplot(313)
plot(t, cv_mean, label='CV')
plot(t, cv2_mean, label='CV2')
plot(t, ir_mean, label='IR')
plot(t, lv_mean, label='LV')
plot(t, si_mean, label='SI')
legend(loc='upper right')
show()

print("Done!")


