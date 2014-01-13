from brian import *

datadir = sys.args[1]

filesindir = os.listdir(datadir)
npzindir = [fid for fid in filesindir if fid.endswith("npz")]
npzcount = len(npzindir)
results = []
configs = []
for idx, npzfile in enumerate(npzindir):
    npzdata = load(npzfile)
    res = npzdata["results"].item()
    conf = npzdata["config"].item()
    results.append(res)
    configs.append(conf)
    print("Finished reading %s (%i/%i)" % (npzfile, idx, npzcount))


WIP
Pickle results and configs (optional)
Organise data into arrays
Plot!!!
