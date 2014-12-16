"""
Read in the npz files from multiple oneshot runs and save them as a single npz
that can be loaded to generate figures using makeplots.py
"""
import sys
import numpy as np
from glob import glob

def read_npz(globstr):
    npzfiles = glob(globstr)
    mnpss = []
    krdists = []
    sync = []
    sigma = []
    weights = []
    inrates = []
    numin = []
    for npz in npzfiles:
        data = np.load(npz)
        curnpss = data["mnpss"]
        curlen = len(curnpss)
        mnpss.append(curnpss)
        data = np.load(npz)
        Nin = data["numin"].item()
        numin.extend([Nin]*curlen)
        fin = data["inrate"].item()
        inrates.extend([fin]*curlen)
        weight = data["inweight"].item()
        weights.extend([weight]*curlen)
        sconf = data["syncconf"]
        sync.extend([sc[0] for sc in sconf])
        sigma.extend([sc[1] for sc in sconf])
        mnpss.extend(data["mnpss"])
        krdists.extend(data["krdists"])
    assert len(mnpss) == len(krdists) == len(sync) == len(sigma) ==\
        len(weights) == len(inrates) == len(numin)
    return (np.array(mnpss), np.array(krdists), np.array(sync), np.array(sigma),
            np.array(weights), np.array(inrates), np.array(numin))

if __name__=="__main__":
    globstr = sys.argv[1]
    mnpss, krdists, sync, sigma, weight, inrates, numin = read_npz(globstr)

