from __future__ import print_function
import sys
import os
import numpy as np

directory = sys.argv[1]

npss_dict = np.load(os.path.join(directory, "npss.npz"))["data"]
modulus_dict = np.load(os.path.join(directory, "modulus.npz"))["data"]
vp_dict = np.load(os.path.join(directory, "victor_purpura.npz"))["data"]
kreuz_dict = np.load(os.path.join(directory, "kreuz.npz"))["data"]

measures = {}
ids = [_n["id"] for _n in npss_dict]
for idx, _id in enumerate(ids):
    # find matching npss
    npss = [_n["npss"] for _n in npss_dict if _n["id"] == _id]
    # find matching vp
    vp = [_v["vp"] for _v in vp_dict if _v["id"] == _id]
    #find matching kreuz
    kr = [_k["kr"] for _k in kreuz_dict if _k["id"] == _id]
    # find matching modulus
    mod = [_m["modulus"] for _m in modulus_dict if _m["id"] == _id]
    metrics_dict = {"npss": npss, "vp": vp, "modulus": mod, "kreuz": kr}
    measures[str(_id)] = metrics_dict
    print("%i/%i" % (idx, len(ids)), end="\r")

output_filename = os.path.join("measures.npz")
np.savez_compressed(output_filename, **measures)


