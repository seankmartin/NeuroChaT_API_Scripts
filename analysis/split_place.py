from collections import OrderedDict
import os

import numpy as np
import matplotlib.pyplot as plt

from neurochat.nc_data import NData
from neurochat.nc_datacontainer import NDataContainer
from neurochat.nc_plot import loc_firing_and_place
from api_utils import make_dir_if_not_exists, save_mixed_dict_to_csv
import numpy as np
from scipy.spatial.distance import cosine


def undo_rot(vec):
    return np.array([vec[2], vec[0], vec[3], vec[1]])


def undo_shuf(vec):
    return np.array([vec[1], vec[2], vec[0], vec[3]])


if __name__ == "__main__":
    in_dir = r"C:\Users\smartin5\Recordings\ER\Odors\cla-13-04-2018-1"
    tetrode = 10
    container = NDataContainer(load_on_fly=True)
    container.add_axona_files_from_dir(
        in_dir, True, True, tetrode_list=[tetrode])
    container.setup()
    out_dir = os.path.join(in_dir, "nc_results")
    make_dir_if_not_exists(out_dir)
    out_dict = {}
    headers = []
    base_list = ["NW", "NE", "SW", "SE"]
    for ap in ["Spikes", "Rate", "Norm_Spikes", "Norm_Rate"]:
        mod_list = [b + "_" + ap for b in base_list]
        headers = headers + mod_list
    out_dict["File"] = headers
    out_vec = {}
    for i, data in enumerate(container):
        # vectors = np.zeros(3, 4)
        key = data.get_unit_no()
        if key not in out_vec.keys():
            out_vec[key] = np.zeros(shape=(3, 4))
        out_name = os.path.basename(container.get_name_at_idx(i, ext=""))
        print(out_name[:-1])
        names, res, norm_res = data.spatial.corner_place_map(
            data.spike.get_unit_stamp(), chop_bound=0)
        for j, name in enumerate(names):
            s, r = res[:, j]
            sn, rn = norm_res[:, j]
            print("{}: s {:d}, r {:.2f}; sn {:.2f}, rn {:.2f}".format(
                name, int(s), r, sn, rn))
        out_arr = np.concatenate((res.flatten(), norm_res.flatten()))
        if "rot" in out_name:
            out_dict[out_name + "Rotated"] = out_arr
            out_vec[key][1] = res[1]
        elif "shu" in out_name:
            out_dict[out_name + "Shuffled"] = out_arr
            out_vec[key][2] = res[1]
        else:
            out_dict[out_name + "Control"] = out_arr
            out_vec[key][0] = res[1]

    for key, vec in out_vec.items():
        print(key, vec)
        cs_c_r = cosine(vec[0], vec[1])
        cs_c_s = cosine(vec[0], vec[2])
        cs_c_ru = cosine(vec[0], undo_rot(vec[1]))
        cs_c_su = cosine(vec[0], undo_shuf(vec[2]))
        print("Unit {:d}: R RU {:.2f} {:.2f}, S {:.2f} {:.2f}".format(
            key, cs_c_r, cs_c_ru, cs_c_s, cs_c_su))
    save_mixed_dict_to_csv(out_dict, out_dir, str(tetrode) + "_obj.csv")
