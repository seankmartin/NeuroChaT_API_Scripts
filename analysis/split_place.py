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
from scipy.stats import rankdata
import seaborn as sns


def undo_rot(vec):
    """
    This undoes the rotation of:
    0 1 --> 1 3
    2 3 --> 0 2

    """
    return np.array([vec[2], vec[0], vec[3], vec[1]])


def undo_shuf(vec):
    """
    This undoes the shuffle of:
    0 1 --> 2 0
    2 3 --> 1 3

    """
    return np.array([vec[1], vec[2], vec[0], vec[3]])


def test_measure():
    vec1 = [1, 0, 0, 0]
    vec2 = [0, 1, 0, 0]
    vec3 = [0, 1, 0, 0]
    vecs = [vec1, vec2, vec3]
    distance_between(vecs, key="Test1")


def calculate_directional_stats(container, out_vec, out_dict):
    for i, data in enumerate(container):
        key = data.get_unit_no()
        if key not in out_vec.keys():
            out_vec[key] = np.zeros(shape=(3, 4))
        out_name = os.path.basename(container.get_name_at_idx(i, ext=""))
        # print("Working on", out_name[:-1])
        names, res, norm_res = data.spatial.corner_place_map(
            data.spike.get_unit_stamp(), chop_bound=0)
        # for j, name in enumerate(names):
        # s, r = res[:, j]
        # sn, rn = norm_res[:, j]
        # print("{}: s {:d}, r {:.2f}; sn {:.2f}, rn {:.2f}".format(
        #     name, int(s), r, sn, rn))
        out_arr = np.concatenate((res.flatten(), norm_res.flatten()))
        if "rot" in out_name:
            out_dict[out_name + "Rotated"] = out_arr
            out_vec[key][1] = norm_res[1]
        elif "shu" in out_name:
            out_dict[out_name + "Shuffled"] = out_arr
            out_vec[key][2] = norm_res[1]
        elif "same" not in out_name:
            out_dict[out_name + "Control"] = out_arr
            out_vec[key][0] = norm_res[1]
    return out_vec, out_dict


def cosine_sim(a, b):
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)
    return cos


def euc_dist(a, b):
    dist = np.linalg.norm(a - b)
    return dist


def manhat_dist(a, b):
    return np.sum(np.abs(a - b))


def distance_between(vecs, measure=cosine_sim, key="Unit"):
    """Expects 3 vectors and a measure."""
    vecs = [np.array(v) for v in vecs]
    print(key, ":", vecs)
    undo_vecs = [vecs[0], undo_rot(vecs[1]), undo_shuf(vecs[2])]
    print(key, ":", undo_vecs)
    cs_c_r = measure(vecs[0], vecs[1])
    cs_c_s = measure(vecs[0], vecs[2])
    cs_c_ru = measure(vecs[0], undo_vecs[1])
    cs_c_su = measure(vecs[0], undo_vecs[2])
    cs_c_sru = measure(vecs[0], undo_rot(vecs[2]))
    print("Unit {}: R {:.2f}, RU {:.2f}, S {:.2f}, SU {:.2f}".format(
        key, cs_c_r, cs_c_ru, cs_c_s, cs_c_su))
    return (
        [cs_c_r, cs_c_ru, cs_c_s, cs_c_su, cs_c_sru],
        vecs, undo_vecs, undo_rot(vecs[2]))


def to_rank(in_dict):
    for key, vec in in_dict.items():
        for i in range(len(vec)):
            in_dict[key][i] = rankdata(-vec[i], method='ordinal').astype(int)
    return in_dict


def main(in_dir, tetrode):
    container = NDataContainer(load_on_fly=True)
    regex = ".*objects.*"
    container.add_axona_files_from_dir(
        in_dir, True, False, tetrode_list=[tetrode], re_filter=regex)
    container.setup()
    out_dir = os.path.join(in_dir, "nc_results")
    make_dir_if_not_exists(out_dir)
    out_dict = OrderedDict()
    headers = []
    base_list = ["NW", "NE", "SW", "SE"]
    for ap in ["Spikes", "Rate", "Norm_Spikes", "Norm_Rate"]:
        mod_list = [b + "_" + ap for b in base_list]
        headers = headers + mod_list
    out_dict["File"] = headers
    out_vec = OrderedDict()
    out_vec, out_dict = calculate_directional_stats(
        container, out_vec, out_dict)

    out_dict["Summary Stats Rate"] = [
        "Rot_Dist", "Rot_U_Dist", "Shuf_Dist", "Shuf_U_Dist", "Shuf_UR_Dist"]
    out_dict["Summary Stats Rate"] += ["Rate" + b for b in base_list]
    out_dict["Summary Stats Rate"] += ["Rot Rate" + b for b in base_list]
    out_dict["Summary Stats Rate"] += ["Undo Rot Rate" + b for b in base_list]
    out_dict["Summary Stats Rate"] += ["Shuf Rate" + b for b in base_list]
    out_dict["Summary Stats Rate"] += ["Undo Shuf Rate" + b for b in base_list]
    out_dict["Summary Stats Rate"] += ["Undo ShufR Rate" + b for b in base_list]
    for key, vec in out_vec.items():
        res, p_vecs, pu_vecs, ur = distance_between(
            vec, key=key, measure=euc_dist)
        out_dict["Rate Unit " + str(key)] = np.concatenate(
            [res, p_vecs[0], p_vecs[1], pu_vecs[1], p_vecs[2], pu_vecs[2], ur])
        fig, ax = plt.subplots()
        heat_arr = np.zeros(shape=(2, 12))
        heat_arr[:, :2] = p_vecs[0].reshape(2, 2)
        heat_arr[:, 2:4] = p_vecs[1].reshape(2, 2)
        heat_arr[:, 4:6] = pu_vecs[1].reshape(2, 2)
        heat_arr[:, 6:8] = p_vecs[2].reshape(2, 2)
        heat_arr[:, 8:10] = pu_vecs[2].reshape(2, 2)
        heat_arr[:, 10:] = ur.reshape(2, 2)
        sns.heatmap(
            heat_arr, ax=ax, annot=True, square=True, center=0.25,
            cmap="Blues")
        ax.invert_yaxis()
        ax.set_ylim(2, 0)
        ax.set_xlim(0, 12)
        ax.vlines([k for k in range(2, 12, 2)], 2, 0, colors="r")
        fig.savefig(
            os.path.join(out_dir, str(key) + "_heatmap.png"))

    out_dict["Summary Stats Rank"] = [
        "Rot_Dist", "Rot_U_Dist", "Shuf_Dist", "Shuf_U_Dist", "Shuf_UR_Dist"]
    out_dict["Summary Stats Rank"] += ["Rank" + b for b in base_list]
    out_dict["Summary Stats Rank"] += ["Rot Rank" + b for b in base_list]
    out_dict["Summary Stats Rank"] += ["Undo Rot Rank" + b for b in base_list]
    out_dict["Summary Stats Rank"] += ["Shuf Rank" + b for b in base_list]
    out_dict["Summary Stats Rank"] += ["Undo Shuf Rank" + b for b in base_list]
    out_dict["Summary Stats Rank"] += ["Undo ShufR Rank" + b for b in base_list]
    out_vec = to_rank(out_vec)
    for key, vec in out_vec.items():
        res, p_vecs, pu_vecs, ur = distance_between(
            vec, key=key, measure=euc_dist)
        out_dict["Rank Unit " + str(key)] = np.concatenate(
            [res, p_vecs[0], p_vecs[1], pu_vecs[1], p_vecs[2], pu_vecs[2], ur])
    print("Saving results to", os.path.join(
        out_dir, str(tetrode) + "_obj.csv"))
    save_mixed_dict_to_csv(out_dict, out_dir, str(tetrode) + "_obj.csv")


def test_random(num_tests):
    matches = np.zeros(shape=(num_tests, ))
    for i in range(num_tests):
        rands1 = np.random.dirichlet(np.ones(4), size=1)
        rands2 = np.random.dirichlet(np.ones(4), size=1)
        matches[i] = euc_dist(rands1, rands2)
    print("Average match is {}".format(np.average(matches[i])))
    print("5 percentile is {}".format(np.percentile(matches, 5)))
    return matches


if __name__ == "__main__":
    test_random(1000000)
    exit(-1)
    sns.set()
    in_dir = r"C:\Users\smartin5\Recordings\ER\Odors\cla-13-04-2018-1"
    tetrode = 10
    main(in_dir, tetrode)
    in_dir = r"C:\Users\smartin5\Recordings\ER\Odors\cla-08-03-2018"
    tetrode = 3
    main(in_dir, tetrode)
