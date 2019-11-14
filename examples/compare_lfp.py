import os
import numpy as np

from neurochat.nc_data import NData


def load_lfp(load_loc, i, data):
    end = str(i + 1)
    if end == "1":
        load_loc = load_loc
    else:
        load_loc = load_loc + end
    data.lfp.load(load_loc)


def get_normalised_diff(s1, s2):
    # MSE of one divided by MSE of main
    return np.sum(np.square(s1 - s2)) / np.sum(np.square(s1))


def compare_lfp(load_loc, out_loc):
    ndata1 = NData()
    ndata2 = NData()
    grid = np.meshgrid(np.arange(32), np.arange(32), indexing='ij')
    stacked = np.stack(grid, 2)
    pairs = stacked.reshape(-1, 2)
    result_a = np.zeros(shape=pairs.shape[0], dtype=np.float32)

    for i, pair in enumerate(pairs):
        load_lfp(load_loc, pair[0], ndata1)
        load_lfp(load_loc, pair[1], ndata2)
        res = get_normalised_diff(
            ndata1.lfp.get_samples(), ndata2.lfp.get_samples())
        result_a[i] = res

    with open(out_loc, "w") as f:
        headers = [str(i) for i in range(1, 33)]
        out_str = ",".join(headers)
        f.write(out_str)
        out_str = ""
        for i, (pair, val) in enumerate(zip(pairs, result_a)):
            if i % 32 == 0:
                f.write(out_str + "\n")
                out_str = ""

            out_str += "{:.2f},".format(val)
            # f.write("({}, {}): {:.2f}\n".format(pair[0], pair[1], val))
        f.write(out_str + "\n")

    return result_a


if __name__ == "__main__":
    lfp_base_dir = r"C:\Users\smartin5\Recordings\ER\LFP-cla-V2L\05112019-orange"
    lfp_base_name = "05112019-orange-L1.eeg"
    lfp_name = os.path.join(lfp_base_dir, lfp_base_name)
    out_name = lfp_base_name[:-3] + "csv"
    out_base_dir = (
        r"C:\Users\smartin5\Google Drive\NeuroScience\Results\ER08112019")
    out_loc = os.path.join(out_base_dir, out_name)
    results = compare_lfp(lfp_name, out_loc)
