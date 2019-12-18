import os
import argparse
from collections import OrderedDict
import numpy as np

from lfp_plot import plot_sample_of_signal
from lfp_signal_measures import lfp_entropy
from lfp_signal_measures import lfp_distribution_measures
from lfp_signal_measures import lfp_theta_dist
from lfp_signal_measures import raw_lfp_power
from lfp_signal_measures import single_main
from api_utils import save_mixed_dict_to_csv


def plot_sample():
    """Plot a small sample of the signal."""
    root = r"C:\Users\smartin5\Recordings\ER"
    name = "29082019-bt2\\29082019-bt2-2nd-LFP.eeg"
    load_loc = os.path.join(root, name)
    out_dir = os.path.join(root, "nc_results")
    print("Saving signal sample to {}".format(os.path.join(
        out_dir, "Sal.pdf")))
    plot_sample_of_signal(
        load_loc, out_dir=out_dir, name="Sal.pdf", offseta=400,
        filt_params=(True, 1.5, 40), length=40)
    name = "30082019-bt2\\30082019-bt2-2nd-LFP.eeg"
    load_loc = os.path.join(root, name)
    print("Saving signal sample to {}".format(os.path.join(
        out_dir, "Ser.pdf")))
    plot_sample_of_signal(
        load_loc, out_dir=out_dir, name="Ser.pdf", offseta=400,
        filt_params=(True, 1.5, 40), length=50)


def main():
    """
    Main control, used to run single_main for many files and compile output.

    Proceeds as follows:
    1. Set up a list of filenames and params to use.
    2. For each filename run single_main, 
        And also save the raw and band power values to an array
    3. 
    """
    root = "C:\\Users\\smartin5\\Recordings\\ER\\"
    from types import SimpleNamespace
    ent = False
    arr_raw_pow = []
    arr_band_pow = []

    # Set up the locations and trial type
    names_list = [
        ("29072019-bt\\29072019-bt-1st30min-LFP.eeg", "firstt", "bt1_1_sal"),
        ("30072019-bt\\30072019-bt-1st30min-LFP-DSER.eeg", "firstt", "bt1_1_ser"),
        ("29072019-bt\\29072019-bt-last30min-LFP.eeg", "lastt", "bt1_2_sal"),
        ("30072019-bt\\30072019-bt-last30min-LFP-DSER.eeg", "lastt", "bt1_2_ser"),
        ("29082019-nt2\\29082019-nt2-LFP-1st-Saline.eeg", "firstt", "nt2_1_sal"),
        ("30082019-nt2\\30082019-nt2-LFP-1st.eeg", "firstt", "nt2_1_ser"),
        ("29082019-nt2\\29082019-nt2-LFP-2nd-Saline.eeg", "lastt", "nt2_2_sal"),
        ("30082019-nt2\\30082019-nt2-LFP-2nd.eeg", "lastt", "nt2_2_ser"),
        ("29082019-bt2\\29082019-bt2-1st-LFP1_MERGE_29082019-bt2-1st-LFP2.eeg",
         "firstt", "bt2_1_sal"),
        ("30082019-bt2\\30082019-bt2-1st-LFP.eeg", "firstt", "bt2_1_ser"),
        ("29082019-bt2\\29082019-bt2-2nd-LFP.eeg", "lastt", "bt2_2_sal"),
        ("30082019-bt2\\30082019-bt2-2nd-LFP.eeg", "lastt", "bt2_2_ser")
    ]

    # Set up the namespace with params for each name and run the file
    for name in names_list:
        args = SimpleNamespace(
            max_freq=90,
            nofilt=False,
            loc=os.path.join(root, name[0][:-4]),
            eeg_num="13",
            splits=[],
            out_loc=name[1],
            every_min=False,
            recording_dur=1800,
            get_entropy=ent,
            g_all=True
        )
        _, all1, band1 = single_main(args)
        arr_raw_pow.append(all1)
        arr_band_pow.append(band1)

    # Get the differences in power over each pair of D-SER, Control recordings.
    for i in range(0, len(arr_raw_pow), 2):
        difference = arr_raw_pow[i + 1] - arr_raw_pow[i]
        print("Mean difference is {:4f}".format(np.mean(difference)))
        print("Std deviation is {:4f}".format(np.std(difference)))

    # Ccompile the results and output to csv
    _results = OrderedDict()
    _results["channels"] = [i + 1 for i in range(32)]

    # Raw power calculation
    for (name, arr) in zip(names_list, arr_raw_pow):
        key_name = name[2]
        _results[key_name] = arr

    # Band power calculation
    band_names = ["theta", "delta", "ratio", "theta rel", "delta rel", "total"]
    for i, bname in enumerate(band_names):
        for (name, arr) in zip(names_list, arr_band_pow):
            key_name = bname + " " + name[2]
            _results[key_name] = arr[i]

    # Some t_tests
    from scipy import stats
    _all = arr_raw_pow
    # Extract the D-Serine last 30min recordings, Saline last 30min recordings
    saline_list = [_all[i].flatten() for i in range(2, 12, 4)]
    serine_list = [_all[i].flatten() for i in range(3, 12, 4)]
    # Take one channel from each tetrode for t-test
    # TODO revise this considering one broken channel
    final1 = np.concatenate(serine_list)[3::4]
    final2 = np.concatenate(saline_list)[3::4]
    t_res = stats.ttest_rel(final1, final2)

    # Output the results to CSV
    headers = [
        "Mean in Saline", "Mean in Serine",
        "Std Error in Saline", "Std Error in Serine",
        "T-test stat", "P-Value"]
    _results["Summary Stats"] = headers

    out_vals = [
        final2.mean(), final1.mean(),
        stats.sem(final2, ddof=1), stats.sem(final1, ddof=1),
        t_res[0], t_res[1]]
    _results["Stats Vals"] = out_vals
    save_mixed_dict_to_csv(
        _results, os.path.join(root, "nc_results"), "power_results_90.csv")


if __name__ == "__main__":
    # main()
    plot_sample()
