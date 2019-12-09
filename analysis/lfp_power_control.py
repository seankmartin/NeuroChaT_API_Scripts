import os
import argparse
from collections import OrderedDict
import numpy as np

from lfp_power import plot_sample_of_signal
from lfp_power import lfp_entropy
from lfp_power import lfp_distribution
from lfp_power import lfp_theta_dist
from lfp_power import raw_lfp_power
from lfp_power import single_main
from api_utils import save_mixed_dict_to_csv


def plot_sample():
    """Plot a small sample of the signal."""
    root = r"C:\Users\smartin5\Recordings\ER"
    name = "29082019-bt2\\29082019-bt2-2nd-LFP.eeg"
    load_loc = os.path.join(root, name)
    plot_sample_of_signal(
        load_loc, out_dir="nc_results", name="Sal", offseta=400)
    name = "30082019-bt2\\30082019-bt2-2nd-LFP.eeg"
    load_loc = os.path.join(root, name)
    plot_sample_of_signal(
        load_loc, out_dir="nc_results", name="Ser", offseta=400)


def single_main_cfg():
    """Main control through cmd options."""
    parser = argparse.ArgumentParser(description="Parse a program location")
    parser.add_argument(
        "--nofilt", "-nf", action="store_true",
        help="Should not pre filter lfp before power and spectral analysis")
    parser.add_argument(
        "--max_freq", "-mf", type=int, default=40,
        help="The maximum lfp frequency to consider"
    )
    parser.add_argument(
        "--loc", type=str, help="Lfp file location"
    )
    parser.add_argument(
        "--eeg_num", "-en", type=str, help="EEG number", default="1"
    )
    parser.add_argument(
        "--splits", "-s", nargs="*", type=int, help="Splits",
        default=[0, 600, 600, 1200, 1200, 1800]
    )
    parser.add_argument(
        "--out_loc", "-o", type=str, default="nc_results",
        help="Relative name of directory to store results in"
    )
    parser.add_argument(
        "--every_min", "-em", action="store_true",
        help="Calculate lfp every minute"
    )
    parser.add_argument(
        "--recording_dur", "-d", type=int, default=30,
        help="How long in minutes the recording lasted"
    )
    parser.add_argument(
        "--get_entropy", "-e", action="store_true",
        help="Calculate entropy"
    )
    parser.add_argument(
        "--g_all", "-a", action="store_true",
        help="Get all values instead of just average"
    )
    parsed = parser.parse_args()

    single_main(parsed)


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
            max_freq=40,
            nofilt=False,
            loc=os.path.join(root, name[0]),
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
    save_mixed_dict_to_csv(_results, "nc_results", "power_results.csv")

    return


if __name__ == "__main__":
    main()
