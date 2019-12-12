import sys
import os
import argparse
import math
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
try:
    import entropy
except Exception as e:
    print("Could not import entropy module with error {}".format(e))

from lfp_plot import plot_long_lfp
from lfp_odict import LfpODict
from api_utils import save_mixed_dict_to_csv
from api_utils import make_dir_if_not_exists

import neurochat.nc_plot as nc_plot


def raw_lfp_power(lfp, splits=None):
    """
    Calculate SUM(lfp_val^2) / num_lfp_samples in each split.

    Args:
        lfp (NLfp): The lfp to get the power of.
        splits (iterable of tuples): 
            The lower and upper bound in seconds for each split.

    Returns:
        OrderedDict: The raw power for each split. OR
        The raw power if splits is None
    """

    if splits is None:
        samples = lfp.get_samples()
        return np.sum(np.square(samples)) / samples.size

    fs = lfp.get_sampling_rate()
    results = OrderedDict()
    for i, (l, u) in enumerate(splits):
        start_idx = math.floor(l * fs)
        end_idx = math.floor(u * fs)
        sample = lfp.get_samples()[start_idx:end_idx]
        power = np.sum(np.square(sample)) / sample.size
        results["Raw power {}".format(i)] = power
    return results


def lfp_entropy(lfp, splits=None, etype="sample"):
    """
    Calculate entropy on the lfp signal over multiple splits.

    etype can be "svd", "spectral", "sample" or "perm".
    see pyentropy for details on these.

    """
    def etrpy(sample, etype):
        if etype == "svd":
            et = entropy.svd_entropy(sample, order=3, delay=1)
        elif etype == "spectral":
            et = entropy.spectral_entropy(
                sample, 100, method='welch', normalize=True)
        elif etype == "sample":
            et = entropy.sample_entropy(sample, order=3)
        elif etype == "perm":
            et = entropy.perm_entropy(sample, order=3, normalize=True)
        else:
            print("Error: unrecognised entropy type {}".format(etype))
            exit(-1)
        return et

    lfp_samples = lfp.get_samples()
    if splits is None:
        return etrpy(lfp_samples, etype)
    fs = lfp.get_sampling_rate()
    results = OrderedDict()
    for i, (l, u) in enumerate(splits):
        start_idx = int(l * fs)
        end_idx = int(u * fs)
        sample = lfp_samples[start_idx:end_idx]
        et = etrpy(sample, etype)
        results["Entropy {}".format(i)] = et

    return results


def lfp_distribution_measures(
        lfp_odict, out_dir, splits, prefilt=False,
        get_entropy=False, get_theta=False, return_all=False):
    """
    Calculate average power and entropy measures over a 
    distribution of channels.

    if prefilt is true, use an already filtered lfp signal for these measures
    useful is a notch was need to be applied, or a band to remove some noise.

    Also plots a histogram of the samples and times over the channels.
    """
    lfp_s_array = []
    lfp_t_array = []
    avg = OrderedDict()
    ent = OrderedDict()

    if return_all:
        power_arr = np.zeros(shape=(len(lfp_odict), len(splits)))
    for j in range(len(splits)):
        avg["Avg power {}".format(j)] = 0
        if get_entropy:
            ent["Avg entropy {}".format(j)] = 0

    dict_to_use = (
        lfp_odict.get_filt_signal() if prefilt
        else lfp_odict.get_signal())
    for i, (_, lfp) in enumerate(dict_to_use.items()):
        lfp_s_array.append(lfp.get_samples())
        lfp_t_array.append(lfp.get_timestamp())
        p_result = raw_lfp_power(lfp, splits)

        for j in range(len(splits)):
            avg["Avg power {}".format(j)] += (
                p_result["Raw power {}".format(j)] / len(lfp_odict))
            power_arr[i, j] = p_result["Raw power {}".format(j)]
        if get_entropy:
            p_result = lfp_entropy(lfp, splits)
            for j in range(len(splits)):
                ent["Avg entropy {}".format(j)] += (
                    p_result["Entropy {}".format(j)] / len(lfp_odict))

    samples = np.concatenate(lfp_s_array)
    times = np.concatenate(lfp_t_array)

    fig, ax = plt.subplots()
    h = ax.hist2d(times, samples, bins=100)
    fig.colorbar(h[3])
    fig.savefig(os.path.join(out_dir, "dist.png"))

    if return_all:
        return avg, ent, power_arr
    return avg, ent


def lfp_theta_dist(lfp_odict, splits, prefilt=False, lower=None, upper=None):
    """
    Calculate theta power and delta power as well as the ratio.

    If prefilt is true, upper and lower need to be passed to match filtering
    of the lfp signal.

    """
    power_arr = np.zeros(shape=(6, len(lfp_odict), len(splits)))
    for i, (_, lfp) in enumerate(lfp_odict.get_signal().items()):
        for j, split in enumerate(splits):
            new_lfp = lfp.subsample(split)
            new_lfp.bandpower_ratio(
                [5, 11], [1.5, 4], 1.33, band_total=prefilt,
                first_name="Theta", second_name="Delta",
                totalband=(lower, upper))
            t_result = new_lfp.get_results()
            power_arr[0, i, j] = t_result["Theta Power"]
            power_arr[1, i, j] = t_result["Delta Power"]
            power_arr[2, i, j] = t_result["Theta Delta Power Ratio"]
            power_arr[3, i, j] = t_result["Theta Power (Relative)"]
            power_arr[4, i, j] = t_result["Delta Power (Relative)"]
            power_arr[5, i, j] = t_result["Total Power"]

    return power_arr


def single_main(parsed):
    """
    Main control function.

    An LFP signal power is analysed across multiple split up times.
    The times can be split to evaluate relationships over the course of change.
    The distribution of the LFP signal is also calculated over the channels.

    Proceeds as follows:
    1. Parse out the information from command line args.
    2. From this, set up the correct splits to analyse over.
    3. Plot a part of the given signal number to show effect of filtering.
    4. Calculate measures on the signal in each split 
        total lfp power, entropy
    5. Calculate these measures for each channel in the recording.
    6. Calculate theta and delta power for each channel in the recording.

    Args:
        parsed (SimpleNamespace): A namespace controlling the behaviour.

    Returns:
        tuple(dict, np.ndarray, np.ndarray) - 
        (power and entropy summary values, 
        raw power for each channel and each split, shape is (chans, splits), 
        bandpowers for each channel, shape is (6, chans, splits))

    """
    def setup_splits(every_min, split_s):
        """
        Determine the length of times to split recordings into.

        This is specifically set up for a 30 minute long recording.
        The full recording is always included in this.

        """
        if every_min:
            splits = [(60 * i, 60 * (i + 1)) for i in range(recording_dur)]
            splits.append((0, 600))
            splits.append((600, 1200))
            splits.append((1200, 1800))

        else:
            splits = []
            for i in range(len(split_s) // 2):
                splits.append((split_s[i * 2], split_s[i * 2 + 1]))

        splits.append((0, recording_dur * 60))

        return splits

    # Extract parsed args
    loc = parsed.loc
    if not loc:
        print("Please pass a file in through CLI")
        exit(-1)

    max_lfp = parsed.max_freq
    filt = not parsed.nofilt
    eeg_num = parsed.eeg_num
    split_s = parsed.splits
    out_loc = parsed.out_loc
    every_min = parsed.every_min
    recording_dur = parsed.recording_dur
    get_entropy = parsed.get_entropy
    return_all = True

    in_dir = os.path.dirname(loc)
    out_dir = os.path.join(in_dir, out_loc)
    print("Saving results to {}".format(out_dir))
    make_dir_if_not_exists(out_dir)

    splits = setup_splits(every_min, split_s)

    # Load the data
    # TODO only load certain channels here
    lfp_odict = LfpODict(loc, filt_params=(filt, 1.5, max_lfp))

    # Plot signals
    out_name = os.path.join(out_dir, "full_signal.png")
    plot_long_lfp(lfp_odict.get_signal(eeg_num), out_name)
    out_name = os.path.join(in_dir, out_dir, "full_signal_filt.png")
    plot_long_lfp(lfp_odict.get_filt_signal(eeg_num), out_name)
    graph_data = lfp_odict.get_signal(
        eeg_num).spectrum(
            fmax=90, db=False, tr=False, prefilt=True,
            filtset=(10, 1.5, 90, "bandpass"))
    fig = nc_plot.lfp_spectrum(graph_data)
    fig.savefig(os.path.join(out_dir, "spec.png"))
    graph_data = lfp_odict.get_signal(
        eeg_num).spectrum(
            fmax=90, db=True, tr=True, prefilt=True,
            filtset=(10, 1.5, 90, "bandpass"))
    fig = nc_plot.lfp_spectrum_tr(graph_data)
    fig.savefig(os.path.join(out_dir, "tr_spec.png"))
    plt.close("all")

    # Calculate power on this lfp channel
    lfp_to_use = (
        lfp_odict.get_filt_signal(eeg_num) if filt else
        lfp_odict.get_signal(eeg_num))
    p_results = raw_lfp_power(lfp_to_use, splits)

    # Calculate measures over the dist
    d_result = lfp_distribution_measures(
        lfp_odict, out_dir, splits[-4:], prefilt=filt,
        get_entropy=get_entropy, return_all=return_all)

    if get_entropy:
        # Calculate entropy on this lfp channel
        e_results = lfp_entropy(lfp_to_use)
        results = {
            "power": p_results,
            "entropy": e_results,
            "avg_power": d_result[0],
            "avg_entropy": d_result[1]
        }
    else:
        results = {
            "power": p_results,
            "avg_power": d_result[0]
        }

    save_mixed_dict_to_csv(results, out_dir)
    t_results = lfp_theta_dist(lfp_odict, splits, filt, 1.5, max_lfp)

    return results, d_result[-1], t_results


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


if __name__ == "__main__":
    single_main_cfg()
