import sys
import os
import argparse
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
try:
    import entropy
except Exception as e:
    print("Could not import entropy module with error {}".format(e))

from general_lfp import plot_long_lfp
from api_utils import save_mixed_dict_to_csv

try:
    import neurochat.nc_plot as nc_plot
    from neurochat.nc_utils import butter_filter, make_dir_if_not_exists
    from neurochat.nc_data import NData
except Exception as e:
    print("Could not import neurochat modules with error {}".format(e))


def raw_lfp_power(lfp_samples, fs, splits, lower, upper, prefilt=False):
    """
    This can be used to get the power before splitting up the signal.

    Minor differences between this and filtering after splitting.
    """

    if prefilt:
        lfp_samples = butter_filter(
            lfp_samples, fs, 10, lower, upper, 'bandpass')

    results = OrderedDict()
    for i, (l, u) in enumerate(splits):
        start_idx = int(l * fs)
        end_idx = int(u * fs)
        sample = lfp_samples[start_idx:end_idx]
        power = np.sum(np.square(sample)) / sample.size
        results["Raw power {}".format(i)] = power
    return results


def lfp_power(new_data, i, max_f, in_dir, prefilt=False, should_plot=True):
    # 1.6 or 2 give similar
    filtset = [10, 1.5, max_f, 'bandpass']

    new_data.bandpower_ratio(
        [5, 11], [1.5, 4], 1.6, band_total=prefilt,
        first_name="Theta", second_name="Delta",
        totalband=filtset[1:3])

    if should_plot:
        graphData = new_data.spectrum(
            window=1.6, noverlap=1, nfft=1000, ptype='psd', prefilt=prefilt,
            filtset=filtset, fmax=max_f, db=False, tr=False)
        fig = nc_plot.lfp_spectrum(graphData)
        fig.savefig(os.path.join(in_dir, "spec" + str(i) + ".png"))

        graphData = new_data.spectrum(
            window=1.6, noverlap=1, nfft=1000, ptype='psd', prefilt=prefilt,
            filtset=filtset, fmax=max_f, db=True, tr=True)
        fig = nc_plot.lfp_spectrum_tr(graphData)
        fig.savefig(os.path.join(in_dir, "spec_tr" + str(i) + ".png"))

    return new_data.get_results()


def lfp_entropy(
        lfp_samples, fs, splits, lower, upper, prefilt=False, etype="sample"):
    results = OrderedDict()

    if prefilt:
        lfp_samples = butter_filter(
            lfp_samples, fs, 10, lower, upper, 'bandpass')

    for i, (l, u) in enumerate(splits):
        start_idx = int(l * fs)
        end_idx = int(u * fs)
        sample = lfp_samples[start_idx:end_idx]
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
            print("Error: unrecognised entropy type {}".format(
                etype
            ))
            exit(-1)
        results["Entropy {}".format(i)] = et

    return results


def lfp_distribution(
        filename, upper, out_dir, splits,
        prefilt=False, get_entropy=False,
        get_theta=False, return_all=False):
    lfp_s_array = []
    lfp_t_array = []
    data = NData()
    avg = OrderedDict()
    ent = OrderedDict()

    if return_all:
        power_arr = np.zeros(shape=(32, len(splits)))
    for j in range(len(splits)):
        avg["Avg power {}".format(j)] = 0
        if get_entropy:
            ent["Avg entropy {}".format(j)] = 0

    for i in range(32):
        end = str(i + 1)
        if end == "1":
            load_loc = filename
        else:
            load_loc = filename + end
        data.lfp.load(load_loc)
        lfp_samples = data.lfp.get_samples()
        fs = data.lfp.get_sampling_rate()
        if prefilt:
            lfp_samples = butter_filter(
                lfp_samples, fs, 10, 1.5, upper, 'bandpass')
        lfp_s_array.append(lfp_samples)
        lfp_t_array.append(data.lfp.get_timestamp())

        p_result = raw_lfp_power(
            lfp_samples, fs, splits, 1.5, upper, prefilt=False)
        for j in range(len(splits)):
            avg["Avg power {}".format(j)] += (
                p_result["Raw power {}".format(j)] / 32)
            power_arr[i, j] = p_result["Raw power {}".format(j)]
        if get_entropy:
            p_result = lfp_entropy(
                lfp_samples, fs, splits, 1.5, upper, prefilt=False)
            for j in range(len(splits)):
                ent["Avg entropy {}".format(j)] += (
                    p_result["Entropy {}".format(j)] / 32)

    samples = np.concatenate(lfp_s_array)
    times = np.concatenate(lfp_t_array)

    fig, ax = plt.subplots()
    h = ax.hist2d(times, samples, bins=100)
    fig.colorbar(h[3])
    fig.savefig(os.path.join(out_dir, "dist.png"))

    if return_all:
        return avg, ent, power_arr
    return avg, ent


def lfp_theta_dist(filename, max_f, splits, prefilt=False):
    data = NData()
    filtset = [10, 1.5, max_f, 'bandpass']
    # This is for theta and delta power
    power_arr = np.zeros(shape=(6, 32, len(splits)))
    for i in range(32):
        end = str(i + 1)
        if end == "1":
            load_loc = filename
        else:
            load_loc = filename + end
        data.lfp.load(load_loc)
        for j, split in enumerate(splits):
            new_data = data.subsample(split)
            new_data.bandpower_ratio(
                [5, 11], [1.5, 4], 1.33, band_total=prefilt,
                first_name="Theta", second_name="Delta",
                totalband=filtset[1:3])
            t_result = new_data.get_results()
            power_arr[0, i, j] = t_result["Theta Power"]
            power_arr[1, i, j] = t_result["Delta Power"]
            power_arr[2, i, j] = t_result["Theta Delta Power Ratio"]
            power_arr[3, i, j] = t_result["Theta Power (Relative)"]
            power_arr[4, i, j] = t_result["Delta Power (Relative)"]
            power_arr[5, i, j] = t_result["Total Power"]

    return power_arr


def plot_sample_of_signal(
        load_loc, out_dir=None, name=None, offseta=0, length=50):
    """Plot a small filtered sample of the LFP signal."""
    in_dir = os.path.dirname(load_loc)
    ndata = NData()
    ndata.lfp.load(load_loc)

    if out_dir is None:
        out_loc = "nc_signal"
        out_dir = os.path.join(in_dir, out_loc)

    if name is None:
        name = "full_signal_filt.png"

    out_name = os.path.join(out_dir, name)
    make_dir_if_not_exists(out_name)
    plot_long_lfp(
        ndata.lfp, 5, 11, out_name, filt=True,
        offset=ndata.lfp.get_sampling_rate() * offseta,
        nsamples=ndata.lfp.get_sampling_rate() * length,
        nsplits=1, ylim=(-0.3, 0.3),
        figsize=(20, 8))


def single_main(parsed):
    """
    Main control function.

    An LFP signal power is analysed across multiple split up times.
    The times can be split to evaluate relationships over the course of change.

    Proceeds as follows:
    1. Parse out the information from command line args.
    2. From this, set up the correct splits to analyse over.
    3. Plot a part of the signal to show effect of filtering.
    4. Calculate measures on the signal in each split 
        total lfp power, entropy
    5. Calculate these measures for each tetrode in the recording.
    6. Calculate theta and delta power for each tetrode in the recording.

    Args:
        parsed (SimpleNamespace): A namespace controlling the behaviour.

    Returns:
        tuple(dict, OrderedDict, np.ndarray) - 
        (power and entropy summary values, raw power for each channel, 
        bandpowers for each channel)

    """
    # Extract parsed args
    max_lfp = parsed.max_freq
    filt = not parsed.nofilt
    loc = parsed.loc
    eeg_num = parsed.eeg_num
    split_s = parsed.splits
    out_loc = parsed.out_loc
    every_min = parsed.every_min
    recording_dur = parsed.recording_dur
    get_entropy = parsed.get_entropy
    return_all = parsed.g_all

    if not loc:
        print("Please pass a file in through CLI")
        exit(-1)

    # This is specifically set up for a 30 minute long recording.
    if every_min:
        splits = [(60 * i, 60 * (i + 1)) for i in range(recording_dur)]
        splits.append((0, 600))
        splits.append((600, 1200))
        splits.append((1200, 1800))

    else:
        splits = []
        for i in range(len(split_s) // 2):
            splits.append((split_s[i * 2], split_s[i * 2 + 1]))

    # Always include the full recording in this
    splits.append((0, recording_dur * 60))

    if eeg_num != "1":
        load_loc = loc + eeg_num
    else:
        load_loc = loc

    in_dir = os.path.dirname(load_loc)
    ndata = NData()
    ndata.lfp.load(load_loc)
    out_dir = os.path.join(in_dir, out_loc)

    if ndata.lfp.get_duration() == 0:
        print("Failed to correctly load lfp at {}".format(
            load_loc))
        exit(-1)

    print("Saving results to {}".format(out_dir))
    make_dir_if_not_exists(os.path.join(out_dir, "dummy.txt"))

    # Plot signals
    out_name = os.path.join(out_dir, "full_signal.png")
    plot_long_lfp(
        ndata.lfp, out_name, filt=False)
    out_name = os.path.join(
        in_dir, out_dir, "full_signal_filt.png")
    filtered_lfp = plot_long_lfp(
        ndata.lfp, out_name, lower=1.5, upper=max_lfp, filt=True)
    if not filt:
        filtered_lfp = ndata.lfp

    # Calculate measures on this tetrode
    fs = ndata.lfp.get_sampling_rate()
    p_results = raw_lfp_power(
        filtered_lfp, fs, splits, 1.5, max_lfp, prefilt=False)

    if get_entropy:
        e_results = lfp_entropy(
            filtered_lfp, fs, splits[-4:], 1.5, max_lfp, prefilt=False)

    # Calculate measures over the dist
    d_result = lfp_distribution(
        loc, max_lfp, out_dir, splits[-4:],
        prefilt=filt, get_entropy=get_entropy, return_all=return_all)

    if get_entropy:
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

    t_results = lfp_theta_dist(
        loc, max_lfp, splits, prefilt=filt)

    return results, d_result[-1], t_results
