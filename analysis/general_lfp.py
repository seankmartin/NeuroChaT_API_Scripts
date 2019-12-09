import math
import os
from collections import OrderedDict
import numpy as np

import matplotlib.pyplot as plt

from neurochat.nc_lfp import NLfp
from neurochat.nc_utils import butter_filter


def get_all_lfp(filename, filt_params=(False, None, None), channels="all"):
    """
    Return an orderedDict of lfp objects, one for each channel.

    Args:
        filename (str): The basename of the file.
            filename+".eegX" are the final loaded files
        filt_params (tuple(bool, float, float), optional): 
            Tuple is structured as follows.
            (Should return filtered signal, lower_bound, upper_bound).
            Defaults to (False, None, None).
        channels (str or List, optional): Defaults to [1, 2, ..., 32].
            The list of channels to load.

    Returns:
        OrderedDict if not filtering
        or tuple(OrderedDict, OrderedDict) if filtering, where 
        the second object is the filtered dict.

    """
    filt, lower, upper = filt_params
    lfp_odict = OrderedDict()
    if channels == "all":
        channels = [i + 1 for i in range(32)]

    for i in channels:
        end = ".eeg"
        if i != 1:
            end = end + str(i)
        load_loc = filename + end
        lfp = NLfp(system="Axona")
        lfp.load(load_loc)
        lfp_odict[str(i)] = lfp

    if filt:
        if (lower is None) or (upper is None):
            print("Must provide lower and upper when filtering")
            exit(-1)
        lfp_filt_odict = OrderedDict()
        for key, lfp in lfp_odict.items():
            fs = lfp.get_sampling_rate()
            filtered_lfp = butter_filter(
                lfp.get_samples(), fs, 10,
                lower, upper, 'bandpass')
            lfp_filt_odict[key] = filtered_lfp
        return lfp_odict, lfp_filt_odict

    return lfp_odict


def plot_long_lfp(
        lfp, out_name, lower=None, upper=None,
        filt=True, nsamples=None, offset=0,
        nsplits=3, figsize=(32, 4), ylim=(-0.4, 0.4)):
    """
    Create a figure to display a long LFP signal in nsplits rows.

    Args:
        lfp (NLfp): The lfp signal to plot.
        out_name (str): The name of the output, including directory.
        lower, upper (int, int, optional): Defaults to None, None
            The lower and upper bands to use if filtering.
        filt (bool, optional): Defaults to True.
            Whether to filter the LFP signal. 
        nsamples (int, optional): Defaults to all samples.
            The number of samples to plot. 
        offset (int, optional): Defaults to 0.
            The number of samples into the lfp to start plotting at.
        nsplits (int, optional): The number of rows in the resulting figure.
        figsize (tuple of int, optional): Defaults to (32, 4)
        ylim (tuple of float, optional): Defaults to (-0.4, 0.4)

    Returns:
        NLfp: The filtered (or raw) lfp signal.

    """
    fs = lfp.get_sampling_rate()

    if nsamples is None:
        nsamples = lfp.get_total_samples()

    if filt:
        if (lower is None) or (upper is None):
            print("Must provide lower and upper when filtering")
            exit(-1)
        filtered_lfp = butter_filter(
            lfp.get_samples(), fs, 10,
            lower, upper, 'bandpass')
    else:
        filtered_lfp = lfp.get_samples()

    fig, axes = plt.subplots(nsplits, 1, figsize=figsize)
    for i in range(nsplits):
        start = int(offset + i * (nsamples // nsplits))
        end = int(offset + (i + 1) * (nsamples // nsplits))
        if nsplits == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.plot(
            lfp.get_timestamp()[start:end],
            filtered_lfp[start:end], color='k')
        ax.set_ylim(ylim)
    plt.tight_layout()
    fig.savefig(out_name, dpi=400)
    plt.close(fig)
    return filtered_lfp


def plot_lfp(
        out_dir, lfp_odict, segment_length=150, in_range=None, dpi=50):
    """
    Create a number of figures to display lfp signal on multiple channels.

    There will be one figure for each split of thetotal length 
    into segment_length, and a row for each value in lfp_odict.

    It is assumed that the input lfps are prefiltered if filtering is required.

    Args:
        out_dir (str): The name of the file to plot to, including dir.
        lfp_odict (OrderedDict): Keys as channels and NLfp objects as vals.
        segment_length (float): Time in seconds of LFP to plot in each figure.
        in_range (tuple(int, int), optional): Time(s) of LFP to plot overall.
            Defaults to None.
        dpi (int, optional): Resulting plot dpi.

    Returns:
        None
    """
    if in_range is None:
        in_range = (0, max([lfp.get_duration() for lfp in lfp_odict.values()]))
    y_axis_max = max([max(lfp.get_samples()) for lfp in lfp_odict.values()])
    y_axis_min = min([min(lfp.get_samples()) for lfp in lfp_odict.values()])
    for split in np.arange(in_range[0], in_range[1], segment_length):
        fig, axes = plt.subplots(
            nrows=len(lfp_odict),
            figsize=(20, len(lfp_odict) * 2))
        a = np.round(split, 2)
        b = np.round(min(split + segment_length, in_range[1]), 2)
        out_name = os.path.join(
            out_dir, "{}s_to_{}s.png".format(a, b))
        for i, (key, lfp) in enumerate(lfp_odict.items()):
            convert = lfp.get_sampling_rate()
            c_start, c_end = math.floor(a * convert), math.floor(b * convert)
            lfp_sample = lfp.get_samples()[c_start:c_end]
            x_pos = lfp.get_timestamp()[c_start:c_end]
            axes[i].plot(x_pos, lfp_sample, c="k")
            axes[i].text(
                0.03, 1, "Channel " + key,
                transform=axes[i].transAxes, c="k")
            axes[i].set_ylim(y_axis_min, y_axis_max)
            axes[i].set_xlim(a, b)
        print("Saving result to {}".format(out_name))
        fig.savefig(out_name, dpi=dpi)
        plt.close("all")
