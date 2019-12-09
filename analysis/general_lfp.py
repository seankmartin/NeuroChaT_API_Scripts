import math
import os
from collections import OrderedDict
import numpy as np

import matplotlib.pyplot as plt

from neurochat.nc_lfp import NLfp


def get_all_lfp(filename, channels="all"):
    """Return an orderedDict of lfp objects, one for each channel."""
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
    return lfp_odict


def plot_lfp(
        out_dir, lfp_odict, segment_length=150, in_range=None, dpi=50):
    """
    Create a number of figures to display lfp signal.

    There will be one figure for each split of the
    total length into segment_length and a row for each value in lfp_odict.
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
            x_pos = [a + (j / convert) for j in range(len(lfp_sample))]
            axes[i].plot(x_pos, lfp_sample, c="k")
            axes[i].text(
                0.03, 1, "Channel " + key,
                transform=axes[i].transAxes, c="k")
            axes[i].set_ylim(y_axis_min, y_axis_max)
            axes[i].set_xlim(a, b)
        print("Saving result to {}".format(out_name))
        fig.savefig(out_name, dpi=dpi)
        plt.close("all")
