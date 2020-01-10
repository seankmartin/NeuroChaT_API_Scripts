import os

from api_utils import read_cfg, parse_args, setup_logging, make_dir_if_not_exists, make_path_if_not_exists
from lfp_odict import LfpODict
import neurochat.nc_plot as nc_plot
from neurochat.nc_utils import get_all_files_in_dir

import matplotlib.pyplot as plt
from lfp_plot import plot_lfp

import api_plot_org as plot_org


def main(fname):
    chans = [i for i in range(1, 17)]
    lfp_odict = LfpODict(
        fname, channels=chans, filt_params=(True, 1.5, 90))
    o_dir = os.path.join(
        os.path.dirname(fname), "LFP")
    make_dir_if_not_exists(o_dir)

    # Plot periodogram for each eeg
    for i, (key, lfp) in enumerate(lfp_odict.get_signal().items()):
        graph_data = lfp.spectrum(
            ptype='psd', prefilt=False,
            db=False, tr=False,
            filtset=[10, 1.0, 40, 'bandpass'])
        fig = nc_plot.lfp_spectrum(graph_data)
        plt.ylim(0, 0.01)
        plt.xlim(0, 40)
        out_name = os.path.join(o_dir, "p", key + "p.png")
        make_path_if_not_exists(out_name)
        fig.savefig(out_name)
        plt.close()

        graph_data = lfp.spectrum(
            ptype='psd', prefilt=False,
            db=True, tr=True,
            filtset=[10, 1.0, 40, 'bandpass'])
        fig = nc_plot.lfp_spectrum_tr(graph_data)
        # plt.ylim(0, 0.01)
        # plt.xlim(0, 40)
        out_name = os.path.join(o_dir, "ptr", key + "ptr.png")
        make_path_if_not_exists(out_name)
        print("Saving result to {}".format(out_name))
        fig.savefig(out_name)
        plt.close()

    plot_lfp(o_dir, lfp_odict.get_filt_signal(), segment_length=60)

# Summary plots
    # Region info for eeg
    cla_idx = list(range(1, 9))
    acc_idx = list(range(9, 13))
    rsc_idx = list(range(13, 17))
    names = ["CLA"] * 8 + ["ACC"] * 4 + ["RSC"] * 4

    # Setup summary grid
    rows, cols = [4, 4]
    gf = plot_org.GridFig(rows, cols, wspace=0.5, hspace=0.5)

    # Plot summary periodogram
    for i, (key, lfp) in enumerate(lfp_odict.get_signal().items()):
        graph_data = lfp.spectrum(
            ptype='psd', prefilt=False,
            db=False, tr=False,
            filtset=[10, 1.0, 40, 'bandpass'])
        ax = gf.get_next(along_rows=False)
        nc_plot.lfp_spectrum(graph_data, ax)
        plt.ylim(0, 0.01)  
        plt.xlim(0, 40)
        if i%4 == 0:
            ax.text(0.49, 1.08, names[i], fontsize=20,
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        
    out_name = os.path.join(o_dir, "Sum", fname.split("\\")[-1] + "_p_sum.png")
    make_path_if_not_exists(out_name)
    print("Saving result to {}".format(out_name))
    gf.fig.savefig(out_name)
    plt.close()

    # Plot summary periodogram tr
    gf = plot_org.GridFig(rows, cols, wspace=0.5, hspace=0.5)
    for i, (key, lfp) in enumerate(lfp_odict.get_signal().items()):
        graph_data = lfp.spectrum(
            ptype='psd', prefilt=True,
            db=True, tr=True,
            filtset=[10, 1.0, 40, 'bandpass'])
        ax = gf.get_next(along_rows=False)
        nc_plot.lfp_spectrum_tr(graph_data, ax)
        plt.ylim(0, 40)  
        # plt.xlim(0, 40)
        if i%4 == 0:
            ax.text(0.49, 1.08, names[i], fontsize=20,
            horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        
    out_name = os.path.join(o_dir, "Sum", fname.split("\\")[-1] + "_ptr_sum.png")
    make_path_if_not_exists(out_name)
    print("Saving result to {}".format(out_name))
    gf.fig.savefig(out_name)
    plt.close()
    

if __name__ == "__main__":
    in_dir = r"F:\Ham Data\Now\CAR-SA2_20191130_1_PreBox"
    filenames = get_all_files_in_dir(
        in_dir, ext=".eeg", recursive=True,
        verbose=True)
    filenames = [fname[:-4] for fname in filenames]
    if len(filenames) == 0:
        print("No set files found for analysis!")
        exit(-1)
    for fname in filenames:
        main(fname)
