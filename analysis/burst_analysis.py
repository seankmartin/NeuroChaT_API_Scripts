"""Burst analysis of cells."""
import csv
import os
import sys
from copy import copy
import logging
import configparser
import json
from pprint import pprint
import argparse

from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as shc
import numpy as np
import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from neurochat.nc_datacontainer import NDataContainer
from neurochat.nc_containeranalysis import place_cell_summary
from neurochat.nc_utils import make_dir_if_not_exists, log_exception, oDict
from neurochat.nc_utils import remove_extension
import neurochat.nc_plot as nc_plot

from api_utils import save_dicts_to_csv


def log_isi(ndata, start=0.0005, stop=10, num_bins=60):
    """
    Compute the log_isi from an NData object

    Params
    ------
    start - the start time in seconds for the ISI
    stop - the stop time in seconds for the ISI
    num_bins - the number of bins in the ISI
    """
    isi_log_bins = np.linspace(
        np.log10(start), np.log10(stop), num_bins + 1)
    hist, _ = np.histogram(
        np.log10(np.diff(ndata.spike.get_unit_stamp())),
        bins=isi_log_bins, density=False)
    return hist / ndata.spike.get_unit_stamp().size, isi_log_bins

    # alternatively, using neurochat built in method.
    # return ndata.isi(bins=isi_log_bins, density=True), isi_log_bins


def cell_classification_stats(
        in_dir, container, out_name,
        should_plot=False, opt_end="", output_spaces=True):
    """
    Compute a csv of cell stats for each unit in a container

    Params
    ------
    in_dir - the data output/input location
    container - the NDataContainer object to get stats for
    should_plot - whether to save some plots for this
    """
    _results = []
    spike_names = container.get_file_dict()["Spike"]
    for i, ndata in enumerate(container):
        try:
            data_idx, unit_idx = container._index_to_data_pos(i)
            name = spike_names[data_idx][0]
            parts = os.path.basename(name).split(".")
            o_name = os.path.join(
                os.path.dirname(name)[len(in_dir + os.sep):],
                parts[0])
            print("Working on unit {} of {}: {}, T{}, U{}".format(
                i, len(container), o_name, parts[-1], ndata.get_unit_no()))

            # Setup up identifier information
            note_dict = oDict()
            dir_t = os.path.dirname(name)
            note_dict["Index"] = i
            note_dict["FullDir"] = dir_t
            if dir_t != in_dir:
                note_dict["RelDir"] = os.path.dirname(
                    name)[len(in_dir + os.sep):]
            else:
                note_dict["RelDir"] = ""
            note_dict["Recording"] = parts[0]
            note_dict["Tetrode"] = int(parts[-1])
            note_dict["Unit"] = ndata.get_unit_no()
            ndata.update_results(note_dict)

            # Caculate cell properties
            ndata.wave_property()
            # ndata.place()
            # ndata.hd_rate()
            # ndata.grid()
            # ndata.border()
            # ndata.multiple_regression()
            ndata.isi()
            ndata.burst(burst_thresh=5)
            ndata.phase_dist()
            ndata.theta_index()
            # ndata.bandpower_ratio(
            #     [5, 11], [1.5, 4], 1.6, relative=True,
            #     first_name="Theta", second_name="Delta")
            result = copy(ndata.get_results(
                spaces_to_underscores=not output_spaces))
            _results.append(result)

            if should_plot:
                # plot_loc = os.path.join(
                #     in_dir, "nc_plots", parts[0] + "_" + parts[-1] + "_" +
                #     str(ndata.get_unit_no()) + "_phase" + opt_end + ".png")
                # make_dir_if_not_exists(plot_loc)
                # fig1, fig2, fig3 = nc_plot.spike_phase(phase_dist)
                # fig2.savefig(plot_loc)
                # plt.close("all")

                if unit_idx == len(container.get_units(data_idx)) - 1:
                    plot_loc = os.path.join(
                        in_dir, "nc_plots", parts[0] +
                        "_lfp_periodogram" + opt_end + ".png")
                    make_dir_if_not_exists(plot_loc)

                    lfp_spectrum = ndata.spectrum()
                    fig = nc_plot.lfp_spectrum(lfp_spectrum)
                    fig.savefig(plot_loc)
                    plt.close(fig)

                if unit_idx == len(container.get_units(data_idx)) - 1:
                    plot_loc = os.path.join(
                        in_dir, "nc_plots", parts[0] +
                        "_lfp_spectrum" + opt_end + ".png")
                    make_dir_if_not_exists(plot_loc)

                    graph_data = ndata.spectrum(
                        ptype='psd', prefilt=True,
                        db=True, tr=True,
                        filtset=[10, 1.0, 40, 'bandpass'],
                        fmax=40)
                    fig = nc_plot.lfp_spectrum_tr(graph_data)
                    fig.savefig(plot_loc)
                    plt.close(fig)

        except Exception as e:
            print("WARNING: Failed to analyse {} unit {}".format(
                os.path.basename(name), note_dict["Unit"]))
            log_exception(e, "Failed on {} unit {}".format(
                os.path.basename(name), note_dict["Unit"]))

    # Save the cell statistics
    make_dir_if_not_exists(out_name)
    save_dicts_to_csv(out_name, _results)
    _results.clear()


def calculate_isi_hist(container, in_dir, opt_end="", s_color=False):
    """Calculate a matrix of isi_hists for each unit in a container"""
    ax1, fig1 = nc_plot._make_ax_if_none(None)
    isi_hist_matrix = np.empty((len(container), 60), dtype=float)
    color = iter(cm.gray(np.linspace(0, 0.8, len(container))))
    for i, ndata in enumerate(container):
        res_isi, bins = log_isi(ndata)
        isi_hist_matrix[i] = res_isi
        bin_centres = bins[:-1] + np.mean(np.diff(bins)) / 2
        c = next(color) if s_color else "k"
        ax1.plot(bin_centres, res_isi, c=c)
        ax1.set_xlim([-3, 1])
        ax1.set_xticks([-3, -2, -1, 0])
    ax1.axvline(x=np.log10(0.006), c="r", ls="--")

    plot_loc = os.path.join(
        in_dir, "nc_plots", "logisi" + opt_end + ".png")
    fig1.savefig(plot_loc, dpi=400)

    return isi_hist_matrix


def calculate_auto_corr(container, in_dir, opt_end="", s_color=False):
    """Calculate a matrix of autocorrs for each unit in a container"""
    ax1, fig1 = nc_plot._make_ax_if_none(None)
    auto_corr_matrix = np.empty((len(container), 20), dtype=float)
    color = iter(cm.gray(np.linspace(0, 0.8, len(container))))
    for i, ndata in enumerate(container):
        auto_corr_data = ndata.isi_auto_corr(bins=1, bound=[0, 20])
        auto_corr_matrix[i] = (auto_corr_data["isiCorr"] /
                               ndata.spike.get_unit_stamp().size)
        bins = auto_corr_data['isiAllCorrBins']
        bin_centres = bins[:-1] + np.mean(np.diff(bins)) / 2
        c = next(color) if s_color else "k"
        ax1.plot(bin_centres / 1000, auto_corr_matrix[i], c=c)
        ax1.set_xlim([0.000, 0.02])
        ax1.set_xticks([0.000, 0.005, 0.01, 0.015, 0.02])
    ax1.axvline(x=0.006, c="r", ls="--")

    plot_loc = os.path.join(
        in_dir, "nc_plots", "autocorr" + opt_end + ".png")
    fig1.savefig(plot_loc, dpi=400)
    return auto_corr_matrix


def perform_pca(data, n_components=3, should_scale=True):
    """
    Perform PCA on a set of data (e.g. ndarray)

    Params
    ------
    data - input data array
    n_components - the number of PCA components to compute
        if this is a float, uses enough components to reach that much variance
    should_scale - whether to scale the data to unit variance
    """
    scaler = StandardScaler()
    pca = PCA(n_components=n_components)

    # Standardise the data to improve PCA performance
    if should_scale:
        std_data = scaler.fit_transform(data)
        after_pca = pca.fit_transform(std_data)
    else:
        after_pca = pca.fit_transform(data)

    print("PCA fraction of explained variance", pca.explained_variance_ratio_)
    return after_pca, pca


def ward_clustering(
        data, in_dir, plot_dim1=0, plot_dim2=1, opt_end="", s_color=False):
    """
    Perform heirarchical clustering using ward's method

    Params
    ------
    data - input data array
    in_dir - where to save the result to
    plot_dim1 - the PCA dimension to plot
    plot_dim2 - the other PCA dimesion to plot
    """
    ax, fig = nc_plot._make_ax_if_none(None)
    if s_color:
        shc.set_link_color_palette(
            ['#4A777A', "#6996AD", "#82CFFD", "#3579DC"])
    else:
        shc.set_link_color_palette(["k"])

    atc = '#bcbddc'
    dend = shc.dendrogram(
        shc.linkage(data, method="ward", optimal_ordering=True),
        ax=ax, above_threshold_color=atc, orientation="right", no_labels=True)
    plot_loc = os.path.join(
        in_dir, "nc_plots", "dendogram" + opt_end + ".png")
    fig.savefig(plot_loc, dpi=400)
    shc.set_link_color_palette(None)

    cluster = AgglomerativeClustering(
        n_clusters=2, affinity="euclidean", linkage="ward")
    cluster.fit_predict(data)

    ax, fig = nc_plot._make_ax_if_none(None)
    markers = list(map(lambda a: "Burst" if a else "Regular", cluster.labels_))
    # ax.scatter(data[:, plot_dim1], data[:, plot_dim2],
    #            c=markers)
    sns.scatterplot(
        data[:, plot_dim1], data[:, plot_dim2], ax=ax,
        style=markers, hue=markers)
    plot_loc = os.path.join(
        in_dir, "nc_plots", "PCAclust" + opt_end + ".png")
    fig.savefig(plot_loc, dpi=400)

    return cluster, dend


def save_pca_res(
        container, fname, n_isi_comps, n_auto_comps,
        isi_pca, corr_pca, clust, dend, joint_pca):
    with open(fname, "w") as f:
        f.write("Type")
        for _ in range(max(n_isi_comps, n_auto_comps)):
            f.write(",Variance Ratio")
        f.write("\nISI_PCA")
        for val in isi_pca.explained_variance_ratio_:
            f.write("," + str(val))
        f.write("\nACH_PCA")
        for val in corr_pca.explained_variance_ratio_:
            f.write("," + str(val))
        f.write("\n")
        f.write("\n")
        f.write("Name,Unit,Clust Label,Dend Leaf,ISI PCA,AC PCA\n")
        d_idx = dend["leaves"][::-1]
        for i in range(len(container)):
            idx_info = container.get_index_info(i, absolute=True)
            idx = d_idx.index(i)
            val = clust.labels_[i]
            pca1 = joint_pca[i, 0]
            pca2 = joint_pca[i, 3]
            f.write("{},{},{},{},{},{}\n".format(
                idx_info["Spike"], idx_info["Units"],
                val, idx, pca1, pca2))
        f.write("\n")


def plot_clustering(
        container, in_dir, isi_hist, ac_hist, dend, cluster, opt_end=""):
    dend_idxs = dend["leaves"][::-1]
    sorted_isi = isi_hist[dend_idxs]
    sorted_ac = ac_hist[dend_idxs]

    ax1, fig1 = nc_plot._make_ax_if_none(None)
    cmap = sns.cubehelix_palette(
        8, start=0.5, rot=-.75, dark=0, light=.95, reverse=True)
    # cmap = sns.color_palette("Blues")
    sns.heatmap(
        sorted_isi, ax=ax1, yticklabels=5,
        xticklabels=10, cmap=cmap)
    ax1.set_ylim([sorted_isi.shape[0], 0])
    ax1.axvline(x=16, c="r", ls="--")
    plot_loc = os.path.join(
        in_dir, "nc_plots", "logisi_hist" + opt_end + ".png")
    fig1.savefig(plot_loc, dpi=400)
    fig1.clear()
    ax1, fig1 = nc_plot._make_ax_if_none(None)
    sns.heatmap(
        sorted_ac, ax=ax1, yticklabels=5,
        xticklabels=5, cmap=cmap)
    ax1.axvline(x=6, c="r", ls="--")
    ax1.set_ylim([sorted_ac.shape[0], 0])
    plot_loc = os.path.join(
        in_dir, "nc_plots", "ac_hist" + opt_end + ".png")
    fig1.savefig(plot_loc, dpi=400)
    fig1.clear()


def pca_clustering(
        container, in_dir, n_isi_comps=3, n_auto_comps=2,
        opt_end="", s_color=False):
    """
    Wraps up other functions to do PCA clustering on a container.

    Computes PCA for ISI and AC and then clusters based on these.

    Params
    ------
    container - the input NDataContainer to consider
    in_dir - the directory to save information to
    n_isi_comps - the number of principal components for isi
    n_auto_comps - the number of principla components for auto_corr
    """
    print("Considering ISIH PCA")
    make_dir_if_not_exists(os.path.join(in_dir, "nc_plots", "dummy.txt"))
    isi_hist_matrix = calculate_isi_hist(
        container, in_dir, opt_end=opt_end, s_color=s_color)
    isi_after_pca, isi_pca = perform_pca(
        isi_hist_matrix, n_isi_comps, True)
    print("Considering ACH PCA")
    auto_corr_matrix = calculate_auto_corr(
        container, in_dir, opt_end=opt_end, s_color=s_color)
    corr_after_pca, corr_pca = perform_pca(
        auto_corr_matrix, n_auto_comps, True)
    joint_pca = np.empty(
        (len(container), n_isi_comps + n_auto_comps), dtype=float)
    joint_pca[:, :n_isi_comps] = isi_after_pca
    joint_pca[:, n_isi_comps:n_isi_comps + n_auto_comps] = corr_after_pca
    clust, dend = ward_clustering(
        joint_pca, in_dir, 0, 3, opt_end=opt_end, s_color=s_color)
    plot_clustering(
        container, in_dir, isi_hist_matrix, auto_corr_matrix,
        dend, clust, opt_end=opt_end)
    fname = os.path.join(
        in_dir, "nc_results", "PCA_results" + opt_end + ".csv")
    save_pca_res(
        container, fname, n_isi_comps, n_auto_comps,
        isi_pca, corr_pca, clust, dend, joint_pca)


def main(args, config):
    # Unpack out the cfg file into easier names
    in_dir = config.get("Setup", "in_dir")
    regex_filter = config.get("Setup", "regex_filter")
    regex_filter = None if regex_filter == "None" else regex_filter
    analysis_flags = json.loads(config.get("Setup", "analysis_flags"))
    tetrode_list = json.loads(config.get("Setup", "tetrode_list"))
    should_filter = config.getboolean("Setup", "should_filter")
    seaborn_style = config.getboolean("Plot", "seaborn_style")
    plot_order = json.loads(config.get("Plot", "plot_order"))
    fixed_color = config.get("Plot", "path_color")
    fixed_color = None if fixed_color == "None" else fixed_color
    if len(fixed_color) > 1:
        fixed_color = json.loads(fixed_color)
    s_color = config.getboolean("Plot", "should_color")
    plot_outname = config.get("Plot", "output_dirname")
    dot_size = config.get("Plot", "dot_size")
    dot_size = None if dot_size == "None" else int(dot_size)
    summary_dpi = int(config.get("Plot", "summary_dpi"))
    hd_predict = config.getboolean("Plot", "hd_predict")
    output_format = config.get("Output", "output_format")
    save_bin_data = config.getboolean("Output", "save_bin_data")
    output_spaces = config.getboolean("Output", "output_spaces")
    opt_end = config.get("Output", "optional_end")
    max_units = int(config.get("Setup", "max_units"))
    isi_bound = int(config.get("Params", "isi_bound"))
    isi_bin_length = int(config.get("Params", "isi_bin_length"))

    setup_logging(in_dir)

    if output_format == "pdf":
        matplotlib.use("pdf")

    if seaborn_style:
        sns.set(palette="colorblind")
    else:
        sns.set_style(
            "ticks",
            {'axes.spines.right': False, 'axes.spines.top': False})

    # Automatic extraction of files from starting dir onwards
    container = NDataContainer(load_on_fly=True)
    out_name = container.add_axona_files_from_dir(
        in_dir, tetrode_list=tetrode_list, recursive=True, re_filter=regex_filter, verbose=False, unit_cutoff=(0, max_units))
    container.setup()
    if len(container) == 0:
        print("Unable to find any files matching regex {}".format(
            regex_filter))
        exit(-1)

    # Show summary of place
    if analysis_flags[0]:
        place_cell_summary(
            container, dpi=summary_dpi, out_dirname=plot_outname,
            filter_place_cells=should_filter, filter_low_freq=should_filter,
            opt_end=opt_end, base_dir=in_dir,
            output_format=output_format, isi_bound=isi_bound,
            isi_bin_length=isi_bin_length, output=plot_order,
            save_data=save_bin_data, fixed_color=fixed_color,
            point_size=dot_size, color_isi=s_color, burst_thresh=6,
            hd_predict=hd_predict)
        plt.close("all")

    # Do numerical analysis of bursting
    should_plot = analysis_flags[2]
    if analysis_flags[1]:
        import re
        out_name = remove_extension(out_name) + "csv"
        out_name = re.sub(r"file_list_", r"cell_stats_", out_name)
        print("Computing cell stats to save to {}".format(out_name))
        cell_classification_stats(
            in_dir, container, out_name,
            should_plot=should_plot, opt_end=opt_end,
            output_spaces=output_spaces)

    # Do PCA based analysis
    if analysis_flags[3]:
        print("Computing pca clustering")
        pca_clustering(container, in_dir, opt_end=opt_end, s_color=s_color)


def setup_logging(in_dir):
    fname = os.path.join(in_dir, 'nc_output.log')
    if os.path.isfile(fname):
        open(fname, 'w').close()
    logging.basicConfig(
        filename=fname, level=logging.DEBUG)
    mpl_logger = logging.getLogger("matplotlib")
    mpl_logger.setLevel(level=logging.WARNING)


def print_config(config, msg=""):
    if msg != "":
        print(msg)
    """Prints the contents of a config file"""
    config_dict = [{x: tuple(config.items(x))} for x in config.sections()]
    pprint(config_dict, width=120)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    here = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(here, "Configs", "burst_analysis.cfg")
    config.read(config_path)

    parser = argparse.ArgumentParser(
        description='Process modifiable parameters from command line')
    args, unparsed = parser.parse_known_args()

    if len(unparsed) != 0:
        print("Unrecognised command line argument passed")
        print(unparsed)
        exit(-1)

    print_config(config, "Program started with configuration")
    if len(sys.argv) > 1:
        print("Command line arguments", args)

    main(args, config)
