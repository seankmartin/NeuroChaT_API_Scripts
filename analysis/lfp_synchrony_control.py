import os
import json
from collections import OrderedDict

from api_utils import read_cfg, parse_args, setup_logging, make_dir_if_not_exists
from lfp_plot import plot_lfp
from lfp_odict import LfpODict
from lfp_coherence import mvl_shuffle
from lfp_coherence import mean_vector_length
from lfp_plot import plot_coherence
from lfp_plot import plot_polar_coupling
from lfp_coherence import calc_coherence
from api_utils import save_mixed_dict_to_csv
import numpy as np

from matplotlib.pyplot import close
from neurochat.nc_utils import get_all_files_in_dir


def main(cfg, args, **kwargs):
    in_dir = cfg.get("Setup", "in_dir")
    out_dir = cfg.get("Output", "out_dirname")
    plot_dir = cfg.get("Output", "plot_dirname")
    re_filter = cfg.get("Setup", "regex_filter")
    s_filt = cfg.getboolean("LFP", "should_filter")
    re_filter = None if re_filter == "None" else re_filter
    analysis_flags = json.loads(config.get("Setup", "analysis_flags"))

    channel_dict_vc = cfg["VC"]
    channel_dict_cla = cfg["CLA"]
    channels = {}
    for key in channel_dict_vc.keys():
        channels[key] = [
            channel_dict_vc[key],
            channel_dict_cla[key]]
    setup_logging(in_dir)
    filenames = get_all_files_in_dir(
        in_dir, ext=".set", recursive=True,
        verbose=True, re_filter=re_filter, case_sensitive_ext=True)
    filenames = [fname[:-4] for fname in filenames]
    if len(filenames) == 0:
        print("No set files found for analysis!")
        exit(-1)

    # Plot signal on each loaded channel
    if analysis_flags[0]:
        for fname in filenames:
            lfp_odict = LfpODict(
                fname, channels="all", filt_params=(s_filt, 0, 90))
            o_dir = os.path.join(
                in_dir, out_dir, os.path.basename(fname))
            r = json.loads(config.get("LFP", "plot_time"))
            seg_len = float(config.get("LFP", "plot_seg_length"))
            make_dir_if_not_exists(o_dir)
            plot_lfp(
                o_dir, lfp_odict.get_filt_signal(),
                in_range=r, segment_length=seg_len, dpi=100)

    if analysis_flags[1]:
        # t_out_dir = os.path.join(in_dir, plot_dir)
        # make_dir_if_not_exists(t_out_dir)
        res_dict = OrderedDict()
        headers = [
            "Low freq chan", "high freq chan",
            "MVL", "MVL 95", "Z-score", "P-val"]
        res_dict["Name"] = headers
        for fname in filenames:
            for key, val in channels.items():
                if key in fname:
                    chan_list = val[::-1]
                    break
            else:
                raise ValueError("No key in {}, keys {}".format(
                    fname, channels.keys()))
            print("Computing mean value for {}, {}".format(
                fname, chan_list))
            res_dict = compute_mvl(fname, chan_list, res_dict)
        save_mixed_dict_to_csv(res_dict, in_dir, "no_mp_norm.csv")

    if analysis_flags[2]:
        make_dir_if_not_exists(os.path.join(in_dir, plot_dir))
        for fname in filenames:
            for key, val in channels.items():
                if key in fname:
                    chan_list = val[::-1]
                    break
            else:
                raise ValueError("No key in {}, keys {}".format(
                    fname, channels.keys()))
            out_basename = "{}_{}.png".format(
                os.path.basename(fname), chan_list)
            out_name = os.path.join(in_dir, plot_dir, out_basename)
            print("Saving coherence to {}".format(out_name))
            lfp_odict = LfpODict(fname, chan_list, (False, 0, 80))
            f, Cxy = calc_coherence(
                lfp_odict.get_filt_signal(0),
                lfp_odict.get_filt_signal(1))
            plot_coherence(f, Cxy, out_name, dpi=200)
            close("all")


def compute_mvl(recording, channels, res_dict, out_dir=None):
    """Assumed that low frequency is first."""
    lfp_odict = LfpODict(recording, channels)
    # Theta range
    low_freq_lfp = lfp_odict.filter(5, 11).get(channels[0])
    # Slow gamma is 30-55, fast gamma is 65-90
    high_freq_lfp = lfp_odict.filter(30, 55).get(channels[1])
    amp_norm = True
    res = np.zeros(6)
    res[0] = channels[0]
    res[1] = channels[1]
    res[2:] = mvl_shuffle(
        low_freq_lfp, high_freq_lfp, amp_norm=amp_norm, nshuffles=200)
    res_dict[os.path.basename(recording)] = res
    # out_name = os.path.join(
    #     out_dir,
    #     os.path.basename(recording) + "_{}_polar.png".format(channels))
    # pv, mvl = mean_vector_length(
    #     low_freq_lfp, high_freq_lfp, amp_norm=amp_norm, return_all=True)
    # plot_polar_coupling(pv, mvl, out_name, dpi=200)
    return res_dict


if __name__ == "__main__":
    """Parse args and cfg and send to main."""
    here = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(here, "Configs", "lfp_synchrony.cfg")
    config = read_cfg(config_path)
    args = parse_args()
    main(config, args)
