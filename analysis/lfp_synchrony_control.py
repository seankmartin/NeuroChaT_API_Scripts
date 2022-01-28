import os
import json
from collections import OrderedDict

from api_utils import (
    read_cfg,
    parse_args,
    setup_logging,
    make_dir_if_not_exists,
    make_path_if_not_exists,
)
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
    filter_range = json.loads(cfg.get("LFP", "filter_range"))
    re_filter = None if re_filter == "None" else re_filter
    analysis_flags = json.loads(cfg.get("Setup", "analysis_flags"))
    res_name = kwargs.get("res_name", "")

    channel_dict_vc = cfg["VC"]
    channel_dict_cla = cfg["CLA"]
    channels = {}
    for key in channel_dict_vc.keys():
        channels[key] = [channel_dict_vc[key], channel_dict_cla[key]]
    setup_logging(in_dir)
    filenames = get_all_files_in_dir(
        in_dir,
        ext=".set",
        recursive=True,
        verbose=True,
        re_filter=re_filter,
        case_sensitive_ext=True,
    )
    filenames = [fname[:-4] for fname in filenames]
    if len(filenames) == 0:
        print("No set files found for analysis!")
        exit(-1)

    # Plot signal on each loaded channel
    if analysis_flags[0]:
        for fname in filenames:
            lfp_odict = LfpODict(
                fname, channels="all", filt_params=(s_filt, *filter_range)
            )
            o_dir = os.path.join(in_dir, out_dir, os.path.basename(fname))
            r = json.loads(cfg.get("LFP", "plot_time"))
            seg_len = float(cfg.get("LFP", "plot_seg_length"))
            make_dir_if_not_exists(o_dir)
            plot_lfp(
                o_dir,
                lfp_odict.get_filt_signal(),
                in_range=r,
                segment_length=seg_len,
                dpi=100,
            )

    if analysis_flags[1]:
        # t_out_dir = os.path.join(in_dir, plot_dir)
        # make_dir_if_not_exists(t_out_dir)
        res_dict = OrderedDict()
        headers = [
            "Low freq chan",
            "high freq chan",
            "MVL",
            "MVL 95",
            "Z-score",
            "P-val",
        ]
        res_dict["Name"] = headers
        for fname in filenames:
            for key, val in channels.items():
                if key in fname:
                    chan_list = val[::-1]
                    break
            else:
                raise ValueError("No key in {}, keys {}".format(fname, channels.keys()))
            print("Computing mean value for {}, {}".format(fname, chan_list))
            res_dict = compute_mvl(fname, chan_list, res_dict)
        save_mixed_dict_to_csv(res_dict, out_dir, "no_mp_norm.csv")

    if analysis_flags[2]:
        res_dict = OrderedDict()
        theta_delta_dict = OrderedDict()
        theta_delta_dict["Name"] = [
            "delta_peak_D",
            "theta_peak_D",
            "delta_avg_D",
            "theta_avg_D",
            "delta_peak_L1",
            "theta_peak_L1",
            "delta_avg_L1",
            "theta_avg_L1",
            "delta_peak_L2",
            "theta_peak_L2",
            "delta_avg_L2",
            "theta_avg_L2",
        ]
        make_dir_if_not_exists(os.path.join(out_dir, plot_dir, "coherence"))
        for fname in filenames:
            for key, val in channels.items():
                if key in fname:
                    chan_list = val
                    break
            else:
                raise ValueError("No key in {}, keys {}".format(fname, channels.keys()))
            if "green" in fname:
                continue

            out_basename = "{}_{}.png".format(os.path.basename(fname), chan_list)
            out_name = os.path.join(out_dir, plot_dir, "coherence", out_basename)
            print("Saving coherence to {}".format(out_name))
            lfp_odict = LfpODict(fname, chan_list, (False, 0, 80))

            f, Cxy = calc_coherence(
                lfp_odict.get_filt_signal(0), lfp_odict.get_filt_signal(1)
            )
            if "Name" not in res_dict:
                res_dict["Name"] = f
            res_dict[fname] = Cxy
            plot_coherence(f, Cxy, out_name, dpi=200)
            close("all")

            fname_without_end = "-".join(fname.split("-")[:-1])
            if fname_without_end not in theta_delta_dict:
                theta_delta_dict[fname_without_end] = []

            delta_bit = np.nonzero(np.logical_and(f >= 1.5, f <= 4.0))
            theta_bit = np.nonzero(np.logical_and(f >= 5.0, f <= 11.0))
            v1 = np.max(Cxy[delta_bit])
            v2 = np.max(Cxy[theta_bit])
            v3 = np.mean(Cxy[delta_bit])
            v4 = np.mean(Cxy[theta_bit])
            for val in [v1, v2, v3, v4]:
                theta_delta_dict[fname_without_end].append(val)
        save_mixed_dict_to_csv(
            res_dict, os.path.join(out_dir, plot_dir), f"Coherence_{res_name}.csv"
        )
        save_mixed_dict_to_csv(
            theta_delta_dict,
            os.path.join(out_dir, plot_dir),
            f"Coherence_avg_{res_name}.csv",
        )

    if analysis_flags[3]:
        import neurochat.nc_plot as nc_plot
        from lfp_plot import plot_long_lfp

        make_dir_if_not_exists(os.path.join(out_dir, plot_dir))
        for fname in filenames:
            for key, val in channels.items():
                if key in fname:
                    chan_list = val
                    break
            else:
                raise ValueError("No key in {}, keys {}".format(fname, channels.keys()))
            lfp_odict = LfpODict(fname, chan_list, (True, 1, 90))

            # Green was corrupted by 50Hz current in LFP
            if "green" in fname:
                lfp_odict.notch_filter(channels=["1", "2"])

            for chan in chan_list:
                out_basepart = os.path.join(
                    out_dir, plot_dir, os.path.basename(fname), chan
                )
                make_path_if_not_exists(out_basepart)
                print("Saving plot results to {}".format(out_basepart))
                out_name = out_basepart + "_full_signal_filt.png"
                plot_long_lfp(lfp_odict.get_filt_signal(chan), out_name)
                graph_data = lfp_odict.get_filt_signal(chan).spectrum(
                    fmax=90,
                    db=False,
                    tr=False,
                    prefilt=False,
                    filtset=(10, 1.5, 90, "bandpass"),
                )
                fig = nc_plot.lfp_spectrum(graph_data)
                fig.savefig(out_basepart + "_spec.png")
                graph_data = lfp_odict.get_filt_signal(chan).spectrum(
                    fmax=90,
                    db=True,
                    tr=True,
                    prefilt=False,
                    filtset=(10, 1.5, 90, "bandpass"),
                )
                fig = nc_plot.lfp_spectrum_tr(graph_data)
                fig.savefig(out_basepart + "_tr_spec.png")
                close("all")

    if analysis_flags[4]:
        out_dirname = os.path.join(out_dir, plot_dir)
        print(
            "Caculating power results to save to {}".format(
                os.path.join(out_dirname, f"power_res_{res_name}.csv")
            )
        )
        results = OrderedDict()
        results["Names"] = [
            "VC Chan",
            "Delta VC",
            "Theta VC",
            "Beta VC",
            "Gamma VC",
            "Total VC",
            "CLA Chan",
            "Delta CLA",
            "Theta CLA",
            "Beta CLA",
            "Gamma CLA",
            "Total CLA",
        ]
        for fname in filenames:
            for key, val in channels.items():
                if key in fname:
                    chan_list = val
                    break
            else:
                raise ValueError("No key in {}, keys {}".format(fname, channels.keys()))
            lfp_odict = LfpODict(fname, chan_list, (True, 1, 90))

            if "green" in fname:
                lfp_odict.notch_filter(channels=["1", "2"])

            o_arr = np.zeros(12)
            if "green" in fname:
                o_arr[:6] = None
            for i, chan in enumerate(chan_list):
                if "green" in fname and i == 0:
                    continue
                start_idx = i * 6
                o_arr[start_idx] = chan
                window_sec = 1.3
                lfp = lfp_odict.get_filt_signal(chan)
                delta_power = lfp.bandpower(band=[1.5, 4], window_sec=window_sec)[
                    "bandpower"
                ]
                theta_power = lfp.bandpower(band=[5, 11], window_sec=window_sec)[
                    "bandpower"
                ]
                beta_power = lfp.bandpower(band=[12, 30], window_sec=window_sec)[
                    "bandpower"
                ]
                h_gamma_power = lfp.bandpower(band=[30, 90], window_sec=window_sec)[
                    "bandpower"
                ]
                o_arr[start_idx + 1 : start_idx + 5] = [
                    delta_power,
                    theta_power,
                    beta_power,
                    h_gamma_power,
                ]
                total_power = lfp.bandpower(band=[1, 90], window_sec=window_sec)[
                    "bandpower"
                ]
                o_arr[start_idx + 5] = total_power
            results[os.path.basename(fname)] = o_arr
        save_mixed_dict_to_csv(results, out_dirname, f"power_res_{res_name}.csv")


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
    res[2:] = mvl_shuffle(low_freq_lfp, high_freq_lfp, amp_norm=amp_norm, nshuffles=200)
    res_dict[os.path.basename(recording)] = res
    # out_name = os.path.join(
    #     out_dir,
    #     os.path.basename(recording) + "_{}_polar.png".format(channels))
    # pv, mvl = mean_vector_length(
    #     low_freq_lfp, high_freq_lfp, amp_norm=amp_norm, return_all=True)
    # plot_polar_coupling(pv, mvl, out_name, dpi=200)
    return res_dict


def power_calc(cfg1, cfg2):
    print(cfg1, cfg2)
    for cnfg, name in zip([cfg1, cfg2], ["first", "second"]):
        config = read_cfg(cnfg)
        args = parse_args()
        main(config, args, res_name=name)


if __name__ == "__main__":
    """Parse args and cfg and send to main."""
    here = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(here, "Configs", "lfp_synchrony.cfg")
    config = read_cfg(config_path)
    args = parse_args()
    main(config, args)

    config_path1 = os.path.join(here, "Configs", "lfp_synchrony_r1.cfg")
    config_path2 = os.path.join(here, "Configs", "lfp_synchrony_r2.cfg")
    # power_calc(config_path1, config_path2)
