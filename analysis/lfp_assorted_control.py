"""Create some figures and stats using LFP signals."""
from pathlib import Path

from lfp_odict import LfpODict
from lfp_plot import plot_lfp, plot_lfp_sig


def main(path1, path2, r, seg_len, channel_to_plot):
    # Set up the paths
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)

    theta_band = [5, 11]
    plot_ext = "png"
    sr = 250  # 250 Hz sample rate

    lfps = [path1, path2]
    types_ = ["CTRL", "D-SER"]
    for lfp_path, type_ in zip(lfps, types_):
        b_name = lfp_path.name
        lfp_odict_l2 = LfpODict(
            str(lfp_path),
            channels="all",
            filt_params=(True, theta_band[0], theta_band[1]),
        )

        sigs = [
            lfp_odict_l2.get_filt_signal(key=channel_to_plot),
            lfp_odict_l2.get_signal(key=channel_to_plot),
        ]
        names = [f"{b_name}_{type_}_L2_theta", f"{b_name}_{type_}_L2_unfiltered"]

        for sig, name in zip(sigs, names):
            if channel_to_plot is None:
                plot_lfp(
                    output_dir,
                    sig,
                    in_range=r,
                    segment_length=seg_len,
                    dpi=400,
                    ext=plot_ext,
                    start_name=name,
                )
            else:
                out_name = output_dir / (name + "." + plot_ext)
                plot_lfp_sig(sig.get_samples()[r[0] * sr : r[1] * sr], out_name)


if __name__ == "__main__":
    main_path = Path("D:\Emanuela Rizzello data")
    lfp_l2_ctrl_path = main_path / "to 09-2019" / "09072019-bt" / "09072019-bt-L2"
    lfp_l2_dser_path = main_path / "to 09-2019" / "10072019-bt" / "10072019-bt-L2"
    r = [20, 21]  # 1 seconds long LFP
    seg_len = 1  # 2 second long segemnts
    channel_to_plot = 1  # Or select a number after checking
    main(lfp_l2_ctrl_path, lfp_l2_dser_path, r, seg_len, channel_to_plot)

    lfp_l2_ctrl_path = (
        main_path / "to 09-2019" / "29072019-bt" / "29072019-bt-last30min-LFP"
    )
    lfp_l2_dser_path = (
        main_path / "to 09-2019" / "30072019-bt" / "30072019-bt-last30min-LFP-DSER"
    )
    r = [20, 40]  # 20 seconds long LFP
    seg_len = 20  # 20 second long segemnts
    channel_to_plot = None  # Use None to check the channels
    main(lfp_l2_ctrl_path, lfp_l2_dser_path, r, seg_len, channel_to_plot)