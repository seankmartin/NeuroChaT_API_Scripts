import os
import json

from api_utils import read_cfg, parse_args, setup_logging, make_dir_if_not_exists
from lfp_plot import plot_lfp

from neurochat.nc_utils import get_all_files_in_dir
from lfp_odict import LfpODict


def main(cfg, args, **kwargs):
    in_dir = cfg.get("Setup", "in_dir")
    out_dir = cfg.get("Output", "out_dirname")
    re_filter = cfg.get("Setup", "regex_filter")
    re_filter = None if re_filter == "None" else re_filter
    analysis_flags = json.loads(config.get("Setup", "analysis_flags"))

    setup_logging(in_dir)
    filenames = get_all_files_in_dir(
        in_dir, ext=".set", recursive=True,
        verbose=True, re_filter=re_filter, case_sensitive_ext=True)
    filenames = [fname[:-4] for fname in filenames]
    if len(filenames) == 0:
        print("No set files found for analysis!")
        exit(-1)

    for fname in filenames:
        # TODO get the correct channels here based on the filename
        channels = [1, 2, 16, 17, 31, 32]
        # TODO can setup filtering here
        lfp_odict = LfpODict(fname, channels=channels)
        o_dir = os.path.join(in_dir, out_dir, os.path.basename(fname))

        # Plot signal on each loaded channel
        if analysis_flags[0]:
            r = json.loads(config.get("LFP", "plot_time"))
            seg_len = float(config.get("LFP", "plot_seg_length"))
            make_dir_if_not_exists(o_dir)
            plot_lfp(
                o_dir, lfp_odict.get_signal(),
                in_range=r, segment_length=seg_len, dpi=100)


if __name__ == "__main__":
    """Parse args and cfg and send to main."""
    here = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(here, "Configs", "lfp_synchrony.cfg")
    config = read_cfg(config_path)
    args = parse_args()
    main(config, args)
