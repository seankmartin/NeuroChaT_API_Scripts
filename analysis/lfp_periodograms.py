import os

from api_utils import read_cfg, parse_args, setup_logging, make_dir_if_not_exists, make_path_if_not_exists
from lfp_odict import LfpODict
import neurochat.nc_plot as nc_plot

import matplotlib.pyplot as plt


def main(fname):
    chans = [i for i in range(1, 17)]
    lfp_odict = LfpODict(
        fname, channels=chans, filt_params=(False, 1, 250))
    o_dir = os.path.join(
        os.path.dirname(fname), "LFP")
    make_dir_if_not_exists(o_dir)
    for i, (key, lfp) in enumerate(lfp_odict.get_signal().items()):
        graph_data = lfp.spectrum(
            ptype='psd', prefilt=False,
            db=False, tr=False,
            filtset=[10, 1.5, 40, 'bandpass'])
        fig = nc_plot.lfp_spectrum(graph_data)
        plt.ylim(0, 0.04)
        plt.xlim(0, 125)
        out_name = os.path.join(o_dir, key + "p.png")
        fig.savefig(out_name)


if __name__ == "__main__":
    fname = r"C:\Users\smartin5\Recordings\recording_example\010416b-LS3-50Hz10V5ms"
    main(fname)
