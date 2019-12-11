from collections import OrderedDict
import os

import numpy as np
import matplotlib.pyplot as plt

from neurochat.nc_data import NData
from neurochat.nc_datacontainer import NDataContainer
from neurochat.nc_plot import loc_firing_and_place
from api_utils import make_dir_if_not_exists

if __name__ == "__main__":
    in_dir = r"C:\Users\smartin5\Recordings\ER\Odors"
    container = NDataContainer(load_on_fly=True)
    container.add_axona_files_from_dir(in_dir, True, True)
    container.setup()
    out_dir = os.path.join(in_dir, "nc_results")
    make_dir_if_not_exists(out_dir)
    for i, data in enumerate(container):
        gdata = data.spatial.corner_place_map(data.spike.get_unit_stamp())
        data.update_results(data.spatial.get_results())
        fig = loc_firing_and_place(
            gdata, style="digitized", colormap="default")
        name = container.get_name_at_idx(i, "png", base_dir=in_dir)
        print("Plotting data to {}".format(name))
        fig.savefig(name, dpi=200)
        plt.close("all")
