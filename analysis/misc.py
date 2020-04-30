import os

import matplotlib.pyplot as plt
import numpy as np

from neurochat.nc_utils import get_all_files_in_dir


def visualise_spacing(N=61, start=5, stop=10000):
    """Plots a visual representation logspacing"""
    # This is equivalent to np.exp(np.linspace)
    x1 = np.logspace(np.log10(start), np.log10(stop), N, base=10)
    y = np.zeros(N)
    plt.plot(x1, y, 'o')
    plt.ylim([-0.5, 1])
    print(x1)
    plt.show()


def get_good_cells(directory):
    with open("result.csv", "w") as out:
        out.write("Location, Unit\n")
        files = get_all_files_in_dir(directory, return_absolute=False)
        good_cells = []
        for f in files:
            parts = os.path.splitext(f)[0].split("--")
            last_bit = parts[-1]
            last_split = last_bit.split("_")
            name = "_".join(last_split[:-2])
            name = name + "." + last_split[-2]
            unit = int(last_split[-1])
            path = os.path.join(*parts[:-1], name)
            out.write("{}, {}\n".format(path, unit))
            good_cells.append([path, unit])
    return good_cells


if __name__ == "__main__":
    directory = r"D:\ATNx_CA1\final_plots\good"
    get_good_cells(directory)
