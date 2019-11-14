"""Plot all spatial cells in an Axona directory."""
from neurochat.nc_datacontainer import NDataContainer
import neurochat.nc_containeranalysis as nca


def main(dir):
    container = NDataContainer(load_on_fly=True)
    container.add_axona_files_from_dir(dir, recursive=True)
    container.setup()
    print(container.string_repr(True))
    nca.place_cell_summary(
        container, dpi=200, out_dirname="nc_spat_plots", output_format="pdf",
        burst_thresh=6, isi_bin_length=1, filter_place_cells=False)


if __name__ == "__main__":
    dir = r'C:\Users\smartin5\Recordings\recording_example'
    main(dir)
