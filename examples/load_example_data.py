"""Can set up this file to provide example data for tests."""
import os
from neurochat.nc_data import NData


def load_data():
    dir = r'C:\Users\smartin5\recording_example'
    spike_file = os.path.join(dir, "010416b-LS3-50Hz10V5ms.2")
    pos_file = os.path.join(dir, "010416b-LS3-50Hz10V5ms_2.txt")
    lfp_file = os.path.join(dir, "010416b-LS3-50Hz10V5ms.eeg")
    unit_no = 7
    ndata = NData()
    ndata.set_spike_file(spike_file)
    ndata.set_spatial_file(pos_file)
    ndata.set_lfp_file(lfp_file)
    ndata.load()
    ndata.set_unit_no(unit_no)

    return ndata


def load_h5_data():
    data_dir = r'C:\Users\smartin5\Recordings\NC_eg'
    main_file = "040513_1.hdf5"
    spike_file = "/processing/Shank/6"
    pos_file = "/processing/Behavioural/Position"
    lfp_file = "/processing/Neural Continuous/LFP/eeg"
    unit_no = 3

    def m_file(x): return os.path.join(data_dir, main_file + x)
    ndata = NData()
    ndata.set_data_format(data_format='NWB')
    ndata.set_spatial_file(m_file(pos_file))
    ndata.set_spike_file(m_file(spike_file))
    ndata.set_lfp_file(m_file(lfp_file))
    ndata.load()
    ndata.set_unit_no(unit_no)

    return ndata


if __name__ == "__main__":
    print(load_h5_data())
