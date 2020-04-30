import numpy as np

from neurochat.nc_spatial import NSpatial
from neurochat.nc_spike import NSpike
from neurochat.nc_utils import histogram


def bin_downsample(self, ftimes):
    set_array = np.zeros(shape=(len(self._pos_x), 4))
    set_array[:, 0] = self._pos_x
    set_array[:, 1] = self._pos_y
    spikes_in_bins = histogram(ftimes, bins=self.get_time())[0]
    set_array[:, 2] = spikes_in_bins
    set_array[:, 3] = self._time

    pos_hist = np.histogram2d(
        set_array[:, 0], set_array[:, 1], 10)
    pos_locs_x = np.searchsorted(
        pos_hist[1][1:], set_array[:, 0], side='left')
    pos_locs_y = np.searchsorted(
        pos_hist[2][1:], set_array[:, 1], side='left')
    new_set = np.zeros(shape=(int(np.sum(pos_hist[0])), 4))
    count = 0

    for i in range(pos_hist[0].shape[0]):
        for j in range(pos_hist[0].shape[1]):
            amount = int(pos_hist[0][i, j])
            subset = np.nonzero(np.logical_and(
                pos_locs_x == i, pos_locs_y == j))[0]
            new_sample_idxs = np.random.choice(subset, amount)
            new_samples = set_array[new_sample_idxs]
            new_set[count:count + amount] = new_samples
            count += amount
    print(np.histogram2d(
        new_set[:, 0], new_set[:, 1], [pos_hist[1], pos_hist[2]])[0])
    return new_set


NSpatial.bin_downsample = bin_downsample

if __name__ == "__main__":
    spatial = NSpatial()
    fname = r"D:\SubRet_recordings_imaging\muscimol_data\CanCSR8_muscimol\05102018\s1_smallsq_before\05102018_CanCSR8_smallsq_10_1_3.txt"
    spatial.set_filename(fname)
    spatial.load()

    spike = NSpike()
    fname = r"D:\SubRet_recordings_imaging\muscimol_data\CanCSR8_muscimol\05102018\s1_smallsq_before\05102018_CanCSR8_smallsq_10_1.3"
    spike.set_filename(fname)
    spike.load()
    spike.set_unit_no(1)

    spatial.bin_downsample(spike.get_unit_stamp())
