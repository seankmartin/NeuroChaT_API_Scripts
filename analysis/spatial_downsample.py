from collections import OrderedDict as oDict

from neurochat.nc_utils import chop_edges, corr_coeff, extrema,\
    find, find2d, find_chunk, histogram, histogram2d, \
    linfit, residual_stat, rot_2d, smooth_1d, smooth_2d, \
    centre_of_mass, find_true_ranges

import numpy as np

from neurochat.nc_spatial import NSpatial
from neurochat.nc_spike import NSpike
from neurochat.nc_utils import histogram
import neurochat.nc_plot as nc_plot


def bin_downsample(
        self, ftimes, other_spatial, other_ftimes, final_bins,
        initial_bin_size=30):
    bin_size = initial_bin_size
    set_array = np.zeros(shape=(len(self._pos_x), 4), dtype=np.float64)
    set_array[:, 0] = self._pos_x
    set_array[:, 1] = self._pos_y
    spikes_in_bins = histogram(ftimes, bins=self.get_time())[0]
    set_array[:, 2] = spikes_in_bins
    set_array[:, 3] = self._time

    pos_hist = np.histogram2d(
        set_array[:, 0], set_array[:, 1], bin_size)
    pos_locs_x = np.searchsorted(
        pos_hist[1][1:], set_array[:, 0], side='left')
    pos_locs_y = np.searchsorted(
        pos_hist[2][1:], set_array[:, 1], side='left')

    set_array1 = np.zeros(shape=(len(other_spatial._pos_x), 2))
    set_array1[:, 0] = other_spatial._pos_x
    set_array1[:, 1] = other_spatial._pos_y
    # spikes_in_bins = histogram(other_ftimes, bins=other_spatial.get_time())[0]
    # set_array1[:, 2] = spikes_in_bins
    # set_array1[:, 3] = other_spatial._time
    pos_hist1 = np.histogram2d(
        set_array1[:, 0], set_array1[:, 1], bin_size)
    # pos_locs_x1 = np.searchsorted(
    #     pos_hist1[1][1:], set_array1[:, 0], side='left')
    # pos_locs_y1 = np.searchsorted(
    #     pos_hist1[2][1:], set_array1[:, 1], side='left')

    new_set = np.zeros(shape=(int(np.sum(pos_hist1[0])), 4))
    count = 0

    for i in range(pos_hist[0].shape[0]):
        for j in range(pos_hist[0].shape[1]):
            amount1 = int(pos_hist1[0][i, j])
            amount2 = int(pos_hist[0][i, j])
            amount = min(amount1, amount2)
            subset = np.nonzero(np.logical_and(
                pos_locs_x == i, pos_locs_y == j))[0]
            if len(subset) > amount2:
                subset = subset[:amount2]
            elif len(subset) == 0:
                continue
            new_sample_idxs = np.random.choice(subset, amount)
            new_samples = set_array[new_sample_idxs]
            new_set[count:count + amount] = new_samples
            count += amount
    # print(np.histogram2d(
    #     new_set[:, 0], new_set[:, 1], [pos_hist[1], pos_hist[2]])[0])
    spike_count = np.histogram2d(
        new_set[:, 1], new_set[:, 0], [final_bins[0], final_bins[1]],
        weights=new_set[:, 2])[0]
    return new_set, spike_count


def reverse_downsample(self, ftimes, other_spatial, other_ftimes, **kwargs):
    return other_spatial.downsample_place(
        other_ftimes, self, ftimes, **kwargs)


def downsample_place(self, ftimes, other_spatial, other_ftimes, **kwargs):
    """
    Calculates the two-dimensional firing rate of the unit with respect to
    the location of the animal in the environment. This is called Firing map.

    Specificity indices are measured to assess the quality of location-specific firing of the unit.

    This method also plot the events of spike occurring superimposed on the
    trace of the animal in the arena, commonly known as Spike Plot.

    Parameters
    ----------
    ftimes : ndarray
        Timestamps of the spiking activity of a unit
    **kwargs
        Keyword arguments

    Returns
    -------
    dict
        Graphical data of the analysis
    """

    _results = oDict()
    graph_data = {}
    update = kwargs.get('update', True)
    pixel = kwargs.get('pixel', 3)
    chop_bound = kwargs.get('chop_bound', 5)
    filttype, filtsize = kwargs.get('filter', ['b', 5])
    lim = kwargs.get('range', [0, self.get_duration()])
    brAdjust = kwargs.get('brAdjust', True)
    thresh = kwargs.get('fieldThresh', 0.2)
    required_neighbours = kwargs.get('minPlaceFieldNeighbours', 9)
    smooth_place = kwargs.get('smoothPlace', False)
    # Can pass another NData object to estimate the border from
    # Can be useful in some cases, such as when the animal
    # only explores a subset of the arena.
    separate_border_data = kwargs.get(
        "separateBorderData", None)

    # xedges = np.arange(0, np.ceil(np.max(self._pos_x)), pixel)
    # yedges = np.arange(0, np.ceil(np.max(self._pos_y)), pixel)

    # Update the border to match the requested pixel size
    if separate_border_data is not None:
        self.set_border(
            separate_border_data.calc_border(**kwargs))
        times = self._time
        lower, upper = (times.min(), times.max())
        new_times = separate_border_data._time
        sample_spatial_idx = (
            (new_times <= upper) & (new_times >= lower)).nonzero()
        self._border_dist = self._border_dist[sample_spatial_idx]
    else:
        self.set_border(self.calc_border(**kwargs))

    xedges = self._xbound
    yedges = self._ybound

    spikeLoc = self.get_event_loc(ftimes, **kwargs)[1]
    posX = self._pos_x[np.logical_and(
        self.get_time() >= lim[0], self.get_time() <= lim[1])]
    posY = self._pos_y[np.logical_and(
        self.get_time() >= lim[0], self.get_time() <= lim[1])]

    new_set, spike_count = self.bin_downsample(
        ftimes, other_spatial, other_ftimes,
        final_bins=[
            np.append(yedges, yedges[-1] + np.mean(np.diff(yedges))),
            np.append(xedges, xedges[-1] + np.mean(np.diff(xedges)))]
    )
    posY = new_set[:, 1]
    posX = new_set[:, 0]

    tmap, yedges, xedges = histogram2d(posY, posX, yedges, xedges)
    if tmap.shape != spike_count.shape:
        print(tmap.shape)
        print(spike_count.shape)
        raise ValueError("Time map does not match firing map")

    tmap /= self.get_sampling_rate()

    ybin, xbin = tmap.shape
    xedges = np.arange(xbin) * pixel
    yedges = np.arange(ybin) * pixel

    fmap = np.divide(spike_count, tmap, out=np.zeros_like(
        spike_count), where=tmap != 0)

    if brAdjust:
        nfmap = fmap / fmap.max()
        if np.sum(np.logical_and(nfmap >= 0.2, tmap != 0)) >= 0.8 * nfmap[tmap != 0].flatten().shape[0]:
            back_rate = np.mean(
                fmap[np.logical_and(nfmap >= 0.2, nfmap < 0.4)])
            fmap -= back_rate
            fmap[fmap < 0] = 0

    if filttype is not None:
        smoothMap = smooth_2d(fmap, filttype, filtsize)
    else:
        smoothMap = fmap

    if smooth_place:
        pmap = smoothMap
    else:
        pmap = fmap

    pmap[tmap == 0] = None
    pfield, largest_group = NSpatial.place_field(
        pmap, thresh, required_neighbours)
    # if largest_group == 0:
    #     if smooth_place:
    #         info = "where the place field was calculated from smoothed data"
    #     else:
    #         info = "where the place field was calculated from raw data"
    #     logging.info(
    #         "Lack of high firing neighbours to identify place field " +
    #         info)
    centroid = NSpatial.place_field_centroid(pfield, pmap, largest_group)
    # centroid is currently in co-ordinates, convert to pixels
    centroid = centroid * pixel + (pixel * 0.5)
    # flip x and y
    centroid = centroid[::-1]

    p_shape = pfield.shape
    maxes = [xedges.max(), yedges.max()]
    scales = (
        maxes[0] / p_shape[1],
        maxes[1] / p_shape[0])
    co_ords = np.array(np.where(pfield == largest_group))
    boundary = [[None, None], [None, None]]
    for i in range(2):
        j = (i + 1) % 2
        boundary[i] = (
            co_ords[j].min() * scales[i],
            np.clip((co_ords[j].max() + 1) * scales[i], 0, maxes[i]))
    inside_x = (
        (boundary[0][0] <= spikeLoc[0]) &
        (spikeLoc[0] <= boundary[0][1]))
    inside_y = (
        (boundary[1][0] <= spikeLoc[1]) &
        (spikeLoc[1] <= boundary[1][1]))
    co_ords = np.nonzero(np.logical_and(inside_x, inside_y))

    if update:
        _results['Spatial Skaggs'] = self.skaggs_info(fmap, tmap)
        _results['Spatial Sparsity'] = self.spatial_sparsity(fmap, tmap)
        _results['Spatial Coherence'] = np.corrcoef(
            fmap[tmap != 0].flatten(), smoothMap[tmap != 0].flatten())[0, 1]
        _results['Found strong place field'] = (largest_group != 0)
        _results['Place field Centroid x'] = centroid[0]
        _results['Place field Centroid y'] = centroid[1]
        _results['Place field Boundary x'] = boundary[0]
        _results['Place field Boundary y'] = boundary[1]
        _results['Number of Spikes in Place Field'] = co_ords[0].size
        _results['Percentage of Spikes in Place Field'] = co_ords[0].size * \
            100 / ftimes.size
        self.update_result(_results)

    smoothMap[tmap == 0] = None

    graph_data['posX'] = posX
    graph_data['posY'] = posY
    graph_data['fmap'] = fmap
    graph_data['smoothMap'] = smoothMap
    graph_data['firingMap'] = fmap
    graph_data['tmap'] = tmap
    graph_data['xedges'] = xedges
    graph_data['yedges'] = yedges
    graph_data['spikeLoc'] = spikeLoc
    graph_data['placeField'] = pfield
    graph_data['largestPlaceGroup'] = largest_group
    graph_data['placeBoundary'] = boundary
    graph_data['indicesInPlaceField'] = co_ords
    graph_data['centroid'] = centroid

    return graph_data


NSpatial.bin_downsample = bin_downsample
NSpatial.downsample_place = downsample_place
NSpatial.reverse_downsample = reverse_downsample

if __name__ == "__main__":
    spatial = NSpatial()
    fname = r"D:\SubRet_recordings_imaging\muscimol_data\CanCSR8_muscimol\05102018\s3_after_smallsq\05102018_CanCSR8_smallsq_10_3_3.txt"
    spatial.set_filename(fname)
    spatial.load()

    spike = NSpike()
    fname = r"D:\SubRet_recordings_imaging\muscimol_data\CanCSR8_muscimol\05102018\s3_after_smallsq\05102018_CanCSR8_smallsq_10_3.3"
    spike.set_filename(fname)
    spike.load()
    spike.set_unit_no(1)

    p_data = spatial.place(spike.get_unit_stamp())
    fig = nc_plot.loc_firing(p_data)
    fig.savefig("normal.png")
    # By bonnevie this is likely stable, what about stability?
    print("A: ", spatial.get_results()['Spatial Coherence'])
    spatial._results.clear()

    p_down_data = spatial.downsample_place(
        spike.get_unit_stamp(), spatial, spike.get_unit_stamp())
    fig = nc_plot.loc_firing(p_down_data)
    fig.savefig("down_v_self.png")

    skaggs = np.zeros(shape=(50))
    for i in range(50):
        p_down_data = spatial.downsample_place(
            spike.get_unit_stamp(), spatial, spike.get_unit_stamp())
        skaggs[i] = spatial.get_results()['Spatial Coherence']
        spatial._results.clear()
    print("A_A: ", np.mean(skaggs))

    spatial2 = NSpatial()
    fname = r"D:\SubRet_recordings_imaging\muscimol_data\CanCSR8_muscimol\05102018\s4_big sq\05102018_CanCSR8_bigsq_10_4_3.txt"
    spatial2.set_filename(fname)
    spatial2.load()

    spike2 = NSpike()
    fname = r"D:\SubRet_recordings_imaging\muscimol_data\CanCSR8_muscimol\05102018\s4_big sq\05102018_CanCSR8_bigsq_10_4.3"
    spike2.set_filename(fname)
    spike2.load()
    spike2.set_unit_no(6)

    p_data = spatial2.place(spike.get_unit_stamp())
    print("B: ", spatial2.get_results()['Spatial Coherence'])
    fig = nc_plot.loc_firing(p_data)
    fig.savefig("normal2.png")

    skaggs = np.zeros(shape=(50))
    for i in range(50):
        p_down_data = spatial.downsample_place(
            spike.get_unit_stamp(), spatial2, spike2.get_unit_stamp())
        skaggs[i] = spatial.get_results()['Spatial Coherence']
        spatial._results.clear()
    print("A_B: ", np.mean(skaggs))
    fig = nc_plot.loc_firing(p_down_data)
    fig.savefig("down_v_2.png")

    skaggs = np.zeros(shape=(50))
    for i in range(50):
        p_down_data = spatial.reverse_downsample(
            spike.get_unit_stamp(), spatial2, spike2.get_unit_stamp())
        skaggs[i] = spatial2.get_results()['Spatial Coherence']
        spatial2._results.clear()
    print("B_A: ", np.mean(skaggs))
    fig = nc_plot.loc_firing(p_down_data)
    fig.savefig("down_v_3.png")
