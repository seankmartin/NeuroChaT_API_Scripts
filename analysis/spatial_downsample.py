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
        sample_bin_amt=[30, 30]):
    bin_size = sample_bin_amt
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


def chop_map(self, chop_edges, ftimes, pixel=3):
    """This is x_l, x_r, y_t, y_b."""
    x_l, x_r, y_t, y_b = np.array(chop_edges) * pixel
    x_r = max(self._pos_x) - x_r
    y_t = max(self._pos_y) - y_t
    in_range_x = np.logical_and(self._pos_x >= x_l, self._pos_x <= x_r)
    in_range_y = np.logical_and(self._pos_y >= y_b, self._pos_y <= y_t)

    spikeLoc = self.get_event_loc(ftimes)[1]
    spike_idxs = spikeLoc[0]
    spike_idxs_to_use = []

    sample_spatial_idx = np.where(np.logical_and(in_range_y, in_range_x))
    for i, val in enumerate(spike_idxs):
        if not np.any(sample_spatial_idx == val):
            spike_idxs_to_use.append(i)
    ftimes = ftimes[np.array(spike_idxs_to_use)]

    self._set_time(self._time[sample_spatial_idx])
    self._set_pos_x(self._pos_x[sample_spatial_idx] - x_l)
    self._set_pos_y(self._pos_y[sample_spatial_idx] - y_b)
    self._set_direction(self._direction[sample_spatial_idx])
    self._set_speed(self._speed[sample_spatial_idx])
    self.set_ang_vel(self._ang_vel[sample_spatial_idx])

    return ftimes


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
    xedges2 = other_spatial._xbound
    yedges2 = other_spatial._ybound

    spikeLoc = self.get_event_loc(ftimes, **kwargs)[1]
    posX = self._pos_x[np.logical_and(
        self.get_time() >= lim[0], self.get_time() <= lim[1])]
    posY = self._pos_y[np.logical_and(
        self.get_time() >= lim[0], self.get_time() <= lim[1])]

    new_set, spike_count = self.bin_downsample(
        ftimes, other_spatial, other_ftimes,
        final_bins=[
            np.append(yedges, yedges[-1] + np.mean(np.diff(yedges))),
            np.append(xedges, xedges[-1] + np.mean(np.diff(xedges)))],
        sample_bin_amt=[len(xedges2) + 1, len(yedges2) + 1])
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


def loc_auto_corr_down(self, ftimes, other_spatial, other_ftimes, **kwargs):
    """
    Calculates the two-dimensional correlation of firing map which is the
    map of the firing rate of the animal with respect to its location

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
    graph_data = {}

    minPixel = kwargs.get('minPixel', 20)
    pixel = kwargs.get('pixel', 3)

    if 'update' in kwargs.keys():
        del kwargs['update']
    placeData = self.downsample_place(
        ftimes, other_spatial, other_ftimes, update=False, **kwargs)

    fmap = placeData['smoothMap']
    fmap[np.isnan(fmap)] = 0
    leny, lenx = fmap.shape

    xshift = np.arange(-(lenx - 1), lenx)
    yshift = np.arange(-(leny - 1), leny)

    corrMap = np.zeros((yshift.size, xshift.size))

    for J, ysh in enumerate(yshift):
        for I, xsh in enumerate(xshift):
            if ysh >= 0:
                map1YInd = np.arange(ysh, leny)
                map2YInd = np.arange(leny - ysh)
            elif ysh < 0:
                map1YInd = np.arange(leny + ysh)
                map2YInd = np.arange(-ysh, leny)

            if xsh >= 0:
                map1XInd = np.arange(xsh, lenx)
                map2XInd = np.arange(lenx - xsh)
            elif xsh < 0:
                map1XInd = np.arange(lenx + xsh)
                map2XInd = np.arange(-xsh, lenx)
            map1 = fmap[tuple(np.meshgrid(map1YInd, map1XInd))]
            map2 = fmap[tuple(np.meshgrid(map2YInd, map2XInd))]
            if map1.size < minPixel:
                corrMap[J, I] = -1
            else:
                corrMap[J, I] = corr_coeff(map1, map2)

    graph_data['corrMap'] = corrMap
    graph_data['xshift'] = xshift * pixel
    graph_data['yshift'] = yshift * pixel

    return graph_data


def always_grid(self, ftimes, **kwargs):
    """
    Analysis of Grid cells characterised by formation of grid-like pattern
    of high activity in the firing-rate map

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
    tol = kwargs.get('angtol', 2)
    binsize = kwargs.get('binsize', 3)
    bins = np.arange(0, 360, binsize)

    graph_data = self.loc_auto_corr(ftimes, update=False, **kwargs)
    corrMap = graph_data['corrMap']
    corrMap[np.isnan(corrMap)] = 0
    xshift = graph_data['xshift']
    yshift = graph_data['yshift']

    pixel = np.int(np.diff(xshift).mean())

    ny, nx = corrMap.shape
    rpeaks = np.zeros(corrMap.shape, dtype=bool)
    cpeaks = np.zeros(corrMap.shape, dtype=bool)
    for j in np.arange(ny):
        rpeaks[j, extrema(corrMap[j, :])[1]] = True
    for i in np.arange(nx):
        cpeaks[extrema(corrMap[:, i])[1], i] = True
    ymax, xmax = find2d(np.logical_and(rpeaks, cpeaks))

    peakDist = np.sqrt((ymax - find(yshift == 0))**2 +
                       (xmax - find(xshift == 0))**2)
    sortInd = np.argsort(peakDist)
    ymax, xmax, peakDist = ymax[sortInd], xmax[sortInd], peakDist[sortInd]

    ymax, xmax, peakDist = (
        ymax[1:7], xmax[1:7], peakDist[1:7]) if ymax.size >= 7 else ([], [], [])
    theta = np.arctan2(yshift[ymax], xshift[xmax]) * 180 / np.pi
    theta[theta < 0] += 360
    sortInd = np.argsort(theta)
    ymax, xmax, peakDist, theta = (
        ymax[sortInd], xmax[sortInd], peakDist[sortInd], theta[sortInd])

    graph_data['ymax'] = yshift[ymax]
    graph_data['xmax'] = xshift[xmax]

    meanDist = peakDist.mean()
    X, Y = np.meshgrid(xshift, yshift)
    distMat = np.sqrt(X**2 + Y**2) / pixel

    _results["First Check"] = (len(ymax) == np.logical_and(
        peakDist > 0.75 * meanDist, peakDist < 1.25 * meanDist).sum())
    maskInd = np.logical_and(
        distMat > 0.5 * meanDist, distMat < 1.5 * meanDist)
    rotCorr = np.array([corr_coeff(rot_2d(corrMap, theta)[
                        maskInd], corrMap[maskInd]) for k, theta in enumerate(bins)])
    ramax, rimax, ramin, rimin = extrema(rotCorr)
    mThetaPk, mThetaTr = (np.diff(bins[rimax]).mean(), np.diff(
        bins[rimin]).mean()) if rimax.size and rimin.size else (None, None)
    graph_data['rimax'] = rimax
    graph_data['rimin'] = rimin
    graph_data['anglemax'] = bins[rimax]
    graph_data['anglemin'] = bins[rimin]
    graph_data['rotAngle'] = bins
    graph_data['rotCorr'] = rotCorr

    if mThetaPk is not None and mThetaTr is not None:
        isGrid = True if 60 - tol < mThetaPk < 60 + \
            tol and 60 - tol < mThetaTr < 60 + tol else False
    else:
        isGrid = False

    meanAlpha = np.diff(theta).mean()
    psi = theta[np.array([2, 3, 4, 5, 0, 1])] - theta
    psi[psi < 0] += 360
    meanPsi = psi.mean()

    _results['Is Grid'] = isGrid and 120 - tol < meanPsi < 120 + \
        tol and 60 - tol < meanAlpha < 60 + tol
    _results['Grid Mean Alpha'] = meanAlpha
    _results['Grid Mean Psi'] = meanPsi
    _results['Grid Spacing'] = meanDist * pixel
    # Difference between highest Pearson R at peaks and lowest at troughs
    _results['Grid Score'] = rotCorr[rimax].max() - \
        rotCorr[rimin].min()
    _results['Grid Orientation'] = theta[0]

    self.update_result(_results)
    return graph_data


def grid_down(self, ftimes, other_spatial, other_ftimes, **kwargs):
    """
    Analysis of Grid cells characterised by formation of grid-like pattern
    of high activity in the firing-rate map        

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
    tol = kwargs.get('angtol', 2)
    binsize = kwargs.get('binsize', 3)
    bins = np.arange(0, 360, binsize)

    graph_data = self.loc_auto_corr_down(
        ftimes, other_spatial, other_ftimes, update=False, **kwargs)
    corrMap = graph_data['corrMap']
    corrMap[np.isnan(corrMap)] = 0
    xshift = graph_data['xshift']
    yshift = graph_data['yshift']

    pixel = np.int(np.diff(xshift).mean())

    ny, nx = corrMap.shape
    rpeaks = np.zeros(corrMap.shape, dtype=bool)
    cpeaks = np.zeros(corrMap.shape, dtype=bool)
    for j in np.arange(ny):
        rpeaks[j, extrema(corrMap[j, :])[1]] = True
    for i in np.arange(nx):
        cpeaks[extrema(corrMap[:, i])[1], i] = True
    ymax, xmax = find2d(np.logical_and(rpeaks, cpeaks))

    peakDist = np.sqrt((ymax - find(yshift == 0))**2 +
                       (xmax - find(xshift == 0))**2)
    sortInd = np.argsort(peakDist)
    ymax, xmax, peakDist = ymax[sortInd], xmax[sortInd], peakDist[sortInd]

    ymax, xmax, peakDist = (
        ymax[1:7], xmax[1:7], peakDist[1:7]) if ymax.size >= 7 else ([], [], [])
    theta = np.arctan2(yshift[ymax], xshift[xmax]) * 180 / np.pi
    theta[theta < 0] += 360
    sortInd = np.argsort(theta)
    ymax, xmax, peakDist, theta = (
        ymax[sortInd], xmax[sortInd], peakDist[sortInd], theta[sortInd])

    graph_data['ymax'] = yshift[ymax]
    graph_data['xmax'] = xshift[xmax]

    meanDist = peakDist.mean()
    X, Y = np.meshgrid(xshift, yshift)
    distMat = np.sqrt(X**2 + Y**2) / pixel

    # if all of them are within tolerance(25%)
    # TODO check tol
    maskInd = np.logical_and(
        distMat > 0.5 * meanDist, distMat < 1.5 * meanDist)
    rotCorr = np.array([corr_coeff(rot_2d(corrMap, theta)[
                        maskInd], corrMap[maskInd]) for k, theta in enumerate(bins)])
    ramax, rimax, ramin, rimin = extrema(rotCorr)
    mThetaPk, mThetaTr = (np.diff(bins[rimax]).mean(), np.diff(
        bins[rimin]).mean()) if rimax.size and rimin.size else (None, None)
    graph_data['rimax'] = rimax
    graph_data['rimin'] = rimin
    graph_data['anglemax'] = bins[rimax]
    graph_data['anglemin'] = bins[rimin]
    graph_data['rotAngle'] = bins
    graph_data['rotCorr'] = rotCorr

    if mThetaPk is not None and mThetaTr is not None:
        isGrid = True if 60 - tol < mThetaPk < 60 + \
            tol and 60 - tol < mThetaTr < 60 + tol else False
    else:
        isGrid = False

    meanAlpha = np.diff(theta).mean()
    psi = theta[np.array([2, 3, 4, 5, 0, 1])] - theta
    psi[psi < 0] += 360
    meanPsi = psi.mean()

    _results["First Check"] = (len(ymax) == np.logical_and(
        peakDist > 0.75 * meanDist, peakDist < 1.25 * meanDist).sum())
    _results['Is Grid'] = isGrid and 120 - tol < meanPsi < 120 + \
        tol and 60 - tol < meanAlpha < 60 + tol
    _results['Grid Mean Alpha'] = meanAlpha
    _results['Grid Mean Psi'] = meanPsi
    _results['Grid Spacing'] = meanDist * pixel
    # Difference between highest Pearson R at peaks and lowest at troughs
    _results['Grid Score'] = rotCorr[rimax].max() - \
        rotCorr[rimin].min()
    _results['Grid Orientation'] = theta[0]

    self.update_result(_results)
    return graph_data


NSpatial.bin_downsample = bin_downsample
NSpatial.downsample_place = downsample_place
NSpatial.reverse_downsample = reverse_downsample
NSpatial.chop_map = chop_map
NSpatial.loc_auto_corr_down = loc_auto_corr_down
NSpatial.grid_down = grid_down
NSpatial.always_grid = always_grid


def random_down(
        spat1, ftimes1, spat2, ftimes2, keys,
        num_iters=50):
    results = {}
    output_dict = {}
    for key in keys:
        results[key] = np.zeros(shape=(num_iters))
    for i in range(num_iters):
        p_down_data = spat1.downsample_place(ftimes1, spat2, ftimes2)
        while p_down_data == -1:
            p_down_data = spat1.downsample_place(ftimes1, spat2, ftimes2)
        for key in keys:
            results[key][i] = spat1.get_results()[key]
        spat1._results.clear()
    output_dict = {}
    for key in keys:
        output_dict[key] = np.nanmean(results[key])
    return output_dict, p_down_data


if __name__ == "__main__":

    # Set up the recordings
    spatial = NSpatial()
    fname = r"D:\SubRet_recordings_imaging\muscimol_data\CanCSR8_muscimol\05102018\s3_after_smallsq\05102018_CanCSR8_smallsq_10_3_3.txt"
    spatial.set_filename(fname)
    spatial.load()

    spike = NSpike()
    fname = r"D:\SubRet_recordings_imaging\muscimol_data\CanCSR8_muscimol\05102018\s3_after_smallsq\05102018_CanCSR8_smallsq_10_3.3"
    spike.set_filename(fname)
    spike.load()
    spike.set_unit_no(1)

    spatial2 = NSpatial()
    fname = r"D:\SubRet_recordings_imaging\muscimol_data\CanCSR8_muscimol\05102018\s4_big sq\05102018_CanCSR8_bigsq_10_4_3.txt"
    spatial2.set_filename(fname)
    spatial2.load()

    spike2 = NSpike()
    fname = r"D:\SubRet_recordings_imaging\muscimol_data\CanCSR8_muscimol\05102018\s4_big sq\05102018_CanCSR8_bigsq_10_4.3"
    spike2.set_filename(fname)
    spike2.load()
    spike2.set_unit_no(6)
    ftimes = spike.get_unit_stamp()
    ftimes2 = spike2.get_unit_stamp()

    # Set up the keys
    keys = ["Spatial Coherence", "Spatial Skaggs", "Spatial Sparsity"]

    # TODO check in Raju thesis why there is tol?
    print("Grid stuff")
    spatial._results.clear()
    spatial.always_grid(ftimes)
    print(spatial._results)
    spatial._results.clear()

    spatial.hd_rate(ftimes)
    print(spatial._results)
    spatial._results.clear()
    # ftimes = spatial.chop_map([3, 3, 3, 3], spike.get_unit_stamp())
    # p_data = spatial.place(ftimes)
    # fig = nc_plot.loc_firing(p_data)
    # fig.savefig("normal_chop.png")
    # # By bonnevie this is likely stable, what about stability?

    p_data = spatial.place(spike.get_unit_stamp())
    fig = nc_plot.loc_firing(p_data)
    fig.savefig("normal.png")
    for key in keys:
        print("A: {} - {}".format(key, spatial.get_results()[key]))
    spatial._results.clear()

    p_down_data = spatial.downsample_place(
        ftimes, spatial, ftimes)
    fig = nc_plot.loc_firing(p_down_data)
    fig.savefig("down_v_self.png")

    res, data = random_down(spatial, ftimes, spatial, ftimes, keys)
    print("A_A: {}".format(res))

    p_data = spatial2.place(spike2.get_unit_stamp())
    for key in keys:
        print("B: ", spatial2.get_results()[key])
    fig = nc_plot.loc_firing(p_data)
    fig.savefig("normal2.png")

    res, p_down_data = random_down(spatial, ftimes, spatial2, ftimes2, keys)
    fig = nc_plot.loc_firing(p_down_data)
    print("A_B: {}".format(res))
    fig.savefig("down_v_2.png")

    res, p_down_data = random_down(spatial2, ftimes2, spatial, ftimes, keys)
    print("B_A: {}".format(res))
    fig = nc_plot.loc_firing(p_down_data)
    fig.savefig("down_v_3.png")

    spatial._results.clear()
    for i in range(10):
        spatial.grid_down(ftimes, spatial, ftimes)
        print(spatial._results)
        spatial._results.clear()
