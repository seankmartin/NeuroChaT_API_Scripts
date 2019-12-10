from copy import deepcopy

from scipy.signal import hilbert
from scipy.stats import norm
import numpy as np

from neurochat.nc_utils import butter_filter
from neurochat.nc_circular import CircStat


def mean_vector_length(low_freq_lfp, high_freq_lfp, amp_norm=True):
    """
    Compute the mean vector length from Hulsemann et al. 2019

    If amp_norm is true, use the sum of the amplitudes to normalise,
    as opposed to the number of observations.
    """
    amplitude = split_into_amp_phase(high_freq_lfp)[0]
    phase = split_into_amp_phase(low_freq_lfp, deg=False)[1]
    if amplitude.size != phase.size:
        raise ValueError(
            "Amp and phase: {} {} elements, equal size needed in MVL".format(
                amplitude.size, phase.size))
    # This could also be computed using CircStat
    norm = np.sum(amplitude) if amp_norm else amplitude.size
    polar_vectors = np.multiply(amplitude, np.exp(1j * phase))
    res_vector = np.sum(polar_vectors)
    mvl = np.abs(res_vector) / norm
    return mvl


def mvl_shuffle(low_freq_lfp, high_freq_lfp, amp_norm=True, nshuffles=200):
    """Compute a shuffled distribution from Hulsemann et al. 2019"""
    samples = high_freq_lfp.get_samples()
    new_lfp = deepcopy(high_freq_lfp)
    observed_mvl = mean_vector_length(low_freq_lfp, high_freq_lfp)
    shuffled_mvl = np.zeros(shape=(nshuffles))
    for i in range(len(shuffled_mvl)):
        sample_idx = int(np.floor(
            (low_freq_lfp.get_total_samples() + 1) * np.random.random_sample()))
        reversed_arr1 = samples[0:sample_idx][::-1]
        reversed_arr2 = samples[sample_idx:samples.size][::-1]
        permuted_amp_time = np.concatenate(
            [reversed_arr1, reversed_arr2], axis=None)
        new_lfp._set_samples(permuted_amp_time)
        mvl = mean_vector_length(low_freq_lfp, new_lfp, amp_norm=amp_norm)
        shuffled_mvl[i] = mvl
    z_val = (observed_mvl - np.mean(shuffled_mvl)) / np.std(shuffled_mvl)
    mvl95 = np.percentile(shuffled_mvl, 95)
    p_val = norm.sf(z_val)

    return observed_mvl, mvl95, z_val, p_val


def split_into_amp_phase(lfp, deg=False):
    """It is assumed that the lfp signal passed in is already filtered."""
    lfp_samples = lfp.get_samples()
    complex_lfp = hilbert(lfp_samples)
    phase = np.angle(complex_lfp, deg=deg)
    amplitude = np.abs(complex_lfp)
    return amplitude, phase


if __name__ == "__main__":
    """Test out these functions."""
    recording = r"C:\Users\smartin5\Recordings\ER\LFP-cla-V2L\LFP-cla-V2L-ctrl\05112019-white\05112019-white-D"
    channels = [1, 5]
    from lfp_odict import LfpODict
    lfp_odict = LfpODict(recording, channels)
    low_freq_lfp = lfp_odict.filter(5, 11).get("5")  # Theta range
    # Slow gamma is 30-55, fast gamma is 65-90
    high_freq_lfp = lfp_odict.filter(30, 55).get("1")
    amp_norm = True
    # print(mean_vector_length(low_freq_lfp, high_freq_lfp, amp_norm=amp_norm))
    print(mvl_shuffle(
        low_freq_lfp, high_freq_lfp, amp_norm=amp_norm, nshuffles=1000))
