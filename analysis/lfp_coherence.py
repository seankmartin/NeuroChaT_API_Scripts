from scipy.signal import hilbert
import numpy as np

from neurochat.nc_utils import butter_filter
from neurochat.nc_circular import CircStat


def mean_vector_length(low_freq_lfp, high_freq_lfp):
    """Compute the mean vector length from Hulsemann et al. 2019"""
    amplitude = split_into_amp_phase(high_freq_lfp)[0]
    phase = split_into_amp_phase(low_freq_lfp, deg=False)[1]
    if amplitude.size != phase.size:
        print("Amplitude and phase must have same size in MVL")
        exit(-1)
    polar_vectors = np.multiply(amplitude, np.exp(1j * phase))
    res_vector = np.sum(polar_vectors)
    mvl1 = np.abs(res_vector) / amplitude.size
    cs = CircStat()
    cs.set_rho(amplitude)
    cs.set_theta(np.rad2deg(phase))
    res = cs.calc_stat()
    mvl2 = res["resultant"]
    mvl3 = np.abs(res_vector) / np.sum(amplitude)  # this way matches circ stat
    return mvl1, mvl2, mvl3


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
    print(mean_vector_length(low_freq_lfp, high_freq_lfp))
