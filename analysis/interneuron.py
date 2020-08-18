import numpy as np


def is_pyramidal(wave_width, spike_rate, mean_autocorr):
    """The values should be in msec."""
    # From https://www.jneurosci.org/content/19/1/274
    # Pyramidal
    # Width 0.44 +- 0.005 ms
    # Mean autocorrelation 7.1 +- 0.12 ms
    # Spike rate 1.4 +- 0.01 Hz
    # Interneuron
    # Width 0.24 +- 0.01 ms
    # Mean autocorrelation 12.0 +- 0.17 ms
    # Spike rate 13.0 +- 1.62
    wave_is_inter = (wave_width < 0.2)
    rate_is_inter = (spike_rate > 5)
    autocorr_is_inter = (mean_autocorr > 10)

    if (int(wave_is_inter) + int(rate_is_inter) + int(autocorr_is_inter)) >= 2:
        return False
    else:
        return True


def cell_type(ndata):
    results = ndata.get_results()
    if "Mean Width" not in results:
        ndata.wave_property()
    mean_width = results["Mean width"] / 1000
    ndata._results["Mean width"] = mean_width
    spike_rate = results["Mean Spiking Freq"]
    isi_data = ndata.isi_corr(bound=[-20, 20])
    isi_corr = isi_data["isiCorr"] / results["Number of Spikes"]
    all_bins = isi_data['isiAllCorrBins']
    centre = np.flatnonzero(all_bins == 0)[0]
    bin_centres = [
        (all_bins[i + 1] + all_bins[i]) / 2 for i in range(len(all_bins) - 1)]
    autocorr_mean = (
        np.sum(bin_centres[centre:] * isi_corr[centre:]) / np.sum(isi_corr[centre:]))
    ndata._results["AC mean"] = autocorr_mean

    return is_pyramidal(mean_width, spike_rate, autocorr_mean)
